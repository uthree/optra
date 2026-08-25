//! A software rasteriser.
//!
//! There is a good reason not to use the GPU that is already in the process for
//! this. The harness this feeds asserts millimetre figures, and a GPU render
//! depends on the driver, the vendor and the adapter that happened to be
//! picked; two machines running the same test would be measuring against two
//! slightly different pictures, and a threshold tight enough to be worth having
//! would fail on somebody's laptop. Everything here is `f64` arithmetic in a
//! fixed order, so the same input produces the same pixels everywhere.
//!
//! It is also the reason the renderer projects through [`Camera::project`]
//! rather than through a projection matrix of its own: the pixels and the
//! ground truth the harness compares them against then come from one piece of
//! code, including the lens distortion. A renderer with its own idea of the
//! lens would let a distortion bug cancel itself out.
//!
//! One approximation follows from that and is worth naming, because it is
//! wrong in principle. Only the three corners of a triangle go through the
//! lens; the edges between them are drawn straight. A straight line in the
//! world is straight in a pinhole image and *curved* in a distorted one, so a
//! rasterised edge is off by however far the real one bends.
//!
//! Measured on the four cameras this harness uses, as the distance from the
//! projected midpoint of an edge to the straight line joining its projected
//! ends — exactly zero for a pinhole, so what is left is the distortion:
//!
//! - a limb segment, twelve centimetres: at most 0.014 px
//! - a floor tile edge, half a metre: at most 0.14 px
//! - a wall panel, two and a half metres: at most 1.23 px
//!
//! The body is built from the first of those, so the thing being measured is
//! unaffected at four decimal places. Subdividing the room would fix the third
//! and cost frame time to make a wall a pixel straighter, which is not a trade
//! worth taking; it is written down instead.

use std::path::Path;

use anyhow::{Context, Result};
use nalgebra::{Point3, Vector3};

use crate::geometry::camera::Camera;
use crate::infer::traits::ImageView;
use crate::sim::Rng;
use crate::sim::mesh::Mesh;

/// Nothing closer than this to the camera is drawn, in metres.
const NEAR: f64 = 0.02;

/// A rendered RGB8 image.
#[derive(Debug, Clone)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    /// Tightly packed RGB8, `width * height * 3` bytes.
    pub rgb: Vec<u8>,
}

impl Image {
    pub fn view(&self) -> ImageView<'_> {
        ImageView::new(self.width, self.height, &self.rgb)
    }

    /// Writes the image out, for looking at when a harness run disagrees with
    /// what it was expected to see.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        image::RgbImage::from_raw(self.width, self.height, self.rgb.clone())
            .context("the image buffer is the wrong size")?
            .save(path)
            .with_context(|| format!("failed to write {}", path.display()))
    }
}

/// How the scene is lit and sampled.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderOptions {
    /// Direction *towards* the light, in world space.
    pub light: Vector3<f64>,
    /// How much of the surface colour survives with no light on it at all.
    pub ambient: f32,
    /// What is drawn where no triangle covers a pixel.
    pub background: [f32; 3],
    /// Render at this multiple of the camera's resolution and average down.
    /// A hard pixel edge is not something a camera produces, and not something
    /// the models were trained on.
    pub supersample: u32,
    /// Standard deviation of the sensor noise added afterwards, as a fraction
    /// of full scale. A noiseless image is not a realistic one.
    pub noise: f32,
    pub seed: u64,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            // Down and to one side, as a ceiling light off-centre from the
            // camera, so the body is shaded rather than flat.
            light: Vector3::new(0.35, 0.88, 0.32),
            ambient: 0.50,
            background: [0.06, 0.07, 0.09],
            supersample: 2,
            noise: 0.006,
            seed: 0x5EED_0F7A,
        }
    }
}

/// One vertex as the rasteriser carries it: camera-space position, so that
/// near-plane clipping is a comparison, plus what is needed to shade it.
#[derive(Clone, Copy)]
struct Fragment {
    camera: Point3<f64>,
    normal: Vector3<f64>,
    color: [f32; 3],
}

impl Fragment {
    fn lerp(&self, other: &Fragment, t: f64) -> Fragment {
        Fragment {
            camera: self.camera + (other.camera - self.camera) * t,
            normal: self.normal + (other.normal - self.normal) * t,
            color: std::array::from_fn(|i| {
                self.color[i] + (other.color[i] - self.color[i]) * t as f32
            }),
        }
    }
}

/// Draws `mesh` as seen by `camera`.
///
/// The image is the camera's own resolution; supersampling happens inside and
/// is averaged away before it returns.
pub fn render(mesh: &Mesh, camera: &Camera, options: &RenderOptions) -> Image {
    let scale = options.supersample.max(1);
    let width = camera.intrinsics.width.max(1);
    let height = camera.intrinsics.height.max(1);

    let large = Camera::new(
        camera
            .intrinsics
            .scaled_to(width * scale, height * scale)
            .expect("scaling both axes by the same integer preserves the aspect ratio"),
        camera.lens,
        camera.pose,
    );

    let (w, h) = (
        large.intrinsics.width as usize,
        large.intrinsics.height as usize,
    );
    let mut color = vec![options.background; w * h];
    // Inverse depth, so that "nothing here yet" is zero and nearer is larger.
    // In `f32` because the buffer is the largest thing here and most of the
    // time goes into touching it; a room is a few metres across and depth
    // ordering does not need seven more digits than that.
    let mut depth = vec![0.0f32; w * h];

    let light = options.light.normalize();
    let frustum = frustum(&large);
    let mut polygon = Vec::with_capacity(8);
    let mut scratch = Vec::with_capacity(8);

    for triangle in &mesh.triangles {
        polygon.clear();
        polygon.extend(triangle.iter().map(|index| {
            let vertex = &mesh.vertices[*index as usize];
            Fragment {
                camera: large.pose.inverse_transform_point(&vertex.position),
                normal: vertex.normal,
                color: vertex.color,
            }
        }));

        for plane in &frustum {
            clip(&mut polygon, &mut scratch, *plane);
            if polygon.len() < 3 {
                break;
            }
        }

        for corner in 1..polygon.len().saturating_sub(1) {
            draw(
                &mut color,
                &mut depth,
                w,
                h,
                &large,
                &[polygon[0], polygon[corner], polygon[corner + 1]],
                light,
                options,
            );
        }
    }

    let mut rgb = downsample(
        &color,
        w,
        h,
        width as usize,
        height as usize,
        scale as usize,
    );
    if options.noise > 0.0 {
        add_noise(&mut rgb, options.noise, options.seed);
    }

    Image { width, height, rgb }
}

/// A half-space of camera space, kept where `a x + b y + c z + d >= 0`.
type Plane = [f64; 4];

/// How far outside the image the sides of the clip volume sit, as a multiple of
/// the image's own half-width. Anything past this projects well outside the
/// frame under any lens the calibration would fit.
const MARGIN: f64 = 2.0;

/// The volume a triangle has to be cut down to before it is projected.
///
/// The near plane is the obvious one: dropping any triangle that crosses it
/// would tear a hole in the floor wherever it passes under the camera, which on
/// a ceiling camera is most of the frame edge.
///
/// The four sides are less obvious and matter more. Cutting at the near plane
/// leaves vertices right against the camera and arbitrarily far off-axis, and a
/// distortion polynomial evaluated at a normalised radius of twenty is not
/// wrong by a little — `1 + k1 r^2` goes negative and folds the point back
/// through the centre of the image, where it lands in front of the scene at a
/// depth taken from a surface behind the camera. The symptom is a wall painted
/// over the subject, on the cameras that have any distortion at all and only
/// on those.
fn frustum(camera: &Camera) -> [Plane; 5] {
    let x = MARGIN * 0.5 * camera.intrinsics.width as f64 / camera.intrinsics.fx;
    let y = MARGIN * 0.5 * camera.intrinsics.height as f64 / camera.intrinsics.fy;

    [
        [0.0, 0.0, 1.0, -NEAR],
        [1.0, 0.0, x, 0.0],
        [-1.0, 0.0, x, 0.0],
        [0.0, 1.0, y, 0.0],
        [0.0, -1.0, y, 0.0],
    ]
}

/// Cuts `polygon` down to the half-space `plane` keeps, in place.
///
/// `scratch` is passed in only so that clipping a whole mesh does not allocate
/// once per triangle per plane.
fn clip(polygon: &mut Vec<Fragment>, scratch: &mut Vec<Fragment>, plane: Plane) {
    let distance = |fragment: &Fragment| {
        plane[0] * fragment.camera.x
            + plane[1] * fragment.camera.y
            + plane[2] * fragment.camera.z
            + plane[3]
    };

    if polygon.iter().all(|fragment| distance(fragment) >= 0.0) {
        return;
    }

    scratch.clear();
    for index in 0..polygon.len() {
        let current = polygon[index];
        let next = polygon[(index + 1) % polygon.len()];
        let (here, there) = (distance(&current), distance(&next));

        if here >= 0.0 {
            scratch.push(current);
        }
        if (here >= 0.0) != (there >= 0.0) {
            scratch.push(current.lerp(&next, here / (here - there)));
        }
    }

    std::mem::swap(polygon, scratch);
}

#[allow(clippy::too_many_arguments)]
fn draw(
    color: &mut [[f32; 3]],
    depth: &mut [f32],
    w: usize,
    h: usize,
    camera: &Camera,
    corners: &[Fragment; 3],
    light: Vector3<f64>,
    options: &RenderOptions,
) {
    // Nothing is back-face culled. Culling would need every builder here to
    // promise a winding, and the z-buffer settles the same question without
    // one; measured, it was worth about a tenth of the frame time against the
    // cost of a whole invariant to keep. `shade` turns the normal to face the
    // camera for the same reason.
    let mut screen = [[0.0f64; 2]; 3];
    let mut inverse_z = [0.0f64; 3];
    let mut shaded = [[0.0f32; 3]; 3];

    let eye = camera.position();
    for (index, corner) in corners.iter().enumerate() {
        let world = camera.pose * corner.camera;
        let Some(pixel) = camera.project(world) else {
            return;
        };
        screen[index] = [pixel.x, pixel.y];
        inverse_z[index] = 1.0 / corner.camera.z;
        shaded[index] = shade(corner, world, eye, light, options);
    }

    let area = edge(screen[0], screen[1], screen[2]);
    if area.abs() < 1e-12 {
        return;
    }

    // Clamped into the buffer as a signed range first: a triangle entirely off
    // to the left has a negative maximum, and a cast straight to `usize`
    // saturates it to zero, which would draw a column of it down the edge.
    let Some((min_x, max_x)) = span(screen.map(|p| p[0]), w) else {
        return;
    };
    let Some((min_y, max_y)) = span(screen.map(|p| p[1]), h) else {
        return;
    };

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let point = [x as f64 + 0.5, y as f64 + 0.5];
            let w0 = edge(screen[1], screen[2], point) / area;
            let w1 = edge(screen[2], screen[0], point) / area;
            let w2 = 1.0 - w0 - w1;
            if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                continue;
            }

            // Inverse depth interpolates linearly in screen space, which is
            // what makes it both the right depth test and the right divisor
            // for perspective-correct attributes.
            let iz = w0 * inverse_z[0] + w1 * inverse_z[1] + w2 * inverse_z[2];
            if iz <= 0.0 {
                continue;
            }

            let index = y * w + x;
            if iz as f32 <= depth[index] {
                continue;
            }
            depth[index] = iz as f32;

            let weights = [
                w0 * inverse_z[0] / iz,
                w1 * inverse_z[1] / iz,
                w2 * inverse_z[2] / iz,
            ];
            color[index] = std::array::from_fn(|channel| {
                (weights[0] as f32 * shaded[0][channel]
                    + weights[1] as f32 * shaded[1][channel]
                    + weights[2] as f32 * shaded[2][channel])
                    .clamp(0.0, 1.0)
            });
        }
    }
}

/// Lambert with a hemisphere ambient term.
///
/// The normal is turned to face the camera first. Winding is not something the
/// shape builders promise, and a mesh with one ring wound the other way would
/// otherwise show up as a black band around a limb.
///
/// Both the normal and the view direction are world-space here. Turning the
/// normal to face a *camera-space* view direction compares two different
/// frames, and the answer is worst for a surface the camera looks straight
/// down at: the floor of a room seen from its ceiling flips to face away from
/// the light, and the whole room renders in ambient alone.
fn shade(
    fragment: &Fragment,
    world: Point3<f64>,
    eye: Point3<f64>,
    light: Vector3<f64>,
    options: &RenderOptions,
) -> [f32; 3] {
    let normal = fragment.normal.normalize();
    let towards_camera = eye - world;
    let normal = if normal.dot(&towards_camera) < 0.0 {
        -normal
    } else {
        normal
    };

    let direct = normal.dot(&light).max(0.0) as f32;
    // Sky above, floor below: a surface facing up picks up more of the room
    // than one facing down, which is most of what stops a render looking flat.
    // Only slightly, though — two of a room's four walls face away from any one
    // light, and a strongly directional ambient leaves those two black, which
    // is not what a room with paint on it looks like.
    let ambient = options.ambient * (0.80 + 0.20 * normal.y as f32);
    let lit = (ambient + (1.0 - options.ambient) * direct).clamp(0.0, 1.2);

    std::array::from_fn(|channel| (fragment.color[channel] * lit).clamp(0.0, 1.0))
}

/// Twice the signed area of the triangle `a b c`.
fn edge(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// The pixels of `0..extent` a triangle spanning `values` touches, or `None`
/// when it falls entirely outside them.
fn span(values: [f64; 3], extent: usize) -> Option<(usize, usize)> {
    let low = values.iter().copied().fold(f64::MAX, f64::min).floor();
    let high = values.iter().copied().fold(f64::MIN, f64::max).ceil();
    if high < 0.0 || low > extent as f64 - 1.0 {
        return None;
    }
    Some((
        low.max(0.0) as usize,
        (high.min(extent as f64 - 1.0)).max(0.0) as usize,
    ))
}

fn downsample(
    color: &[[f32; 3]],
    w: usize,
    h: usize,
    width: usize,
    height: usize,
    scale: usize,
) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    let samples = (scale * scale) as f32;

    for y in 0..height {
        for x in 0..width {
            let mut sum = [0.0f32; 3];
            for dy in 0..scale {
                for dx in 0..scale {
                    let sx = (x * scale + dx).min(w - 1);
                    let sy = (y * scale + dy).min(h - 1);
                    let pixel = color[sy * w + sx];
                    for channel in 0..3 {
                        sum[channel] += pixel[channel];
                    }
                }
            }
            let out = (y * width + x) * 3;
            for channel in 0..3 {
                rgb[out + channel] =
                    ((sum[channel] / samples) * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    rgb
}

/// Adds sensor noise.
///
/// Not [`Rng::normal`], which is a logarithm, a square root and a sine per pair
/// of draws. There are two and a half million channels in a 720p frame and the
/// harness renders five hundred of them, and a Box-Muller transform per channel
/// cost more than everything else in the renderer put together — an empty room
/// took thirty-seven milliseconds a frame with not one triangle in it.
///
/// Four bytes of one draw, summed, is Irwin-Hall with n = 4: close enough to
/// Gaussian that eight-bit noise cannot tell the difference, and one call to
/// the generator serves two channels.
fn add_noise(rgb: &mut [u8], sigma: f32, seed: u64) {
    /// Standard deviation of the sum of four uniform bytes.
    const SPREAD: f32 = 147.8;
    /// Where that sum is centred.
    const MIDDLE: i32 = 510;

    let mut rng = Rng::new(seed);
    for pair in rgb.chunks_mut(2) {
        let draw = rng.next_u64().to_le_bytes();
        for (channel, half) in pair.iter_mut().zip(draw.as_chunks::<4>().0) {
            let sum: i32 = half.iter().map(|byte| *byte as i32).sum();
            let noise = (sum - MIDDLE) as f32 / SPREAD;
            *channel = (*channel as f32 + noise * sigma * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::camera::Intrinsics;
    use crate::geometry::lens::Lens;
    use crate::sim::mesh::Mesh;

    fn camera() -> Camera {
        Camera::look_at(
            Intrinsics::from_fov(160, 120, 70f64.to_radians()),
            Lens::default(),
            Point3::new(0.0, 0.0, -3.0),
            Point3::origin(),
            Vector3::y(),
        )
    }

    fn ball() -> Mesh {
        let mut mesh = Mesh::default();
        mesh.add_sphere(Point3::origin(), 0.5, [0.9, 0.2, 0.2], 24, 16);
        mesh
    }

    fn quiet() -> RenderOptions {
        RenderOptions {
            noise: 0.0,
            ..RenderOptions::default()
        }
    }

    fn covered(image: &Image, background: [f32; 3]) -> usize {
        let background: [u8; 3] =
            std::array::from_fn(|i| (background[i] * 255.0).round().clamp(0.0, 255.0) as u8);
        image
            .rgb
            .as_chunks::<3>()
            .0
            .iter()
            .filter(|pixel| {
                pixel
                    .iter()
                    .zip(&background)
                    .any(|(a, b)| a.abs_diff(*b) > 6)
            })
            .count()
    }

    #[test]
    fn an_empty_scene_is_the_background_and_nothing_else() {
        let options = quiet();
        let image = render(&Mesh::default(), &camera(), &options);
        assert_eq!(image.width, 160);
        assert_eq!(image.height, 120);
        assert_eq!(covered(&image, options.background), 0);
    }

    #[test]
    fn a_ball_in_front_of_the_camera_covers_the_middle_of_the_frame() {
        let options = quiet();
        let image = render(&ball(), &camera(), &options);

        let centre = {
            let index = (60 * 160 + 80) * 3;
            [image.rgb[index], image.rgb[index + 1], image.rgb[index + 2]]
        };
        assert!(centre[0] > centre[1] + 40, "the red ball should be red");
        let painted = covered(&image, options.background);
        assert!(
            (0.03..0.5).contains(&(painted as f32 / (160.0 * 120.0))),
            "the ball covered {painted} pixels"
        );
    }

    /// The claim the whole harness rests on. Two runs on one machine is the
    /// cheap half of it; the expensive half is that nothing here consults a
    /// clock, a thread or a GPU.
    #[test]
    fn the_same_scene_renders_to_the_same_pixels_twice() {
        let camera = camera();
        let first = render(&ball(), &camera, &RenderOptions::default());
        let second = render(&ball(), &camera, &RenderOptions::default());
        assert_eq!(first.rgb, second.rgb);
    }

    #[test]
    fn the_noise_is_visible_but_does_not_swamp_the_picture() {
        let camera = camera();
        let quiet = render(&ball(), &camera, &quiet());
        let noisy = render(
            &ball(),
            &camera,
            &RenderOptions {
                noise: 0.02,
                ..RenderOptions::default()
            },
        );

        let changed = quiet
            .rgb
            .iter()
            .zip(&noisy.rgb)
            .filter(|(a, b)| a != b)
            .count();
        assert!(changed > quiet.rgb.len() / 4, "the noise did nothing");

        let worst = quiet
            .rgb
            .iter()
            .zip(&noisy.rgb)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        assert!(worst < 80, "the noise swamped the picture: {worst}");
    }

    /// A ball nearer the camera has to hide the one behind it, whichever order
    /// the triangles happen to arrive in.
    #[test]
    fn the_nearer_surface_wins() {
        let mut mesh = Mesh::default();
        mesh.add_sphere(Point3::new(0.0, 0.0, 0.6), 0.5, [0.1, 0.9, 0.1], 24, 16);
        mesh.add_sphere(Point3::new(0.0, 0.0, -0.6), 0.6, [0.9, 0.1, 0.1], 24, 16);

        let image = render(&mesh, &camera(), &quiet());
        let index = (60 * 160 + 80) * 3;
        let centre = [image.rgb[index], image.rgb[index + 1], image.rgb[index + 2]];
        assert!(
            centre[0] > centre[1] + 40,
            "the near red ball should be in front, got {centre:?}"
        );
    }

    /// A floor running under the camera crosses the near plane. Dropping those
    /// triangles instead of cutting them leaves a wedge of background across
    /// the bottom of every ceiling camera's view.
    #[test]
    fn a_surface_running_under_the_camera_is_cut_rather_than_dropped() {
        let mut mesh = Mesh::default();
        for tile in -8..8 {
            let z = tile as f64 * 1.0;
            mesh.add_quad(
                [
                    Point3::new(-8.0, -1.0, z),
                    Point3::new(8.0, -1.0, z),
                    Point3::new(8.0, -1.0, z + 1.0),
                    Point3::new(-8.0, -1.0, z + 1.0),
                ],
                [0.7, 0.7, 0.7],
            );
        }

        let looking_down = Camera::look_at(
            Intrinsics::from_fov(160, 120, 100f64.to_radians()),
            Lens::default(),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, -1.0, 2.0),
            Vector3::y(),
        );

        let options = quiet();
        let image = render(&mesh, &looking_down, &options);
        let painted = covered(&image, options.background);
        assert!(
            painted as f32 / (160.0 * 120.0) > 0.55,
            "only {painted} pixels of floor survived the near plane"
        );
    }

    /// Supersampling is the only anti-aliasing there is here, so it has to
    /// actually produce intermediate values along an edge.
    #[test]
    fn supersampling_softens_the_edges() {
        let camera = camera();
        let hard = render(
            &ball(),
            &camera,
            &RenderOptions {
                supersample: 1,
                ..quiet()
            },
        );
        let soft = render(
            &ball(),
            &camera,
            &RenderOptions {
                supersample: 3,
                ..quiet()
            },
        );

        let distinct = |image: &Image| {
            let mut seen = std::collections::BTreeSet::new();
            for pixel in image.rgb.as_chunks::<3>().0 {
                seen.insert(*pixel);
            }
            seen.len()
        };
        assert!(
            distinct(&soft) > distinct(&hard),
            "supersampling should add intermediate colours"
        );
    }

    /// A ceiling camera spends most of its frame looking at the floor, and the
    /// floor is the surface most nearly square-on to it. Shading it against a
    /// view direction taken from the wrong frame left the whole room in ambient
    /// alone — dark enough that a detector has nothing to separate a person
    /// from, and a fault that only shows up on exactly the geometry this
    /// project is about.
    #[test]
    fn a_floor_lit_from_above_is_lit_and_not_merely_ambient() {
        let mut mesh = Mesh::default();
        mesh.add_quad(
            [
                Point3::new(-4.0, 0.0, -4.0),
                Point3::new(4.0, 0.0, -4.0),
                Point3::new(4.0, 0.0, 4.0),
                Point3::new(-4.0, 0.0, 4.0),
            ],
            [0.8, 0.8, 0.8],
        );

        let overhead = Camera::look_at(
            Intrinsics::from_fov(64, 48, 70f64.to_radians()),
            Lens::default(),
            Point3::new(0.0, 2.4, 0.0),
            Point3::new(0.0, 0.0, 0.1),
            Vector3::y(),
        );

        let options = RenderOptions {
            light: Vector3::y(),
            ambient: 0.4,
            ..quiet()
        };
        let image = render(&mesh, &overhead, &options);
        let index = (24 * 64 + 32) * 3;

        // Lit: 0.8 * (0.4 * (0.80 + 0.20) + 0.6 * 1.0) = 0.8. Ambient alone
        // with the normal flipped away would be 0.8 * 0.4 * 0.60 = 0.19.
        assert!(
            image.rgb[index] > 180,
            "the floor came out at {}, which is ambient and no light",
            image.rgb[index]
        );
    }

    #[test]
    fn a_scene_entirely_behind_the_camera_draws_nothing() {
        let mut mesh = Mesh::default();
        mesh.add_sphere(Point3::new(0.0, 0.0, -6.0), 0.5, [0.9, 0.2, 0.2], 16, 10);

        let options = quiet();
        let image = render(&mesh, &camera(), &options);
        assert_eq!(covered(&image, options.background), 0);
    }
}
