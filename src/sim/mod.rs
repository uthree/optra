//! A simulated room, rendered, with the answer known.
//!
//! This exists so that the path from pixels to a 3D skeleton can be measured
//! without a room, without cameras and without a headset. A scene puts a
//! walking figure in a tiled room, hangs unlike cameras in the ceiling corners,
//! and renders what each of them sees — and because the figure was built by
//! forward kinematics from stated bone lengths, it can also say exactly where
//! every joint was while the shutter was open.
//!
//! That is the difference between this and the synthetic keypoints the fusion
//! tests use. Those start from perfectly projected joints and measure what
//! happens after; this starts from an image and measures the model too. The
//! stage between them — a detector deciding where a person is, and a pose model
//! deciding where their knee is inside that box — is the one stage of the
//! pipeline nothing else in the project can put a number on.
//!
//! Two rules hold everything together:
//!
//! - The renderer projects through [`geometry::camera::Camera`], the same code
//!   the solver and the triangulation use, lens distortion included. A bug in
//!   the projection cannot hide by being present at both ends, because there is
//!   only one end.
//! - Nothing here consults a clock, a thread or a GPU. The same scene renders
//!   to the same bytes on every machine, which is what makes it reasonable for
//!   a test to assert a figure in millimetres.
//!
//! [`geometry::camera::Camera`]: crate::geometry::camera::Camera

pub mod body;
pub mod figure;
pub mod mesh;
pub mod render;
pub mod room;

use crate::geometry::camera::Camera;
use crate::sim::body::{Anatomy, Posture, Walk};
use crate::sim::figure::Shape;
use crate::sim::mesh::Mesh;
use crate::sim::render::{Image, RenderOptions};
use crate::sim::room::Room;

/// Everything needed to render a moment of the simulation, and to say what was
/// true at that moment.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Scene {
    pub room: Room,
    pub anatomy: Anatomy,
    pub shape: Shape,
    pub walk: Walk,
    pub render: RenderOptions,
}

impl Scene {
    /// Where every joint is at `t` seconds into the walk. This is the ground
    /// truth the reconstruction is scored against.
    pub fn posture(&self, t: f64) -> Posture {
        self.walk.posture(&self.anatomy, t)
    }

    /// The room and the figure in it at `t`.
    pub fn mesh(&self, t: f64) -> Mesh {
        let mut mesh = self.room.mesh();
        mesh.merge(&figure::build(&self.anatomy, &self.shape, &self.posture(t)));
        mesh
    }

    /// What `camera` sees at `t`.
    pub fn view(&self, camera: &Camera, t: f64) -> Image {
        render::render(&self.mesh(t), camera, &self.render)
    }

    /// The cameras this room ends up with; see [`Room::cameras`].
    pub fn cameras(&self, count: usize) -> Vec<Camera> {
        self.room.cameras(count)
    }
}

/// A small deterministic generator.
///
/// Sensor noise and jitter have to be repeatable or a test that asserts an
/// error figure is asserting against a different sample every run. This is
/// xorshift64\*, which is neither cryptographic nor especially good, and is
/// entirely adequate for shaking a picture.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
    spare: Option<f64>,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self {
            // A zero state is a fixed point of xorshift, so it is never one.
            state: seed | 1,
            spare: None,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        self.state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        // The top 53 bits, which is exactly the mantissa of an f64.
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal, by the Box-Muller transform. The pair it produces is
    /// kept, so two draws cost one transform.
    pub fn normal(&mut self) -> f64 {
        if let Some(spare) = self.spare.take() {
            return spare;
        }

        let radius = (-2.0 * (1.0 - self.unit()).ln()).sqrt();
        let angle = std::f64::consts::TAU * self.unit();
        self.spare = Some(radius * angle.sin());
        radius * angle.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Joint;

    #[test]
    fn a_scene_renders_what_each_camera_can_see() {
        let scene = Scene::default();
        for camera in scene.cameras(4) {
            let image = scene.view(&camera, 1.3);
            assert_eq!(image.width, camera.intrinsics.width);
            assert_eq!(image.height, camera.intrinsics.height);
            assert_eq!(
                image.rgb.len(),
                image.width as usize * image.height as usize * 3
            );
        }
    }

    /// The figure has to be drawn where the truth says it is, or every number
    /// the harness reports is measured against the wrong body. Sampling the
    /// rendered pixel at each projected joint and comparing it against the
    /// empty room is the cheapest possible statement of that.
    #[test]
    fn the_figure_is_drawn_where_the_ground_truth_says_it_is() {
        let scene = Scene {
            render: RenderOptions {
                noise: 0.0,
                ..RenderOptions::default()
            },
            ..Scene::default()
        };
        let empty = Scene {
            walk: Walk {
                // Walked out of the room, so the same frame is the room alone.
                radius: 40.0,
                ..scene.walk
            },
            ..scene.clone()
        };

        for camera in scene.cameras(4) {
            let occupied = scene.view(&camera, 2.1);
            let bare = empty.view(&camera, 2.1);
            let posture = scene.posture(2.1);

            let mut hits = 0;
            let mut looked = 0;
            for (_, point) in posture.iter() {
                let Some(pixel) = camera.project(point) else {
                    continue;
                };
                let (x, y) = (pixel.x as i32, pixel.y as i32);
                if x < 0 || y < 0 || x >= occupied.width as i32 || y >= occupied.height as i32 {
                    continue;
                }
                looked += 1;
                if occupied.view().sample(x, y) != bare.view().sample(x, y) {
                    hits += 1;
                }
            }

            assert!(
                looked >= Joint::ALL.len() - 2,
                "only {looked} joints in view"
            );
            assert!(
                hits * 10 >= looked * 9,
                "only {hits} of {looked} joints landed on the figure"
            );
        }
    }

    #[test]
    fn the_room_alone_is_the_same_picture_every_time() {
        let scene = Scene::default();
        let camera = scene.cameras(1).remove(0);
        assert_eq!(scene.view(&camera, 0.5).rgb, scene.view(&camera, 0.5).rgb);
    }

    #[test]
    fn the_figure_moves_between_frames() {
        let scene = Scene::default();
        let camera = scene.cameras(1).remove(0);
        assert_ne!(scene.view(&camera, 0.0).rgb, scene.view(&camera, 1.7).rgb);
    }

    #[test]
    fn the_generator_repeats_for_a_seed_and_differs_between_seeds() {
        let draw = |seed| {
            let mut rng = Rng::new(seed);
            (0..8).map(|_| rng.unit()).collect::<Vec<_>>()
        };
        assert_eq!(draw(7), draw(7));
        assert_ne!(draw(7), draw(8));
    }

    #[test]
    fn the_generator_stays_inside_the_unit_interval() {
        let mut rng = Rng::new(0);
        for _ in 0..10_000 {
            let value = rng.unit();
            assert!((0.0..1.0).contains(&value));
        }
    }

    #[test]
    fn the_normal_draws_look_normal() {
        let mut rng = Rng::new(1234);
        let samples: Vec<f64> = (0..20_000).map(|_| rng.normal()).collect();

        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;

        assert!(mean.abs() < 0.05, "mean {mean}");
        assert!((variance - 1.0).abs() < 0.05, "variance {variance}");
    }
}
