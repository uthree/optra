//! The room the figure walks in, and where the cameras hang in it.

use nalgebra::{Point3, Vector3};

use crate::geometry::camera::{Camera, Intrinsics};
use crate::geometry::lens::Lens;
use crate::sim::mesh::Mesh;

/// A square room with a tiled floor and four walls.
///
/// The floor is tiled rather than flat because a plain floor gives a detector
/// nothing to separate a person from, and because a room with no texture in it
/// makes every rendered frame look like a diagram instead of a photograph.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Room {
    /// Half the floor's width, in metres.
    pub half_extent: f64,
    /// Ceiling height, in metres. The cameras hang just under it.
    pub ceiling: f64,
    /// Side of one floor tile, in metres.
    pub tile: f64,
    /// The two floor tile colours.
    pub floor: ([f32; 3], [f32; 3]),
    pub wall: [f32; 3],
    pub ceiling_color: [f32; 3],
}

impl Default for Room {
    fn default() -> Self {
        Self {
            half_extent: 2.5,
            ceiling: 2.4,
            tile: 0.5,
            floor: ([0.52, 0.50, 0.47], [0.40, 0.39, 0.37]),
            wall: [0.72, 0.71, 0.68],
            ceiling_color: [0.80, 0.80, 0.78],
        }
    }
}

impl Room {
    /// The floor, the four walls and the ceiling.
    ///
    /// The ceiling is here because the cameras hang just under it and look
    /// across the room, so without one they see over the tops of the far walls
    /// and out into nothing: a large flat region of background in the upper
    /// third of every frame, which is not what any camera in any room produces.
    pub fn mesh(&self) -> Mesh {
        let mut mesh = Mesh::default();
        let extent = self.half_extent;
        let corner = Point3::new(-extent, 0.0, -extent);
        let span = 2.0 * extent;

        tile(
            &mut mesh,
            corner,
            Vector3::x() * span,
            Vector3::z() * span,
            self.tile,
            |column, row| {
                if (column + row) % 2 == 0 {
                    self.floor.0
                } else {
                    self.floor.1
                }
            },
        );

        for (from, along) in [
            (corner, Vector3::x() * span),
            (corner + Vector3::x() * span, Vector3::z() * span),
            (corner + Vector3::new(span, 0.0, span), Vector3::x() * -span),
            (corner + Vector3::z() * span, Vector3::z() * -span),
        ] {
            tile(
                &mut mesh,
                from,
                along,
                Vector3::y() * self.ceiling,
                self.tile,
                |_, _| self.wall,
            );
        }

        tile(
            &mut mesh,
            corner + Vector3::y() * self.ceiling,
            Vector3::x() * span,
            Vector3::z() * span,
            self.tile,
            |_, _| self.ceiling_color,
        );

        mesh
    }

    /// Where camera `seat` hangs, cycling through the four ceiling corners.
    pub fn seat(&self, seat: u32) -> Point3<f64> {
        let inset = self.half_extent - 0.12;
        let corners = [
            Point3::new(-inset, self.ceiling - 0.12, -inset),
            Point3::new(inset, self.ceiling - 0.12, -inset),
            Point3::new(inset, self.ceiling - 0.12, inset),
            Point3::new(-inset, self.ceiling - 0.12, inset),
        ];
        corners[seat as usize % corners.len()]
    }

    /// A camera in one of the ceiling corners, aimed at knee height in the
    /// middle of the room.
    ///
    /// Knee height rather than chest height, which is the aim a first guess
    /// would pick. A camera in a ceiling corner is already looking down, and
    /// aiming it at the middle of a standing body pushes the feet of somebody
    /// standing near it off the bottom of the frame — the feet being the whole
    /// reason this application exists. Everything above the waist is redundant
    /// across four cameras; the feet are not.
    pub fn ceiling_camera(
        &self,
        seat: u32,
        width: u32,
        height: u32,
        horizontal_fov: f64,
        lens: Lens,
    ) -> Camera {
        Camera::look_at(
            Intrinsics::from_fov(width, height, horizontal_fov),
            lens,
            self.seat(seat),
            Point3::new(0.0, 0.6, 0.0),
            Vector3::y(),
        )
    }

    /// The set of cameras a room like this actually ends up with.
    ///
    /// Deliberately unalike — three resolutions, three fields of view, and one
    /// with real distortion on it — because a set of identical cameras hides
    /// every bug that comes from mixing them, and mixing them is the normal
    /// case for a user assembling a rig out of whatever webcams they own.
    pub fn cameras(&self, count: usize) -> Vec<Camera> {
        // The distortion is mild, and deliberately so. `1 + k1 r^2` stops being
        // monotonic at `r = sqrt(-1/(3 k1))`, and past that peak two different
        // directions land on the same pixel and no pixel has a ray at all —
        // which is not a lens, it is a polynomial out of its range. A wide
        // camera has a large normalised radius at the corner of its frame, so
        // the wider the camera the smaller the `k1` it can carry before its own
        // corners fall off the far side of that peak. These stay inside it with
        // room to spare, which is also what a solved room looks like: a lens
        // needing more than this is a fisheye, and there is a separate model
        // for those.
        const FITTED: [(u32, u32, f64, f64); 4] = [
            (1280, 720, 80.0, 0.0),
            (1920, 1080, 78.0, -0.08),
            (640, 480, 96.0, -0.035),
            (1280, 960, 84.0, -0.05),
        ];

        (0..count as u32)
            .map(|seat| {
                let (width, height, fov, k1) = FITTED[seat as usize % FITTED.len()];
                let lens = Lens::RadialTangential {
                    k1,
                    k2: 0.0,
                    // A trace of tangential distortion on one camera, so that
                    // the terms are not silently untested by every camera
                    // having a perfectly centred sensor.
                    p1: if seat == 1 { 0.0004 } else { 0.0 },
                    p2: if seat == 1 { -0.0003 } else { 0.0 },
                };
                self.ceiling_camera(seat, width, height, fov.to_radians(), lens)
            })
            .collect()
    }
}

/// Fills a rectangle with quads no larger than `step` on a side, colouring each
/// by its position in the grid.
///
/// Tiling rather than emitting one large quad is not only about the
/// checkerboard. A triangle spanning a whole wall runs from just in front of
/// the camera to the far side of the room, and a rasteriser interpolating depth
/// across it in screen space is at its least accurate exactly there.
fn tile(
    mesh: &mut Mesh,
    origin: Point3<f64>,
    across: Vector3<f64>,
    up: Vector3<f64>,
    step: f64,
    color: impl Fn(i32, i32) -> [f32; 3],
) {
    let columns = (across.norm() / step).round().max(1.0) as i32;
    let rows = (up.norm() / step).round().max(1.0) as i32;

    for row in 0..rows {
        for column in 0..columns {
            let a = across / columns as f64;
            let b = up / rows as f64;
            let at = origin + a * column as f64 + b * row as f64;
            mesh.add_quad([at, at + a, at + a + b, at + b], color(column, row));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::body::{Anatomy, Walk};

    #[test]
    fn the_floor_is_tiled_in_two_colours() {
        let room = Room::default();
        let mesh = room.mesh();

        let floor: Vec<[f32; 3]> = mesh
            .vertices
            .iter()
            .filter(|vertex| vertex.position.y.abs() < 1e-9)
            .map(|vertex| vertex.color)
            .collect();
        assert!(floor.contains(&room.floor.0));
        assert!(floor.contains(&room.floor.1));
    }

    #[test]
    fn the_walls_reach_the_ceiling_and_the_room_is_closed() {
        let room = Room::default();
        let mesh = room.mesh();

        let highest = mesh
            .vertices
            .iter()
            .map(|vertex| vertex.position.y)
            .fold(f64::MIN, f64::max);
        assert!((highest - room.ceiling).abs() < 1e-9);

        for vertex in &mesh.vertices {
            assert!(vertex.position.coords.xz().amax() <= room.half_extent + 1e-9);
        }
    }

    #[test]
    fn the_cameras_hang_under_the_ceiling_in_different_corners() {
        let room = Room::default();
        let cameras = room.cameras(4);
        assert_eq!(cameras.len(), 4);

        for camera in &cameras {
            let position = camera.position();
            assert!(position.y > room.ceiling - 0.2);
            assert!(position.y < room.ceiling);
            assert!(camera.forward().y < 0.0, "a ceiling camera looks down");
        }

        for (index, camera) in cameras.iter().enumerate() {
            for other in &cameras[index + 1..] {
                assert!((camera.position() - other.position()).norm() > 1.0);
            }
        }
    }

    #[test]
    fn no_two_cameras_are_the_same_kind() {
        let cameras = Room::default().cameras(4);
        for (index, camera) in cameras.iter().enumerate() {
            for other in &cameras[index + 1..] {
                assert!(
                    camera.intrinsics.width != other.intrinsics.width
                        || camera.intrinsics.height != other.intrinsics.height
                        || (camera.intrinsics.horizontal_fov() - other.intrinsics.horizontal_fov())
                            .abs()
                            > 1e-3,
                    "two cameras in the set are identical"
                );
            }
        }
        assert!(
            cameras.iter().any(|camera| !camera.lens.is_identity()),
            "at least one camera should have a lens worth undistorting"
        );
    }

    /// A camera that cannot see the walk is worse than no camera: the harness
    /// would report a fusion failure that is really a room layout mistake.
    #[test]
    fn every_camera_sees_the_whole_walk() {
        let room = Room::default();
        let anatomy = Anatomy::default();
        let walk = Walk::default();

        for (seat, camera) in room.cameras(4).iter().enumerate() {
            for step in 0..120 {
                let posture = walk.posture(&anatomy, step as f64 * 0.1);
                for (joint, point) in posture.iter() {
                    let pixel = camera.project(point).unwrap_or_else(|| {
                        panic!("camera {seat} lost {joint:?} behind itself at step {step}")
                    });
                    assert!(
                        pixel.x >= 0.0
                            && pixel.y >= 0.0
                            && pixel.x < camera.intrinsics.width as f64
                            && pixel.y < camera.intrinsics.height as f64,
                        "camera {seat} lost {joint:?} off the frame at step {step}: {pixel:?}"
                    );
                }
            }
        }
    }
}
