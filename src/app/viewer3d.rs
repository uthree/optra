//! A small 3D view, drawn with egui's painter.
//!
//! The scene is a floor grid, a few camera frusta and a walk path — a few
//! hundred line segments. A wgpu render pass would be more machinery than that
//! is worth, and the projection needed is one the application already has: the
//! viewer builds a [`Camera`] looking at the scene and reuses the same
//! `project` the calibration solves against.
//!
//! What it is for is judging a calibration at a glance. Residuals in degrees
//! say whether the cameras agree with each other; a picture of where they came
//! out, with the recorded walk drawn through them, says whether they agree with
//! the room. Those are different questions, and the second one is the one a
//! user can answer.

use egui::{Align2, Color32, FontId, Pos2, Rect, Response, Sense, Stroke, Ui, Vec2};
use nalgebra::{Point3, Vector3};

use crate::geometry::camera::{Camera, Intrinsics};
use crate::geometry::lens::Lens;

/// Nothing closer than this to the eye is drawn; segments crossing it are cut.
const NEAR: f64 = 0.05;

/// Field of view of the virtual camera, in degrees. Narrow enough that a room
/// does not look like a fisheye photograph of itself.
const FOV: f64 = 50.0;

/// A line to draw, in world coordinates.
pub struct Segment {
    pub from: Point3<f64>,
    pub to: Point3<f64>,
    pub colour: Color32,
    pub width: f32,
}

/// A world-space label.
pub struct Label {
    pub at: Point3<f64>,
    pub text: String,
    pub colour: Color32,
}

#[derive(Default)]
pub struct Scene {
    pub segments: Vec<Segment>,
    pub labels: Vec<Label>,
}

impl Scene {
    pub fn line(&mut self, from: Point3<f64>, to: Point3<f64>, colour: Color32, width: f32) {
        self.segments.push(Segment {
            from,
            to,
            colour,
            width,
        });
    }

    pub fn label(&mut self, at: Point3<f64>, text: impl Into<String>, colour: Color32) {
        self.labels.push(Label {
            at,
            text: text.into(),
            colour,
        });
    }

    /// A grid on the floor, which is what gives everything else a sense of
    /// scale. Without it a camera three metres up and a camera thirty
    /// centimetres up look identical.
    pub fn floor(&mut self, extent: f64, step: f64) {
        let faint = Color32::from_rgb(58, 62, 70);
        let axis = Color32::from_rgb(86, 92, 104);

        let mut at = -extent;
        while at <= extent + 1e-9 {
            let colour = if at.abs() < 1e-9 { axis } else { faint };
            self.line(
                Point3::new(at, 0.0, -extent),
                Point3::new(at, 0.0, extent),
                colour,
                1.0,
            );
            self.line(
                Point3::new(-extent, 0.0, at),
                Point3::new(extent, 0.0, at),
                colour,
                1.0,
            );
            at += step;
        }

        // Up, so the floor is unmistakably the floor.
        self.line(
            Point3::origin(),
            Point3::new(0.0, 1.0, 0.0),
            Color32::from_rgb(110, 130, 160),
            1.0,
        );
    }

    /// A camera as its position and the pyramid it can see.
    ///
    /// The pyramid is built from the camera's own intrinsics, so a wide camera
    /// draws wide. That is the point: two cameras pointing at the same spot
    /// from the same place still cover very different amounts of the room, and
    /// the picture should show it.
    pub fn camera(&mut self, camera: &Camera, label: &str, colour: Color32, depth: f64) {
        let eye = camera.position();
        let intrinsics = &camera.intrinsics;

        let corner = |x: f64, y: f64| {
            let direction = camera.pose.rotation
                * Vector3::new(
                    (x - intrinsics.cx) / intrinsics.fx,
                    (y - intrinsics.cy) / intrinsics.fy,
                    1.0,
                );
            eye + direction * depth
        };

        let (w, h) = (intrinsics.width as f64, intrinsics.height as f64);
        let corners = [
            corner(0.0, 0.0),
            corner(w, 0.0),
            corner(w, h),
            corner(0.0, h),
        ];

        for corner in &corners {
            self.line(eye, *corner, colour, 1.0);
        }
        for index in 0..4 {
            self.line(corners[index], corners[(index + 1) % 4], colour, 1.5);
        }

        // A stub marking which way is up in the image, so a camera mounted
        // upside down is visible as such rather than as a puzzle.
        let top = corners[0] + (corners[1] - corners[0]) * 0.5;
        self.line(top, top + (top - eye) * 0.25, colour, 1.5);

        self.line(
            eye,
            Point3::new(eye.x, 0.0, eye.z),
            Color32::from_rgb(70, 74, 84),
            1.0,
        );
        self.label(eye, label, colour);
    }

    /// A path through the room, drawn as a polyline.
    pub fn path(&mut self, points: &[Point3<f64>], colour: Color32) {
        for pair in points.windows(2) {
            self.line(pair[0], pair[1], colour, 1.0);
        }
    }
}

/// An orbiting view of a scene.
pub struct Viewer3d {
    yaw: f32,
    pitch: f32,
    distance: f32,
    target: Point3<f64>,
}

impl Default for Viewer3d {
    fn default() -> Self {
        Self {
            yaw: 0.7,
            pitch: 0.55,
            distance: 7.0,
            // Chest height at the middle of the room: what the user cares
            // about is where they stand, not where the floor is.
            target: Point3::new(0.0, 1.0, 0.0),
        }
    }
}

impl Viewer3d {
    /// Points the view at a set of world positions, framing all of them.
    pub fn frame(&mut self, points: &[Point3<f64>]) {
        if points.is_empty() {
            return;
        }

        let centre = points
            .iter()
            .fold(Vector3::zeros(), |sum, p| sum + p.coords)
            / points.len() as f64;
        let radius = points
            .iter()
            .map(|p| (p.coords - centre).norm())
            .fold(0.0, f64::max);

        self.target = Point3::from(centre);
        self.distance = (radius * 2.6).clamp(2.0, 30.0) as f32;
    }

    pub fn show(&mut self, ui: &mut Ui, scene: &Scene, height: f32) -> Response {
        let width = ui.available_width().max(120.0);
        let (rect, response) =
            ui.allocate_exact_size(Vec2::new(width, height), Sense::click_and_drag());

        if response.dragged() {
            let drag = response.drag_delta();
            self.yaw -= drag.x * 0.008;
            // Stopped short of straight down: at the pole the up vector the
            // camera basis is built from becomes parallel to the view and the
            // orientation is undefined.
            self.pitch = (self.pitch + drag.y * 0.008).clamp(-1.45, 1.45);
        }
        if response.hovered() {
            let scroll = ui.input(|input| input.smooth_scroll_delta.y);
            if scroll != 0.0 {
                self.distance = (self.distance * (1.0 - scroll * 0.0015)).clamp(0.5, 40.0);
            }
        }

        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 2.0, Color32::from_rgb(24, 26, 30));

        let eye = self.eye();
        let camera = Camera::look_at(
            Intrinsics::from_fov(rect.width() as u32, rect.height() as u32, FOV.to_radians()),
            Lens::default(),
            eye,
            self.target,
            Vector3::y(),
        );

        for segment in &scene.segments {
            if let Some((from, to)) = project_segment(&camera, rect, segment.from, segment.to) {
                painter.line_segment([from, to], Stroke::new(segment.width, segment.colour));
            }
        }

        for label in &scene.labels {
            if let Some(at) = project_point(&camera, rect, label.at) {
                painter.text(
                    at + Vec2::new(0.0, -6.0),
                    Align2::CENTER_BOTTOM,
                    &label.text,
                    FontId::proportional(11.0),
                    label.colour,
                );
            }
        }

        response
    }

    fn eye(&self) -> Point3<f64> {
        let (yaw, pitch) = (self.yaw as f64, self.pitch as f64);
        let direction = Vector3::new(
            pitch.cos() * yaw.sin(),
            pitch.sin(),
            pitch.cos() * yaw.cos(),
        );
        self.target + direction * self.distance as f64
    }
}

fn project_point(camera: &Camera, rect: Rect, world: Point3<f64>) -> Option<Pos2> {
    let pixel = camera.project(world)?;
    Some(Pos2::new(
        rect.min.x + pixel.x as f32,
        rect.min.y + pixel.y as f32,
    ))
}

/// Projects a segment, cutting it where it crosses behind the eye.
///
/// Skipping such segments entirely would make the floor grid vanish in pieces
/// as the view turns, which reads as a rendering fault rather than as the
/// horizon.
fn project_segment(
    camera: &Camera,
    rect: Rect,
    from: Point3<f64>,
    to: Point3<f64>,
) -> Option<(Pos2, Pos2)> {
    let a = camera.pose.inverse_transform_point(&from);
    let b = camera.pose.inverse_transform_point(&to);

    let (a, b) = match (a.z >= NEAR, b.z >= NEAR) {
        (true, true) => (a, b),
        (false, false) => return None,
        (true, false) => (a, clip(a, b)),
        (false, true) => (clip(b, a), b),
    };

    let to_screen = |point: Point3<f64>| {
        Pos2::new(
            rect.min.x + (camera.intrinsics.fx * point.x / point.z + camera.intrinsics.cx) as f32,
            rect.min.y + (camera.intrinsics.fy * point.y / point.z + camera.intrinsics.cy) as f32,
        )
    };

    Some((to_screen(a), to_screen(b)))
}

/// Where the segment from `inside` to `outside` crosses the near plane.
fn clip(inside: Point3<f64>, outside: Point3<f64>) -> Point3<f64> {
    let span = inside.z - outside.z;
    let t = if span.abs() < 1e-12 {
        0.0
    } else {
        (inside.z - NEAR) / span
    };
    inside + (outside - inside) * t.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn viewer() -> Viewer3d {
        Viewer3d::default()
    }

    fn camera_at(eye: Point3<f64>, target: Point3<f64>) -> Camera {
        Camera::look_at(
            Intrinsics::from_fov(800, 600, FOV.to_radians()),
            Lens::default(),
            eye,
            target,
            Vector3::y(),
        )
    }

    #[test]
    fn framing_a_room_puts_the_view_around_it() {
        let mut viewer = viewer();
        viewer.frame(&[
            Point3::new(-2.0, 2.4, -2.0),
            Point3::new(2.0, 2.4, -2.0),
            Point3::new(2.0, 2.4, 2.0),
            Point3::new(-2.0, 2.4, 2.0),
        ]);

        assert!((viewer.target.coords - Vector3::new(0.0, 2.4, 0.0)).norm() < 1e-9);
        assert!(
            viewer.distance > 5.0 && viewer.distance < 12.0,
            "a four metre room framed at {} m",
            viewer.distance
        );
    }

    #[test]
    fn framing_nothing_leaves_the_view_alone() {
        let mut viewer = viewer();
        let before = (viewer.target, viewer.distance);
        viewer.frame(&[]);
        assert_eq!((viewer.target, viewer.distance), before);
    }

    /// A segment with both ends behind the eye is not drawn at all.
    #[test]
    fn a_segment_behind_the_eye_is_dropped() {
        let camera = camera_at(Point3::origin(), Point3::new(0.0, 0.0, 1.0));
        let rect = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));

        assert!(
            project_segment(
                &camera,
                rect,
                Point3::new(-1.0, 0.0, -2.0),
                Point3::new(1.0, 0.0, -3.0)
            )
            .is_none()
        );
    }

    /// One that crosses the eye plane is cut rather than dropped, so the floor
    /// grid recedes to a horizon instead of disappearing in pieces.
    #[test]
    fn a_segment_crossing_the_eye_plane_is_cut() {
        let camera = camera_at(Point3::origin(), Point3::new(0.0, 0.0, 1.0));
        let rect = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));

        let (near, far) = project_segment(
            &camera,
            rect,
            Point3::new(0.0, 0.0, -3.0),
            Point3::new(0.0, 0.0, 5.0),
        )
        .expect("part of it is in front of the eye");

        assert!(near.x.is_finite() && far.x.is_finite());
        assert!(
            (near.x - 400.0).abs() < 1.0 && (far.x - 400.0).abs() < 1.0,
            "a segment straight ahead should stay on the centre line"
        );
    }

    /// The frustum is built from the camera's own intrinsics, so a wide camera
    /// has to come out wider than a narrow one from the same place.
    #[test]
    fn a_wide_camera_draws_a_wider_frustum() {
        let footprint = |fov: f64| {
            let camera = Camera::look_at(
                Intrinsics::from_fov(1280, 720, fov.to_radians()),
                Lens::default(),
                Point3::new(0.0, 2.4, 0.0),
                Point3::new(0.0, 0.0, 0.0),
                Vector3::z(),
            );

            let mut scene = Scene::default();
            scene.camera(&camera, "cam", Color32::WHITE, 1.0);

            // The four rays from the eye come first; the widest of them says
            // how much of the room the camera covers.
            scene.segments[..4]
                .iter()
                .map(|segment| (segment.to - segment.from).norm())
                .fold(0.0, f64::max)
        };

        assert!(
            footprint(100.0) > footprint(60.0) * 1.3,
            "a hundred degree camera should reach much wider than a sixty degree one"
        );
    }

    #[test]
    fn a_floor_grid_is_made_of_lines_on_the_floor() {
        let mut scene = Scene::default();
        scene.floor(2.0, 0.5);

        assert!(scene.segments.len() > 8);
        let on_the_floor = scene
            .segments
            .iter()
            .filter(|segment| segment.from.y == 0.0 && segment.to.y == 0.0)
            .count();
        assert_eq!(
            on_the_floor,
            scene.segments.len() - 1,
            "everything but the up marker should lie flat"
        );
    }
}
