//! The camera model.
//!
//! Everything the calibration solves for, and everything triangulation needs,
//! lives here: where a camera is, where it looks, and how it turns a direction
//! into a pixel.

use nalgebra::{
    Isometry3, Matrix3, Point2, Point3, Rotation3, Translation3, Unit, UnitQuaternion, Vector3,
};
use serde::{Deserialize, Serialize};

use super::lens::Lens;

/// Pinhole intrinsics, in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Intrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    /// Image size the intrinsics were solved at. Keeping it means a preview at
    /// another resolution can be scaled instead of silently mismatching.
    pub width: u32,
    pub height: u32,
}

impl Intrinsics {
    /// A first guess from a field of view, for seeding the calibration.
    pub fn from_fov(width: u32, height: u32, horizontal_fov: f64) -> Self {
        let fx = 0.5 * width as f64 / (horizontal_fov * 0.5).tan();
        Self {
            fx,
            fy: fx,
            cx: width as f64 * 0.5,
            cy: height as f64 * 0.5,
            width,
            height,
        }
    }

    pub fn matrix(&self) -> Matrix3<f64> {
        Matrix3::new(self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0)
    }

    /// Horizontal field of view in radians.
    pub fn horizontal_fov(&self) -> f64 {
        2.0 * (0.5 * self.width as f64 / self.fx).atan()
    }

    /// Angular size of one pixel at the image centre, in radians.
    ///
    /// This is the number that makes cameras comparable: a pixel on a wide
    /// 480p camera covers several times the solid angle of a pixel on a narrow
    /// 1080p one, so any threshold or weight expressed in pixels means
    /// something different on each.
    pub fn radians_per_pixel(&self) -> f64 {
        1.0 / self.fx.max(1e-9)
    }
}

/// A calibrated camera: where it is, and how it sees.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Camera {
    pub intrinsics: Intrinsics,
    pub lens: Lens,
    /// Camera-to-world transform. Camera axes follow the OpenCV convention:
    /// +x right, +y down, +z into the scene.
    pub pose: Isometry3<f64>,
}

impl Camera {
    pub fn new(intrinsics: Intrinsics, lens: Lens, pose: Isometry3<f64>) -> Self {
        Self {
            intrinsics,
            lens,
            pose,
        }
    }

    /// Builds a camera looking from `eye` towards `target`.
    ///
    /// `world_up` only fixes the roll; a camera pointing straight down needs a
    /// different one, which is why it is a parameter rather than a constant.
    pub fn look_at(
        intrinsics: Intrinsics,
        lens: Lens,
        eye: Point3<f64>,
        target: Point3<f64>,
        world_up: Vector3<f64>,
    ) -> Self {
        let forward = (target - eye).normalize();
        let right = forward.cross(&world_up).normalize();
        // Image y grows downward. The order matters: `right.cross(&forward)`
        // points the other way and makes the basis a reflection rather than a
        // rotation, which silently produces a camera that sees nothing.
        let down = forward.cross(&right).normalize();

        let rotation =
            Rotation3::from_matrix_unchecked(Matrix3::from_columns(&[right, down, forward]));
        let pose = Isometry3::from_parts(
            Translation3::from(eye.coords),
            UnitQuaternion::from_rotation_matrix(&rotation),
        );

        Self::new(intrinsics, lens, pose)
    }

    /// Where the camera is, in world space.
    pub fn position(&self) -> Point3<f64> {
        Point3::from(self.pose.translation.vector)
    }

    /// The direction the camera looks, in world space.
    pub fn forward(&self) -> Vector3<f64> {
        self.pose.rotation * Vector3::z()
    }

    /// Projects a world point into pixels, or `None` if it is behind the camera.
    pub fn project(&self, world: Point3<f64>) -> Option<Point2<f64>> {
        let local = self.pose.inverse_transform_point(&world);
        if local.z <= 1e-6 {
            return None;
        }

        let (x, y) = self.lens.distort(local.x / local.z, local.y / local.z);
        Some(Point2::new(
            self.intrinsics.fx * x + self.intrinsics.cx,
            self.intrinsics.fy * y + self.intrinsics.cy,
        ))
    }

    /// The world-space ray a pixel corresponds to, as a unit direction.
    pub fn ray(&self, pixel: Point2<f64>) -> Unit<Vector3<f64>> {
        let x = (pixel.x - self.intrinsics.cx) / self.intrinsics.fx;
        let y = (pixel.y - self.intrinsics.cy) / self.intrinsics.fy;
        let (x, y) = self.lens.undistort(x, y);

        Unit::new_normalize(self.pose.rotation * Vector3::new(x, y, 1.0))
    }

    /// Angle between where a world point actually appears and where it was
    /// observed, in radians.
    ///
    /// Reprojection error is measured as an angle rather than in pixels so that
    /// the same threshold means the same thing on every camera in a mixed set.
    pub fn angular_error(&self, world: Point3<f64>, observed: Point2<f64>) -> Option<f64> {
        let expected = self.ray(self.project(world)?);
        let actual = self.ray(observed);
        Some(expected.angle(&actual))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn intrinsics() -> Intrinsics {
        Intrinsics::from_fov(1280, 720, 70f64.to_radians())
    }

    fn camera() -> Camera {
        Camera::look_at(
            intrinsics(),
            Lens::default(),
            Point3::new(1.8, 2.4, 1.8),
            Point3::new(0.0, 1.0, 0.0),
            Vector3::y(),
        )
    }

    #[test]
    fn the_field_of_view_survives_the_round_trip() {
        let intrinsics = intrinsics();
        assert!((intrinsics.horizontal_fov().to_degrees() - 70.0).abs() < 1e-9);
    }

    #[test]
    fn a_pixel_is_worth_less_angle_on_a_longer_lens() {
        let wide = Intrinsics::from_fov(640, 480, 90f64.to_radians());
        let narrow = Intrinsics::from_fov(1920, 1080, 65f64.to_radians());
        assert!(
            wide.radians_per_pixel() > 4.0 * narrow.radians_per_pixel(),
            "a wide low-resolution pixel should cover several times the angle"
        );
    }

    #[test]
    fn the_camera_looks_where_it_was_pointed() {
        let camera = camera();
        let target = Point3::new(0.0, 1.0, 0.0);

        let pixel = camera.project(target).expect("the target is in front");
        assert!((pixel.x - camera.intrinsics.cx).abs() < 1e-6);
        assert!((pixel.y - camera.intrinsics.cy).abs() < 1e-6);
    }

    #[test]
    fn a_point_behind_the_camera_does_not_project() {
        let camera = camera();
        // Mirrored through the camera, so it sits directly behind it.
        let behind = Point3::new(3.6, 3.8, 3.6);
        assert!(camera.project(behind).is_none());
    }

    #[test]
    fn projecting_and_unprojecting_recovers_the_direction() {
        let camera = camera();

        for point in [
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(-0.5, 0.2, 0.7),
            Point3::new(0.4, 1.7, -0.3),
        ] {
            let pixel = camera.project(point).expect("in front of the camera");
            let ray = camera.ray(pixel);
            let expected = (point - camera.position()).normalize();
            assert!(
                ray.angle(&Unit::new_normalize(expected)) < 1e-9,
                "the ray through the projected pixel should point back at the point"
            );
        }
    }

    #[test]
    fn projection_round_trips_through_a_distorted_lens() {
        let camera = Camera::new(
            intrinsics(),
            Lens::RadialTangential {
                k1: -0.3,
                k2: 0.1,
                p1: 0.0005,
                p2: -0.0008,
            },
            camera().pose,
        );

        for point in [
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(-0.9, 0.1, 0.9),
            Point3::new(0.8, 1.9, -0.6),
        ] {
            let pixel = camera.project(point).expect("in front of the camera");
            let ray = camera.ray(pixel);
            let expected = Unit::new_normalize(point - camera.position());
            assert!(
                ray.angle(&expected) < 1e-6,
                "distortion should undo cleanly, off by {} rad",
                ray.angle(&expected)
            );
        }
    }

    #[test]
    fn angular_error_is_zero_for_a_perfect_observation() {
        let camera = camera();
        let point = Point3::new(-0.3, 0.6, 0.4);
        let pixel = camera.project(point).unwrap();

        let error = camera.angular_error(point, pixel).unwrap();
        assert!(error < 1e-12, "expected no error, got {error}");
    }

    #[test]
    fn angular_error_grows_with_the_miss() {
        let camera = camera();
        let point = Point3::new(-0.3, 0.6, 0.4);
        let pixel = camera.project(point).unwrap();

        let near = camera
            .angular_error(point, Point2::new(pixel.x + 1.0, pixel.y))
            .unwrap();
        let far = camera
            .angular_error(point, Point2::new(pixel.x + 10.0, pixel.y))
            .unwrap();

        assert!(far > near);
        // One pixel is one pixel's worth of angle, near the centre.
        assert!((near - camera.intrinsics.radians_per_pixel()).abs() < 1e-4);
    }
}
