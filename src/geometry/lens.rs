//! Lens models.
//!
//! A camera in a ceiling corner is often a wide one, and a wide lens does not
//! obey the pinhole model that everything downstream assumes. The lens model is
//! per camera, because a room is likely to mix an ordinary webcam with a wide
//! one and a single global model would be wrong for at least one of them.

use serde::{Deserialize, Serialize};

use crate::config::LensKind;

/// Distortion parameters, in the form matching the camera's lens kind.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "model", rename_all = "snake_case")]
pub enum Lens {
    /// Radial and tangential distortion, for ordinary and moderately wide
    /// lenses. The usual OpenCV `k1 k2 p1 p2` model.
    RadialTangential { k1: f64, k2: f64, p1: f64, p2: f64 },
    /// Equidistant projection, for lenses past roughly 120 degrees, where the
    /// radial model can no longer be fitted.
    Fisheye { k1: f64, k2: f64, k3: f64, k4: f64 },
}

impl Default for Lens {
    fn default() -> Self {
        Lens::RadialTangential {
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }
}

impl Lens {
    /// A zeroed lens of the shape a camera's configured kind calls for.
    pub fn for_kind(kind: LensKind) -> Self {
        match kind {
            LensKind::Standard | LensKind::Wide => Lens::default(),
            LensKind::Fisheye => Lens::Fisheye {
                k1: 0.0,
                k2: 0.0,
                k3: 0.0,
                k4: 0.0,
            },
        }
    }

    pub fn is_identity(&self) -> bool {
        match self {
            Lens::RadialTangential { k1, k2, p1, p2 } => {
                *k1 == 0.0 && *k2 == 0.0 && *p1 == 0.0 && *p2 == 0.0
            }
            Lens::Fisheye { k1, k2, k3, k4 } => {
                *k1 == 0.0 && *k2 == 0.0 && *k3 == 0.0 && *k4 == 0.0
            }
        }
    }

    /// The free parameters, for the calibration solver.
    pub fn parameters(&self) -> [f64; 4] {
        match self {
            Lens::RadialTangential { k1, k2, p1, p2 } => [*k1, *k2, *p1, *p2],
            Lens::Fisheye { k1, k2, k3, k4 } => [*k1, *k2, *k3, *k4],
        }
    }

    pub fn with_parameters(&self, values: [f64; 4]) -> Self {
        match self {
            Lens::RadialTangential { .. } => Lens::RadialTangential {
                k1: values[0],
                k2: values[1],
                p1: values[2],
                p2: values[3],
            },
            Lens::Fisheye { .. } => Lens::Fisheye {
                k1: values[0],
                k2: values[1],
                k3: values[2],
                k4: values[3],
            },
        }
    }

    /// Applies distortion to a normalized camera-plane point.
    ///
    /// The input is `(x/z, y/z)`; the output is what the sensor actually sees,
    /// still in normalized units.
    pub fn distort(&self, x: f64, y: f64) -> (f64, f64) {
        match self {
            Lens::RadialTangential { k1, k2, p1, p2 } => {
                let r2 = x * x + y * y;
                let radial = 1.0 + k1 * r2 + k2 * r2 * r2;
                (
                    x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
                    y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y,
                )
            }
            Lens::Fisheye { k1, k2, k3, k4 } => {
                let r = (x * x + y * y).sqrt();
                if r < 1e-12 {
                    return (x, y);
                }
                // Equidistant: the image radius follows the incidence angle
                // rather than its tangent, which is what lets a fisheye see
                // past 180 degrees at all.
                let theta = r.atan();
                let t2 = theta * theta;
                let distorted = theta
                    * (1.0 + k1 * t2 + k2 * t2 * t2 + k3 * t2 * t2 * t2 + k4 * t2 * t2 * t2 * t2);
                let scale = distorted / r;
                (x * scale, y * scale)
            }
        }
    }

    /// Removes distortion, by iterating the forward model.
    ///
    /// There is no closed form for either model, and a fixed number of
    /// iterations keeps the cost predictable; ten is far more than enough for
    /// the distortion levels a webcam has.
    pub fn undistort(&self, x: f64, y: f64) -> (f64, f64) {
        if self.is_identity() {
            return (x, y);
        }

        match self {
            Lens::RadialTangential { k1, k2, p1, p2 } => {
                // Solve for the undistorted point that maps to this one, by
                // repeatedly removing the tangential term and dividing out the
                // radial one. Twenty passes is far more than the distortion of
                // a webcam lens needs.
                let (mut ux, mut uy) = (x, y);
                for _ in 0..20 {
                    let r2 = ux * ux + uy * uy;
                    let radial = 1.0 + k1 * r2 + k2 * r2 * r2;
                    if radial.abs() < 1e-12 {
                        break;
                    }
                    let tangential_x = 2.0 * p1 * ux * uy + p2 * (r2 + 2.0 * ux * ux);
                    let tangential_y = p1 * (r2 + 2.0 * uy * uy) + 2.0 * p2 * ux * uy;
                    ux = (x - tangential_x) / radial;
                    uy = (y - tangential_y) / radial;
                }
                (ux, uy)
            }
            Lens::Fisheye { .. } => {
                let rd = (x * x + y * y).sqrt();
                if rd < 1e-12 {
                    return (x, y);
                }
                // Solve for the incidence angle that produces this radius, then
                // return to the pinhole plane through its tangent.
                let mut theta = rd;
                for _ in 0..10 {
                    let (dx, _) = self.distort(theta.tan(), 0.0);
                    let error = dx - rd;
                    if error.abs() < 1e-12 {
                        break;
                    }
                    // Numerical derivative: the analytic one buys nothing at
                    // ten iterations and is easy to get wrong.
                    let h = 1e-7;
                    let (dxh, _) = self.distort((theta + h).tan(), 0.0);
                    let slope = (dxh - dx) / h;
                    if slope.abs() < 1e-12 {
                        break;
                    }
                    theta -= error / slope;
                }
                let scale = theta.tan() / rd;
                (x * scale, y * scale)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: (f64, f64), b: (f64, f64), tolerance: f64) {
        assert!(
            (a.0 - b.0).abs() < tolerance && (a.1 - b.1).abs() < tolerance,
            "{a:?} is not within {tolerance} of {b:?}"
        );
    }

    #[test]
    fn an_undistorted_lens_changes_nothing() {
        let lens = Lens::default();
        assert_close(lens.distort(0.3, -0.2), (0.3, -0.2), 1e-12);
        assert_close(lens.undistort(0.3, -0.2), (0.3, -0.2), 1e-12);
    }

    #[test]
    fn radial_distortion_round_trips() {
        let lens = Lens::RadialTangential {
            k1: -0.28,
            k2: 0.09,
            p1: 0.001,
            p2: -0.002,
        };

        for (x, y) in [(0.0, 0.0), (0.2, 0.1), (-0.4, 0.35), (0.6, -0.55)] {
            let distorted = lens.distort(x, y);
            let recovered = lens.undistort(distorted.0, distorted.1);
            assert_close(recovered, (x, y), 1e-9);
        }
    }

    #[test]
    fn fisheye_distortion_round_trips() {
        let lens = Lens::Fisheye {
            k1: -0.02,
            k2: 0.004,
            k3: -0.001,
            k4: 0.0002,
        };

        for (x, y) in [(0.0, 0.0), (0.3, 0.2), (-0.8, 0.6), (1.4, -1.1)] {
            let distorted = lens.distort(x, y);
            let recovered = lens.undistort(distorted.0, distorted.1);
            assert_close(recovered, (x, y), 1e-6);
        }
    }

    /// A fisheye compresses the edges of the frame: a point far off axis lands
    /// closer to the centre than the pinhole model would put it.
    #[test]
    fn fisheye_pulls_the_edges_inward() {
        let lens = Lens::Fisheye {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            k4: 0.0,
        };
        let (x, _) = lens.distort(2.0, 0.0);
        assert!(x < 2.0, "an equidistant lens should compress, got {x}");
        assert!(x > 1.0);
    }

    #[test]
    fn parameters_round_trip_through_the_solver_representation() {
        let lens = Lens::Fisheye {
            k1: 1.0,
            k2: 2.0,
            k3: 3.0,
            k4: 4.0,
        };
        assert_eq!(lens.with_parameters(lens.parameters()), lens);
    }
}
