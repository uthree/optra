//! How far behind each camera is.
//!
//! A USB webcam hands over a frame tens of milliseconds after the light landed
//! on its sensor, and the delay differs per device — a cheap camera and a good
//! one in the same room are not looking at the same instant. Nothing reports
//! this number, so it has to be measured.
//!
//! The measurement is a search rather than a correlation: for a range of
//! candidate delays, the recorded device track is sampled that far *back* from
//! each frame's timestamp and reprojected, and the delay that explains the
//! pixels best wins. That is the same objective the bundle refinement
//! minimizes, scanned over a time shift, so it uses the geometry already
//! solved instead of a separate notion of similarity.
//!
//! It only works while the user is moving. A delay is invisible against a
//! stationary head, so the estimate carries how sharply the error rose either
//! side of the answer, and a flat curve means the walk could not tell.

use std::time::Duration;

use nalgebra::{Point3, Vector3};
use serde::{Deserialize, Serialize};

use crate::geometry::camera::Camera;

use super::recorder::{CameraTrail, Recording};

#[derive(Debug, Clone)]
pub struct LatencyOptions {
    /// Longest delay to consider. Past this a camera is broken rather than
    /// slow, and a search that wide starts finding spurious minima where the
    /// walk happened to repeat itself.
    pub max: Duration,
    /// Resolution of the search.
    pub step: Duration,
    /// How far either side of the answer the error is checked, to judge whether
    /// the walk pinned it down at all.
    pub probe: Duration,
    /// Samples a camera needs before its delay is worth estimating.
    pub min_samples: usize,
}

impl Default for LatencyOptions {
    fn default() -> Self {
        Self {
            max: Duration::from_millis(250),
            step: Duration::from_millis(2),
            probe: Duration::from_millis(30),
            min_samples: 60,
        }
    }
}

/// What one camera's delay came out as.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Estimate {
    pub latency: Duration,
    /// RMS angular reprojection error at that delay, in radians.
    pub rms: f64,
    /// The same with no delay applied. The difference is what the estimate
    /// bought.
    pub rms_uncorrected: f64,
    /// How much worse the fit gets when the delay is moved by `probe` either
    /// way, in pixels of this camera.
    ///
    /// Measured in pixels rather than as a fraction of the best fit, because a
    /// fit that is nearly exact makes any relative measure enormous — a
    /// stationary head fits perfectly at every delay, and a ratio would call
    /// that the most confident result of all.
    pub sharpness: f64,
}

impl Estimate {
    pub fn millis(&self) -> f64 {
        self.latency.as_secs_f64() * 1000.0
    }

    /// Whether the walk actually constrained the delay.
    ///
    /// A user who ambles produces a curve so flat that the minimum is noise.
    /// Applying that to tracking would be worse than applying nothing. One
    /// pixel is about where the rise stops being distinguishable from the
    /// keypoint noise it is competing with.
    pub fn is_confident(&self) -> bool {
        self.sharpness > 1.0
    }

    /// Whether this delay is one a webcam could plausibly have.
    ///
    /// Tens of milliseconds is a USB camera. Approaching a fifth of a second is
    /// a camera buffering frames, a starved capture thread, or a search that
    /// wandered — and applying it would be worse than applying nothing, so it
    /// is worth saying out loud rather than quietly discarding.
    pub fn is_plausible(&self) -> bool {
        self.latency < Duration::from_millis(120)
    }
}

/// Shortest reach the confidence probe will use, either side of the answer.
///
/// Below this the scaling back to a full probe amplifies whatever noise the
/// two samples carried more than it recovers signal.
const MIN_REACH: Duration = Duration::from_millis(8);

/// Estimates the delay of one camera against the recorded device tracks.
pub fn estimate(
    camera: &Camera,
    trail: &CameraTrail,
    recording: &Recording,
    offsets: &[Vector3<f64>],
    options: &LatencyOptions,
) -> Option<Estimate> {
    if trail.samples.len() < options.min_samples || options.step.is_zero() {
        return None;
    }

    let steps = (options.max.as_secs_f64() / options.step.as_secs_f64()).round() as usize;
    let lag = |index: usize| options.step.mul_f64(index as f64);

    let mut errors = Vec::with_capacity(steps + 1);
    for index in 0..=steps {
        errors.push(rms_at(camera, trail, recording, offsets, lag(index))?);
    }

    let (best, &lowest) = errors
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))?;

    // The search grid is two milliseconds across, and the true delay does not
    // land on it. A parabola through the winner and its neighbours puts the
    // minimum between them.
    let refined = if best > 0 && best + 1 < errors.len() {
        let (left, right) = (errors[best - 1], errors[best + 1]);
        let curvature = left - 2.0 * lowest + right;
        let shift = if curvature.abs() > 1e-18 {
            (0.5 * (left - right) / curvature).clamp(-0.5, 0.5)
        } else {
            0.0
        };
        lag(best).as_secs_f64() + shift * options.step.as_secs_f64()
    } else {
        lag(best).as_secs_f64()
    };

    let latency = Duration::from_secs_f64(refined.max(0.0));
    let rms = rms_at(camera, trail, recording, offsets, latency)?;
    let rms_uncorrected = errors[0];

    // How much worse the fit gets a little either side, which is what says
    // whether the minimum is a minimum or a dip in noise.
    //
    // The probe wants room on both sides of the answer, and a camera quicker
    // than the probe distance does not have it — a frame cannot arrive before
    // the light landed on the sensor, so there is nothing below zero to sample.
    // Refusing to judge in that case is what this used to do, and it meant
    // every camera under thirty milliseconds reported as unmeasurable however
    // sharply its delay was pinned down. Instead the reach shrinks to what
    // fits, and the rise is scaled back up to what the full probe would have
    // shown: near its minimum the curve is quadratic, so the rise goes as the
    // square of the distance.
    let reach = options.probe.min(latency);
    let (worse, over) = if reach >= MIN_REACH {
        let before = latency
            .checked_sub(reach)
            .and_then(|shifted| rms_at(camera, trail, recording, offsets, shifted));
        let after = rms_at(camera, trail, recording, offsets, latency + reach);
        match (before, after) {
            (Some(before), Some(after)) => (Some(before.min(after)), reach),
            _ => (None, reach),
        }
    } else {
        // Pinned against zero, where the boundary is real rather than an edge
        // the search ran into: the delay cannot be negative. The rise above it
        // is the whole story, and it is enough of one.
        (
            rms_at(camera, trail, recording, offsets, latency + options.probe),
            options.probe,
        )
    };

    let sharpness = worse
        .map(|worse| {
            let scale = (options.probe.as_secs_f64() / over.as_secs_f64()).powi(2);
            (worse - rms) * scale / camera.intrinsics.radians_per_pixel()
        })
        .unwrap_or(0.0);

    Some(Estimate {
        latency,
        rms,
        rms_uncorrected,
        sharpness,
    })
}

/// RMS angular reprojection error with the device tracks sampled `lag` back.
fn rms_at(
    camera: &Camera,
    trail: &CameraTrail,
    recording: &Recording,
    offsets: &[Vector3<f64>],
    lag: Duration,
) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;

    for sample in &trail.samples {
        let Some(when) = sample.at.checked_sub(lag) else {
            continue;
        };
        let Some(track) = recording.tracks.get(sample.rig) else {
            continue;
        };
        let Some(offset) = offsets.get(sample.rig) else {
            continue;
        };
        let Some(anchor) = track.at(when) else {
            continue;
        };

        let world = anchor * Point3::from(*offset);
        let Some(residual) = camera.angular_error(world, sample.pixel) else {
            continue;
        };

        sum += residual * residual;
        count += 1;
    }

    // A lag that puts most of the walk outside the recorded track would be
    // compared against a different, easier set of samples than its neighbours.
    (count * 2 >= trail.samples.len()).then(|| (sum / count.max(1) as f64).sqrt())
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};

    use super::*;
    use crate::calib::recorder::{CameraTrail, Recording, Rig, Sample};
    use crate::geometry::camera::Intrinsics;
    use crate::geometry::lens::Lens;
    use crate::models::keypoints::Joint;
    use crate::vr::{Role, Track};

    const OFFSET: Vector3<f64> = Vector3::new(0.01, 0.06, 0.13);

    fn camera() -> Camera {
        Camera::look_at(
            Intrinsics::from_fov(1280, 720, 72f64.to_radians()),
            Lens::default(),
            Point3::new(-1.9, 2.45, -1.9),
            Point3::new(0.0, 1.0, 0.0),
            Vector3::y(),
        )
    }

    /// A head moving at the speed a calibration walk moves at.
    fn head(t: f64, speed: f64) -> Isometry3<f64> {
        let t = t * speed;
        Isometry3::from_parts(
            Translation3::new(
                1.2 * t.sin(),
                1.4 + 0.25 * (1.7 * t).sin(),
                1.0 * (0.7 * t).cos(),
            ),
            UnitQuaternion::from_euler_angles(0.2 * (1.3 * t).sin(), 0.9 * t, 0.0),
        )
    }

    /// Builds a recording where the camera's frames really are `delay` late.
    fn recording(delay: Duration, speed: f64) -> (Recording, CameraTrail) {
        let camera = camera();
        let start = Instant::now() + Duration::from_secs(10);

        let mut track = Track::default();
        for step in 0..2000 {
            let at = start + Duration::from_millis(step * 4);
            track.push(at, head(step as f64 * 0.004, speed));
        }

        let mut trail = CameraTrail::new("cam0".to_owned());
        trail.width = 1280;
        trail.height = 720;

        for step in 0..500 {
            // The frame is stamped when it arrived, which is `delay` after the
            // instant it actually shows.
            let exposed = start + Duration::from_millis(step * 15 + 300);
            let stamped = exposed + delay;

            let Some(anchor) = track.at(exposed) else {
                continue;
            };
            let Some(pixel) = camera.project(anchor * Point3::from(OFFSET)) else {
                continue;
            };

            trail.record(Sample {
                at: stamped,
                rig: 0,
                pixel,
                confidence: 0.9,
            });
        }

        let recording = Recording {
            rigs: vec![Rig {
                role: Role::Head,
                joint: Joint::Head,
            }],
            tracks: vec![track],
            cameras: vec![trail.clone()],
            duration: Duration::from_secs(8),
        };

        (recording, trail)
    }

    #[test]
    fn a_known_delay_is_recovered() {
        for millis in [0u64, 17, 48, 96] {
            let delay = Duration::from_millis(millis);
            let (recording, trail) = recording(delay, 1.0);

            let estimate = estimate(
                &camera(),
                &trail,
                &recording,
                &[OFFSET],
                &LatencyOptions::default(),
            )
            .expect("the walk is long enough to measure");

            assert!(
                (estimate.millis() - millis as f64).abs() < 2.0,
                "a {millis} ms delay came out as {:.1} ms",
                estimate.millis()
            );
            assert!(
                estimate.rms < 1e-4,
                "the corrected fit should be near exact, got {}",
                estimate.rms
            );
        }
    }

    /// The delay only shows because the head moves. Correcting for it should
    /// visibly beat not correcting for it.
    #[test]
    fn correcting_for_the_delay_beats_ignoring_it() {
        let (recording, trail) = recording(Duration::from_millis(60), 1.0);
        let estimate = estimate(
            &camera(),
            &trail,
            &recording,
            &[OFFSET],
            &LatencyOptions::default(),
        )
        .unwrap();

        assert!(
            estimate.rms_uncorrected > estimate.rms * 100.0,
            "uncorrected {} against corrected {}",
            estimate.rms_uncorrected,
            estimate.rms
        );
        assert!(estimate.is_confident());
    }

    /// A camera quicker than the probe distance has no room below its answer to
    /// sample, and used to be reported as unmeasurable for that reason alone —
    /// so the delays most worth trusting were the ones always thrown away.
    #[test]
    fn a_camera_quicker_than_the_probe_is_still_measured() {
        for millis in [0u64, 6, 14, 25] {
            let delay = Duration::from_millis(millis);
            let (recording, trail) = recording(delay, 1.0);

            let estimate = estimate(
                &camera(),
                &trail,
                &recording,
                &[OFFSET],
                &LatencyOptions::default(),
            )
            .expect("the walk is long enough to measure");

            assert!(
                (estimate.millis() - millis as f64).abs() < 2.0,
                "a {millis} ms delay came out as {:.1} ms",
                estimate.millis()
            );
            assert!(
                estimate.is_confident(),
                "a {millis} ms delay on a brisk walk should be trusted, sharpness was {}",
                estimate.sharpness
            );
        }
    }

    /// Tens of milliseconds is a webcam. A fifth of a second is a camera
    /// buffering, a starved capture thread, or a search that wandered.
    #[test]
    fn an_absurd_delay_is_not_treated_as_plausible() {
        let (recording, trail) = recording(Duration::from_millis(30), 1.0);
        let estimate = estimate(
            &camera(),
            &trail,
            &recording,
            &[OFFSET],
            &LatencyOptions::default(),
        )
        .unwrap();

        assert!(estimate.is_plausible());
        assert!(
            !Estimate {
                latency: Duration::from_millis(194),
                ..estimate
            }
            .is_plausible()
        );
    }

    /// A user who ambles leaves a curve so flat that its minimum is noise. The
    /// estimate has to say so rather than hand back a confident wrong number.
    #[test]
    fn a_slow_walk_cannot_pin_the_delay_down() {
        let (recording, trail) = recording(Duration::from_millis(60), 0.02);
        let estimate = estimate(
            &camera(),
            &trail,
            &recording,
            &[OFFSET],
            &LatencyOptions::default(),
        )
        .unwrap();

        assert!(
            !estimate.is_confident(),
            "a near-stationary walk should not be trusted, sharpness was {}",
            estimate.sharpness
        );
    }

    #[test]
    fn too_few_samples_are_not_estimated() {
        let (recording, mut trail) = recording(Duration::from_millis(40), 1.0);
        trail.samples.truncate(10);

        assert!(
            estimate(
                &camera(),
                &trail,
                &recording,
                &[OFFSET],
                &LatencyOptions::default()
            )
            .is_none()
        );
    }
}
