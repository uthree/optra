//! Measuring the floor the user is standing on.
//!
//! Everything Optra computes is expressed against SteamVR's idea of the floor,
//! and nothing in the pipeline ever checks that idea. The cameras are solved
//! from headset positions, so a room setup run with the headset resting on a
//! desk gives a floor that is wrong by the height of the desk — and the whole
//! solve slides with it, staying perfectly self-consistent the entire way. The
//! reprojection error stays low, the reconstructed room looks right, and the
//! feet come out underground.
//!
//! But the cameras can see the feet, and a foot on the ground *is* the floor.
//! That makes this an independent measurement of a quantity the rest of the
//! application takes on trust, which is exactly the kind of measurement worth
//! having.
//!
//! It reports and does not correct. A floor that disagrees means the room setup
//! is wrong, and quietly compensating for it here would leave the user with a
//! working Optra and a broken SteamVR — every other application on the machine
//! would still put them a foot underground.

use std::collections::VecDeque;

use crate::models::Joint;

use super::fuse::Pose3d;

/// Joints that touch the ground.
///
/// The toe and heel rather than the ankle: the ankle sits some way above the
/// sole, by an amount that varies with the person and their shoes, and using it
/// would build that error into the answer.
const GROUNDED: [Joint; 4] = [
    Joint::LeftHeel,
    Joint::RightHeel,
    Joint::LeftBigToe,
    Joint::RightBigToe,
];

#[derive(Debug, Clone)]
pub struct FloorOptions {
    /// Positional uncertainty a foot must be under to count, in metres.
    pub max_sigma: f64,
    /// Samples kept. At sixty a second this is a few seconds of walking, which
    /// is enough for several footfalls.
    pub capacity: usize,
    /// Samples needed before the estimate is offered at all.
    pub min_samples: usize,
    /// How far the measured floor may sit from zero before it is worth saying
    /// something, in metres.
    pub tolerance: f64,
}

impl Default for FloorOptions {
    fn default() -> Self {
        Self {
            max_sigma: 0.05,
            capacity: 600,
            min_samples: 120,
            tolerance: 0.06,
        }
    }
}

/// Watches where the feet actually end up.
#[derive(Debug, Clone)]
pub struct FloorMeter {
    options: FloorOptions,
    /// Lowest confident foot on each tick, in metres.
    lows: VecDeque<f64>,
}

impl Default for FloorMeter {
    fn default() -> Self {
        Self::new(FloorOptions::default())
    }
}

impl FloorMeter {
    pub fn new(options: FloorOptions) -> Self {
        Self {
            lows: VecDeque::with_capacity(options.capacity),
            options,
        }
    }

    pub fn reset(&mut self) {
        self.lows.clear();
    }

    /// Records the lowest foot in one reconstruction.
    ///
    /// The *raw* reconstruction, never the fitted one. The fit holds every
    /// joint above its own idea of the floor, so measuring its output would
    /// return that idea back unchanged — a loop that confirms whatever it was
    /// told.
    pub fn observe(&mut self, pose: &Pose3d) {
        let lowest = GROUNDED
            .iter()
            .filter_map(|joint| pose.get(*joint))
            .filter(|foot| foot.sigma <= self.options.max_sigma)
            .map(|foot| foot.point.y)
            .fold(f64::INFINITY, f64::min);

        if !lowest.is_finite() {
            return;
        }

        if self.lows.len() == self.options.capacity {
            self.lows.pop_front();
        }
        self.lows.push_back(lowest);
    }

    /// Where the floor appears to be, in metres, relative to where SteamVR
    /// says it is.
    ///
    /// The tenth percentile of the lowest-foot heights rather than the median.
    /// A foot is only on the ground for part of each stride and is in the air
    /// for the rest, so the middle of the distribution is a swinging foot, not
    /// the floor. Not the outright minimum either, which is whichever single
    /// frame the triangulation was most wrong on.
    pub fn estimate(&self) -> Option<f64> {
        if self.lows.len() < self.options.min_samples {
            return None;
        }

        let mut sorted: Vec<f64> = self.lows.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        Some(sorted[sorted.len() / 10])
    }

    /// How far off SteamVR's floor is, when it is far enough off to matter.
    pub fn disagreement(&self) -> Option<f64> {
        self.estimate()
            .filter(|floor| floor.abs() > self.options.tolerance)
    }

    pub fn samples(&self) -> usize {
        self.lows.len()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use nalgebra::Point3;

    use super::*;
    use crate::fusion::fuse::FusedJoint;

    /// One tick of a walking body whose feet reach `floor` at the bottom of
    /// each stride.
    fn stride(step: usize, floor: f64, sigma: f64) -> Pose3d {
        let mut pose = Pose3d::empty(Instant::now());
        let phase = step as f64 * 0.31;

        for (heel, toe, offset) in [
            (Joint::LeftHeel, Joint::LeftBigToe, 0.0),
            (Joint::RightHeel, Joint::RightBigToe, std::f64::consts::PI),
        ] {
            // A foot spends part of the stride down and the rest in the air.
            let lift = 0.18 * (phase + offset).sin().max(0.0);
            for joint in [heel, toe] {
                pose.set(
                    joint,
                    FusedJoint {
                        point: Point3::new(0.1, floor + lift, 0.0),
                        sigma,
                        residual: 0.0,
                        weights: vec![(0, 0.5), (1, 0.5)],
                        rejected: Vec::new(),
                    },
                );
            }
        }
        pose
    }

    #[test]
    fn a_floor_that_agrees_says_nothing() {
        let mut meter = FloorMeter::default();
        for step in 0..400 {
            meter.observe(&stride(step, 0.0, 0.01));
        }

        assert!(meter.estimate().unwrap().abs() < 0.02);
        assert_eq!(meter.disagreement(), None);
    }

    /// The case this exists for: a room setup run with the headset on a desk
    /// puts the floor most of a metre too high, and every foot underground.
    #[test]
    fn a_floor_set_too_high_is_measured() {
        let mut meter = FloorMeter::default();
        for step in 0..400 {
            meter.observe(&stride(step, -0.68, 0.01));
        }

        let measured = meter
            .disagreement()
            .expect("that is far too much to ignore");
        assert!(
            (measured + 0.68).abs() < 0.03,
            "measured {measured} instead of -0.68"
        );
    }

    /// Walking normally, one foot is planted whenever the other is not, so the
    /// lowest foot is on the ground almost always and any sensible statistic
    /// finds it. The case that separates them is a leg the cameras only manage
    /// to see part of the time — behind the other one for the rest — which
    /// leaves a record of a foot that is mostly in the air.
    #[test]
    fn a_foot_seen_mostly_in_the_air_does_not_raise_the_floor() {
        let mut meter = FloorMeter::default();

        for step in 0..600 {
            let mut pose = Pose3d::empty(Instant::now());
            let lift = 0.09 * (1.0 - (step as f64 * 0.31).cos());
            for joint in [Joint::LeftHeel, Joint::LeftBigToe] {
                pose.set(
                    joint,
                    FusedJoint {
                        point: Point3::new(0.1, lift, 0.0),
                        sigma: 0.01,
                        residual: 0.0,
                        weights: vec![(0, 0.5), (1, 0.5)],
                        rejected: Vec::new(),
                    },
                );
            }
            meter.observe(&pose);
        }

        let mut sorted: Vec<f64> = meter.lows.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        let median = sorted[sorted.len() / 2];

        let estimate = meter.estimate().unwrap();
        assert!(
            estimate < 0.03,
            "the floor came out at {estimate}, well above the ground"
        );
        assert!(
            median > estimate + 0.05,
            "the median should be a swinging foot, and was {median}"
        );
    }

    #[test]
    fn feet_nobody_could_locate_are_not_measured() {
        let mut meter = FloorMeter::default();
        for step in 0..400 {
            meter.observe(&stride(step, -0.68, 0.4));
        }

        assert_eq!(meter.samples(), 0);
        assert_eq!(meter.estimate(), None);
    }

    #[test]
    fn a_moment_of_walking_is_not_enough_to_judge_by() {
        let mut meter = FloorMeter::default();
        for step in 0..20 {
            meter.observe(&stride(step, -0.68, 0.01));
        }

        assert_eq!(meter.estimate(), None);
    }
}
