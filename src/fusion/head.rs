//! Checking the reconstruction against the one thing in the room that knows
//! where it is.
//!
//! The floor meter next door watches one number the application takes on trust.
//! This watches all three, and it is the sharper instrument, because the
//! headset reports its own position continuously and to a millimetre. The
//! cameras can see the user's head. Those are two independent answers to the
//! same question, and the difference between them is the total error of
//! everything in between: the camera calibration, the lens model, the room
//! transform, the pose models, and the clock they are all aligned on.
//!
//! It is not a circular check, though the cameras were solved from this same
//! headset. A calibration is solved once, from one walk, over whatever part of
//! the room the user happened to cover. This asks the same question afterwards,
//! continuously, wherever they are standing now — which is how a camera that
//! has been knocked, a room profile loaded for the wrong setup, or a solve that
//! converged on the wrong scale show up. All three look perfect from the
//! inside.
//!
//! The scale is the one worth dwelling on, because nothing else here can see
//! it. **A uniformly scaled set of cameras is perfectly self-consistent.** Every
//! ray still meets every other ray, every reprojection residual is still zero,
//! and the body that comes out is simply the wrong size — so the RMS the wizard
//! reports is happy, the agreement factor is 1.0, and the reconstruction is
//! two-thirds life size. Scale cannot be recovered from the cameras at all. It
//! needs an external metric reference, and there is exactly one in the room.
//!
//! Which is measured by *moving*, not by standing: how far the headset went
//! against how far the cameras thought the head went. A ratio of one is a room
//! solved to life size, and anything else is a body that will never line up
//! with an avatar however the trackers are calibrated in the game.
//!
//! Like the floor, it reports and does not correct. A head half a metre from
//! the headset does not mean the trackers should be shifted half a metre; it
//! means the room profile is wrong and everything built on it is untrustworthy,
//! including whatever correction would have been applied.

use std::collections::VecDeque;

use nalgebra::{Point3, Vector3};

use crate::models::Joint;

use super::fuse::Pose3d;

/// Where to look for the head, best first.
///
/// The headset sits in front of the face and above the ears, so none of these
/// is the headset's own position and no offset is subtracted. What is being
/// asked is not "are these the same point" but "are they the width of a head
/// apart or the width of a room".
const HEAD: [Joint; 4] = [Joint::Head, Joint::Nose, Joint::Neck, Joint::LeftEye];

#[derive(Debug, Clone)]
pub struct HeadOptions {
    /// Positional uncertainty a head keypoint must be under to count, in
    /// metres.
    pub max_sigma: f64,
    /// Samples kept, at the fusion rate.
    pub capacity: usize,
    /// Samples needed before the offset is offered at all.
    pub min_samples: usize,
    /// How far the two answers may be apart before it is worth saying
    /// something, in metres. A head is about this big, and the keypoint is
    /// somewhere on it.
    pub tolerance: f64,
    /// How far apart in samples two positions are compared for scale.
    ///
    /// Half a second at the default fusion rate: long enough that an ordinary
    /// movement clears the noise, short enough that most of a five-second
    /// window still yields a comparison.
    pub scale_lag: usize,
    /// How far the headset has to have moved between two samples for the pair
    /// to say anything about scale, in metres.
    pub scale_travel: f64,
    /// Comparisons needed before a scale is offered.
    pub scale_samples: usize,
}

impl Default for HeadOptions {
    fn default() -> Self {
        Self {
            max_sigma: 0.08,
            capacity: 300,
            min_samples: 60,
            tolerance: 0.30,
            scale_lag: 30,
            scale_travel: 0.10,
            scale_samples: 30,
        }
    }
}

/// One tick's pair of answers.
#[derive(Debug, Clone, Copy)]
struct Sighting {
    headset: Point3<f64>,
    head: Point3<f64>,
}

/// Watches how far the reconstructed head is from the headset, and whether it
/// travels as far.
#[derive(Debug, Clone)]
pub struct HeadMeter {
    options: HeadOptions,
    sightings: VecDeque<Sighting>,
}

impl Default for HeadMeter {
    fn default() -> Self {
        Self::new(HeadOptions::default())
    }
}

impl HeadMeter {
    pub fn new(options: HeadOptions) -> Self {
        Self {
            sightings: VecDeque::with_capacity(options.capacity),
            options,
        }
    }

    pub fn reset(&mut self) {
        self.sightings.clear();
    }

    /// Records one tick, if both answers are available.
    ///
    /// The *raw* reconstruction, never the fitted one, for the same reason the
    /// floor uses it: the fit places what nothing saw, and comparing an
    /// invented head against the headset measures the fit rather than the room.
    pub fn observe(&mut self, pose: &Pose3d, headset: Option<Point3<f64>>) {
        let Some(headset) = headset else {
            return;
        };
        let Some(head) = HEAD.iter().find_map(|joint| {
            pose.get(*joint)
                .filter(|seen| seen.sigma <= self.options.max_sigma)
                .map(|seen| seen.point)
        }) else {
            return;
        };

        if self.sightings.len() == self.options.capacity {
            self.sightings.pop_front();
        }
        self.sightings.push_back(Sighting { headset, head });
    }

    /// The typical offset from the headset to the reconstructed head, in
    /// metres.
    ///
    /// Median per axis, which is the robust version of the average and is what
    /// makes one badly triangulated frame not the verdict. Kept as a vector
    /// rather than a distance because the *direction* is most of the diagnosis:
    /// an offset that is nearly all vertical is a room setup run at the wrong
    /// height, and one that points anywhere else is the calibration.
    pub fn estimate(&self) -> Option<Vector3<f64>> {
        if self.sightings.len() < self.options.min_samples {
            return None;
        }

        let mut axis = Vector3::zeros();
        for index in 0..3 {
            let mut values: Vec<f64> = self
                .sightings
                .iter()
                .map(|sighting| sighting.head[index] - sighting.headset[index])
                .collect();
            values.sort_by(f64::total_cmp);
            axis[index] = values[values.len() / 2];
        }
        Some(axis)
    }

    /// How large the room is reconstructed, as a multiple of life size.
    ///
    /// The only measurement here that can see a scale error, and the only one
    /// anywhere in the application that can. A uniformly scaled set of cameras
    /// agrees with itself perfectly — every ray still meets, every residual is
    /// still zero — so no amount of looking at the cameras will ever find it.
    /// What finds it is that the headset is a metre rule: it went this far, and
    /// the cameras thought the head went that far.
    ///
    /// Needs the user to move. Standing still returns `None`, which is the
    /// honest answer rather than a ratio of two noise floors.
    pub fn scale(&self) -> Option<f64> {
        let lag = self.options.scale_lag.max(1);
        if self.sightings.len() <= lag {
            return None;
        }

        let mut ratios = Vec::with_capacity(self.sightings.len());
        for index in 0..self.sightings.len() - lag {
            let from = self.sightings[index];
            let to = self.sightings[index + lag];

            let travelled = (to.headset - from.headset).norm();
            if travelled < self.options.scale_travel {
                continue;
            }
            ratios.push((to.head - from.head).norm() / travelled);
        }

        if ratios.len() < self.options.scale_samples {
            return None;
        }
        ratios.sort_by(f64::total_cmp);
        Some(ratios[ratios.len() / 2])
    }

    /// How far apart the two answers are, when they are far enough apart to
    /// matter.
    pub fn disagreement(&self) -> Option<Vector3<f64>> {
        self.estimate()
            .filter(|offset| offset.norm() > self.options.tolerance)
    }

    pub fn samples(&self) -> usize {
        self.sightings.len()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::fusion::fuse::FusedJoint;

    fn pose(head: Point3<f64>, sigma: f64) -> Pose3d {
        let mut pose = Pose3d::empty(Instant::now());
        pose.set(
            Joint::Head,
            FusedJoint {
                point: head,
                sigma,
                residual: 0.0,
                weights: Vec::new(),
                rejected: Vec::new(),
            },
        );
        pose
    }

    /// A user walking back and forth, with the cameras reconstructing their
    /// movement at `scale` of life size about the room origin.
    fn walk(meter: &mut HeadMeter, scale: f64) {
        for tick in 0..300 {
            let along = (tick as f64 * 0.02).sin() * 1.2;
            let headset = Point3::new(along, 1.6, 0.0);
            let head = Point3::new(along * scale, 1.6 * scale, 0.0);
            meter.observe(&pose(head, 0.01), Some(headset));
        }
    }

    #[test]
    fn a_head_where_the_headset_is_reports_nothing_worth_saying() {
        let mut meter = HeadMeter::default();
        let headset = Point3::new(0.0, 1.6, 0.0);

        for _ in 0..200 {
            // A hand's width away, which is where a head keypoint sits when
            // everything is right.
            meter.observe(&pose(Point3::new(0.0, 1.68, -0.06), 0.01), Some(headset));
        }

        assert!(meter.estimate().is_some());
        assert_eq!(meter.disagreement(), None);
    }

    #[test]
    fn a_room_solved_half_a_metre_low_says_so_and_says_it_is_vertical() {
        let mut meter = HeadMeter::default();
        let headset = Point3::new(0.0, 1.6, 0.0);

        for _ in 0..200 {
            meter.observe(&pose(Point3::new(0.0, 1.12, -0.06), 0.01), Some(headset));
        }

        let offset = meter.disagreement().expect("half a metre is worth saying");
        assert!(offset.y < -0.4, "expected a drop, got {offset:?}");
        assert!(
            offset.y.abs() > (offset.x.abs() + offset.z.abs()) * 3.0,
            "the offset should read as vertical: {offset:?}"
        );
    }

    #[test]
    fn a_head_nothing_can_place_is_not_evidence() {
        let mut meter = HeadMeter::default();
        let headset = Point3::new(0.0, 1.6, 0.0);

        for _ in 0..200 {
            meter.observe(&pose(Point3::new(0.0, 0.2, 0.0), 0.4), Some(headset));
        }

        assert_eq!(meter.samples(), 0);
        assert_eq!(meter.estimate(), None);
    }

    #[test]
    fn one_bad_frame_is_not_the_verdict() {
        let mut meter = HeadMeter::default();
        let headset = Point3::new(0.0, 1.6, 0.0);

        for tick in 0..200 {
            let head = if tick == 100 {
                Point3::new(4.0, 4.0, 4.0)
            } else {
                Point3::new(0.0, 1.68, -0.06)
            };
            meter.observe(&pose(head, 0.01), Some(headset));
        }

        assert_eq!(meter.disagreement(), None);
    }

    /// The measurement nothing else in the application can make.
    #[test]
    fn a_room_solved_two_thirds_life_size_is_caught_by_walking_about() {
        let mut meter = HeadMeter::default();
        walk(&mut meter, 0.63);

        let scale = meter.scale().expect("a walk is enough to measure scale");
        assert!(
            (scale - 0.63).abs() < 0.02,
            "expected about two thirds, got {scale:.3}"
        );
    }

    #[test]
    fn a_room_solved_to_life_size_reads_as_one() {
        let mut meter = HeadMeter::default();
        walk(&mut meter, 1.0);

        let scale = meter.scale().expect("a walk is enough to measure scale");
        assert!((scale - 1.0).abs() < 0.02, "expected one, got {scale:.3}");
    }

    /// Standing still is two noise floors divided by each other, and the ratio
    /// of those means nothing at all.
    #[test]
    fn standing_still_says_nothing_about_scale() {
        let mut meter = HeadMeter::default();
        let headset = Point3::new(0.0, 1.6, 0.0);

        for _ in 0..300 {
            meter.observe(&pose(Point3::new(0.0, 1.68, -0.06), 0.01), Some(headset));
        }

        assert_eq!(meter.scale(), None);
    }
}
