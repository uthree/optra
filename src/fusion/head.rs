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
    /// Samples needed before the estimate is offered at all.
    pub min_samples: usize,
    /// How far the two answers may be apart before it is worth saying
    /// something, in metres. A head is about this big, and the keypoint is
    /// somewhere on it.
    pub tolerance: f64,
}

impl Default for HeadOptions {
    fn default() -> Self {
        Self {
            max_sigma: 0.08,
            capacity: 300,
            min_samples: 60,
            tolerance: 0.30,
        }
    }
}

/// Watches how far the reconstructed head is from the headset.
#[derive(Debug, Clone)]
pub struct HeadMeter {
    options: HeadOptions,
    /// Reconstructed head minus headset, per tick, in metres.
    offsets: VecDeque<Vector3<f64>>,
}

impl Default for HeadMeter {
    fn default() -> Self {
        Self::new(HeadOptions::default())
    }
}

impl HeadMeter {
    pub fn new(options: HeadOptions) -> Self {
        Self {
            offsets: VecDeque::with_capacity(options.capacity),
            options,
        }
    }

    pub fn reset(&mut self) {
        self.offsets.clear();
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

        if self.offsets.len() == self.options.capacity {
            self.offsets.pop_front();
        }
        self.offsets.push_back(head - headset);
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
        if self.offsets.len() < self.options.min_samples {
            return None;
        }

        let mut axis = Vector3::zeros();
        for index in 0..3 {
            let mut values: Vec<f64> = self.offsets.iter().map(|offset| offset[index]).collect();
            values.sort_by(f64::total_cmp);
            axis[index] = values[values.len() / 2];
        }
        Some(axis)
    }

    /// How far apart the two answers are, when they are far enough apart to
    /// matter.
    pub fn disagreement(&self) -> Option<Vector3<f64>> {
        self.estimate()
            .filter(|offset| offset.norm() > self.options.tolerance)
    }

    pub fn samples(&self) -> usize {
        self.offsets.len()
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
}
