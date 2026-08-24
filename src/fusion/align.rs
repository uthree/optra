//! Putting every camera on the same clock.
//!
//! Cameras do not agree about when anything happened. They run at different
//! rates, their frames arrive on no particular phase, and each one hands its
//! frames over a different amount late — a delay the calibration measured.
//! Triangulating a joint from rays taken at three different instants
//! reconstructs a position the body was never in, and during a step the error
//! is larger than the thing being measured.
//!
//! So the fusion clock picks an instant and asks every camera what it saw
//! *then*, which is a question no camera can answer directly. The answer is
//! interpolated between the two frames either side of it, and how much that
//! interpolation is trusted depends on how far it had to reach and how fast the
//! joint was moving while it reached.

use std::time::{Duration, Instant};

use nalgebra::Point2;

use crate::geometry::triangulate::pixel_sigma;
use crate::models::Joint;
use crate::pipeline::PoseFrame;

/// One camera's keypoints as they were at the fusion tick.
#[derive(Debug, Clone)]
pub struct Aligned {
    joints: [Option<AlignedJoint>; Joint::ALL.len()],
    /// Distance from the tick to the nearer of the two frames it fell between.
    pub gap: Duration,
    /// How far apart those two frames were, which is this camera's frame
    /// interval unless it dropped one.
    pub bracket: Duration,
}

/// One joint, resampled onto the tick.
#[derive(Debug, Clone, Copy)]
pub struct AlignedJoint {
    pub pixel: Point2<f64>,
    pub confidence: f64,
    /// Multiplier on this ray's angular uncertainty, earned by having been
    /// interpolated rather than observed.
    ///
    /// One means the tick landed on a real frame. It grows with the distance
    /// the interpolation had to reach and with how fast the joint was moving,
    /// which is what makes a 30 fps camera quietly lose its vote during a step
    /// without being switched off between them.
    pub penalty: f64,
}

impl Aligned {
    pub fn get(&self, joint: Joint) -> Option<AlignedJoint> {
        self.joints[joint.index()]
    }

    pub fn iter(&self) -> impl Iterator<Item = (Joint, AlignedJoint)> + '_ {
        Joint::ALL
            .iter()
            .filter_map(|joint| self.get(*joint).map(|aligned| (*joint, aligned)))
    }

    pub fn count(&self) -> usize {
        self.joints.iter().filter(|joint| joint.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }
}

/// Resamples a camera's keypoints at `at`, between the two frames that bracket
/// it.
///
/// `at` is expected to lie within `[before, after]`; a caller that has no
/// bracketing pair has nothing to align and should skip the camera for this
/// tick rather than reach for the nearest frame.
pub fn align(before: &PoseFrame, after: &PoseFrame, at: Instant) -> Aligned {
    let bracket = after
        .captured_at
        .saturating_duration_since(before.captured_at);
    let into = at.saturating_duration_since(before.captured_at);
    let alpha = if bracket.is_zero() {
        0.0
    } else {
        (into.as_secs_f64() / bracket.as_secs_f64()).clamp(0.0, 1.0)
    };

    let to_after = bracket.saturating_sub(into);
    let gap = into.min(to_after);

    // How fast the body was moving across the frame, in pixels per second,
    // measured from the joints both frames agree on. A joint that only one of
    // them has cannot report its own speed, and this is the closest honest
    // stand-in: the body it belongs to moved this fast.
    let typical = body_speed(before, after, bracket);

    let mut joints = [None; Joint::ALL.len()];
    for joint in Joint::ALL {
        let start = before.keypoints.get(joint);
        let end = after.keypoints.get(joint);

        joints[joint.index()] = match (start, end) {
            (Some(start), Some(end)) => {
                let from = point(start.x, start.y);
                let to = point(end.x, end.y);
                let speed = if bracket.is_zero() {
                    0.0
                } else {
                    (to - from).norm() / bracket.as_secs_f64()
                };

                // The weaker end sets the confidence. A joint the model was
                // sure of in one frame and unsure of in the next is not
                // half-sure in between; the interpolation is only as sound as
                // the shakier observation it rests on.
                let confidence = (start.confidence.min(end.confidence)) as f64;

                Some(AlignedJoint {
                    pixel: from + (to - from) * alpha,
                    confidence,
                    penalty: penalty(interpolation_drift(speed, gap, bracket), confidence),
                })
            }
            // Present in only one of the two frames, which happens whenever a
            // limb crosses the model's confidence threshold. Using the frame
            // that has it keeps a camera's vote on a joint the others may not
            // see at all; the reach is the whole way to that frame, so it pays
            // for the distance.
            (Some(only), None) => Some(one_sided(only, into, typical)),
            (None, Some(only)) => Some(one_sided(only, to_after, typical)),
            (None, None) => None,
        };
    }

    Aligned {
        joints,
        gap,
        bracket,
    }
}

fn one_sided(
    keypoint: crate::infer::traits::Keypoint,
    reach: Duration,
    speed: f64,
) -> AlignedJoint {
    let confidence = keypoint.confidence as f64;
    AlignedJoint {
        pixel: point(keypoint.x, keypoint.y),
        confidence,
        // Nothing to interpolate against, so this is a position held across the
        // reach rather than blended across it, and it is wrong by the whole
        // distance the joint travelled in that time.
        penalty: penalty(speed * reach.as_secs_f64(), confidence),
    }
}

/// How far a linear interpolation can be from the truth, in pixels.
///
/// Straight-line interpolation is exact for a joint moving at a constant speed,
/// so the error is whatever the joint did *besides* that — and two frames
/// cannot measure it. What they bound is how much it could be: a foot can
/// reverse within one frame interval, which puts the acceleration at around a
/// speed per bracket. Carrying that through gives an error that vanishes at
/// both ends of the bracket, peaks in the middle, and grows with how long the
/// camera left the fusion clock waiting.
fn interpolation_drift(speed: f64, gap: Duration, bracket: Duration) -> f64 {
    if bracket.is_zero() {
        return 0.0;
    }
    let gap = gap.as_secs_f64();
    let bracket = bracket.as_secs_f64();
    speed * gap * (bracket - gap) / bracket
}

/// How much to inflate a ray's uncertainty for not having been observed.
///
/// The drift is compared against the keypoint noise it competes with, because
/// below that it is not the thing limiting the answer. Equal to it, the ray
/// counts for a quarter of what an observed one would.
fn penalty(drift: f64, confidence: f64) -> f64 {
    1.0 + drift / pixel_sigma(confidence)
}

/// Mean pixel speed of the joints both frames have, in pixels per second.
fn body_speed(before: &PoseFrame, after: &PoseFrame, bracket: Duration) -> f64 {
    if bracket.is_zero() {
        return 0.0;
    }

    let mut sum = 0.0;
    let mut count = 0usize;
    for (joint, start) in before.keypoints.iter() {
        let Some(end) = after.keypoints.get(joint) else {
            continue;
        };
        sum += (point(end.x, end.y) - point(start.x, start.y)).norm();
        count += 1;
    }

    if count == 0 {
        0.0
    } else {
        sum / count as f64 / bracket.as_secs_f64()
    }
}

fn point(x: f32, y: f32) -> Point2<f64> {
    Point2::new(x as f64, y as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::traits::{Keypoint, Keypoints2d};

    fn frame(at: Instant, joints: &[(Joint, f32, f32, f32)]) -> PoseFrame {
        let mut keypoints = Keypoints2d::default();
        for (joint, x, y, confidence) in joints {
            keypoints.set(
                *joint,
                Keypoint {
                    x: *x,
                    y: *y,
                    confidence: *confidence,
                },
            );
        }
        PoseFrame {
            seq: 0,
            captured_at: at,
            width: 1280,
            height: 720,
            detection: None,
            keypoints,
        }
    }

    #[test]
    fn the_midpoint_of_a_bracket_is_halfway_between_its_frames() {
        let start = Instant::now();
        let before = frame(start, &[(Joint::LeftAnkle, 100.0, 200.0, 0.9)]);
        let after = frame(
            start + Duration::from_millis(40),
            &[(Joint::LeftAnkle, 140.0, 220.0, 0.9)],
        );

        let aligned = align(&before, &after, start + Duration::from_millis(20));
        let ankle = aligned.get(Joint::LeftAnkle).unwrap();

        assert!((ankle.pixel.x - 120.0).abs() < 1e-9);
        assert!((ankle.pixel.y - 210.0).abs() < 1e-9);
        assert_eq!(aligned.gap, Duration::from_millis(20));
        assert_eq!(aligned.bracket, Duration::from_millis(40));
    }

    /// A tick that lands on a real frame is not a guess and should not be
    /// charged for one.
    #[test]
    fn landing_on_a_frame_costs_nothing() {
        let start = Instant::now();
        let before = frame(start, &[(Joint::LeftAnkle, 100.0, 200.0, 0.9)]);
        let after = frame(
            start + Duration::from_millis(40),
            &[(Joint::LeftAnkle, 400.0, 200.0, 0.9)],
        );

        for tick in [start, start + Duration::from_millis(40)] {
            let aligned = align(&before, &after, tick);
            assert!((aligned.get(Joint::LeftAnkle).unwrap().penalty - 1.0).abs() < 1e-12);
        }
    }

    /// The point of the penalty: reaching across a fast movement is a worse
    /// guess than reaching across a slow one, over the same interval.
    #[test]
    fn a_fast_joint_is_trusted_less_than_a_slow_one() {
        let start = Instant::now();
        let tick = start + Duration::from_millis(20);
        let span = Duration::from_millis(40);

        let slow = align(
            &frame(start, &[(Joint::LeftAnkle, 100.0, 200.0, 0.9)]),
            &frame(start + span, &[(Joint::LeftAnkle, 102.0, 200.0, 0.9)]),
            tick,
        );
        let fast = align(
            &frame(start, &[(Joint::LeftAnkle, 100.0, 200.0, 0.9)]),
            &frame(start + span, &[(Joint::LeftAnkle, 500.0, 200.0, 0.9)]),
            tick,
        );

        let slow = slow.get(Joint::LeftAnkle).unwrap().penalty;
        let fast = fast.get(Joint::LeftAnkle).unwrap().penalty;
        assert!(slow < 1.5, "a barely moving joint should be nearly free");
        assert!(
            fast > 5.0 * slow,
            "reaching across a stride should cost more, got {fast} against {slow}"
        );
    }

    /// A camera that only manages a frame every 100 ms is guessing across a
    /// much longer reach than one running at 60, and should count for less at
    /// the same speed.
    #[test]
    fn a_slower_camera_pays_more_for_the_same_movement() {
        let start = Instant::now();
        let speed = 400.0f32; // pixels per second

        let mut penalties = Vec::new();
        for millis in [16u64, 100] {
            let span = Duration::from_millis(millis);
            let travel = speed * span.as_secs_f32();
            let aligned = align(
                &frame(start, &[(Joint::LeftAnkle, 100.0, 200.0, 0.9)]),
                &frame(
                    start + span,
                    &[(Joint::LeftAnkle, 100.0 + travel, 200.0, 0.9)],
                ),
                start + span / 2,
            );
            penalties.push(aligned.get(Joint::LeftAnkle).unwrap().penalty);
        }

        assert!(
            penalties[1] > 3.0 * penalties[0],
            "the slow camera should pay more, got {:?}",
            penalties
        );
    }

    #[test]
    fn the_weaker_end_of_an_interpolation_sets_the_confidence() {
        let start = Instant::now();
        let aligned = align(
            &frame(start, &[(Joint::LeftKnee, 100.0, 200.0, 0.95)]),
            &frame(
                start + Duration::from_millis(40),
                &[(Joint::LeftKnee, 110.0, 200.0, 0.05)],
            ),
            start + Duration::from_millis(20),
        );

        assert!((aligned.get(Joint::LeftKnee).unwrap().confidence - 0.05).abs() < 1e-6);
    }

    /// A limb the model loses hold of for one frame should still vote, using
    /// the frame that has it and paying for the whole reach.
    #[test]
    fn a_joint_only_one_frame_has_is_still_used() {
        let start = Instant::now();
        let before = frame(
            start,
            &[
                (Joint::LeftAnkle, 100.0, 200.0, 0.9),
                (Joint::LeftKnee, 100.0, 150.0, 0.9),
            ],
        );
        let after = frame(
            start + Duration::from_millis(40),
            &[(Joint::LeftAnkle, 180.0, 200.0, 0.9)],
        );

        let aligned = align(&before, &after, start + Duration::from_millis(30));
        let knee = aligned.get(Joint::LeftKnee).expect("the knee still counts");

        assert_eq!(knee.pixel.x, 100.0, "it sits where it was actually seen");
        assert!(
            knee.penalty > aligned.get(Joint::LeftAnkle).unwrap().penalty,
            "reaching 30 ms back should cost more than interpolating 10 ms"
        );
    }

    #[test]
    fn frames_with_nothing_in_common_align_to_nothing() {
        let start = Instant::now();
        let aligned = align(
            &frame(start, &[]),
            &frame(start + Duration::from_millis(40), &[]),
            start + Duration::from_millis(20),
        );
        assert!(aligned.is_empty());
    }
}
