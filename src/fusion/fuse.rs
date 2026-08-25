//! From aligned 2D keypoints to one 3D pose.
//!
//! Each joint is solved on its own from the cameras that can see it. Solving
//! them independently is deliberate: a joint hidden from two of three cameras
//! is a bad joint, not a bad pose, and letting it drag the rest of the body
//! with it is exactly the failure that makes multi-camera tracking feel worse
//! than single-camera tracking. What holds the body together is the skeleton
//! fit downstream, which knows about anatomy — this stage only knows about
//! rays.
//!
//! Every joint that comes out carries how much it can be believed, in metres.
//! That number, rather than the reprojection residual, is what tells a user
//! their cameras are badly placed: two cameras side by side agree perfectly
//! with each other about a point neither of them can locate.

use std::time::Instant;

use nalgebra::Point3;

use crate::geometry::camera::Camera;
use crate::geometry::triangulate::{Observation, Triangulation, closest_approach, triangulate};
use crate::models::{Joint, JointMap};

use super::align::Aligned;
use super::settle::Settling;

#[derive(Debug, Clone)]
pub struct FuseOptions {
    /// Confidence a keypoint needs before its ray is used at all.
    ///
    /// Below this the pose model is not reporting a joint so much as declining
    /// to, and the position it gives is wherever the joint usually is on a
    /// body rather than where this one is.
    pub min_confidence: f64,
    /// Angular disagreement past which a ray is treated as seeing something
    /// else, in radians.
    pub inlier_threshold: f64,
    /// Positional uncertainty past which a joint is not reported at all, in
    /// metres.
    ///
    /// A joint located to within a third of a metre is not a measurement, and
    /// passing it downstream would only give the filter something confident to
    /// smooth.
    pub max_sigma: f64,
    /// Consecutive solved ticks a joint owes before it is used again, having
    /// failed one of the tests above.
    ///
    /// Zero turns it off. See [`crate::fusion::settle`] for why holding a joint
    /// out is better than letting it alternate.
    pub settle_ticks: u32,
}

impl Default for FuseOptions {
    fn default() -> Self {
        Self {
            min_confidence: 0.3,
            // About half a degree. Loose enough for keypoint noise on a weak
            // detection, tight enough that a limb the model put on the wrong
            // leg does not pass for the right one.
            inlier_threshold: 0.01,
            max_sigma: 0.10,
            // A tenth of a second at sixty hertz. Long enough that a joint
            // sitting on a threshold cannot rattle through it, short enough
            // that a leg genuinely coming back into view is not noticeably
            // late. Counted in ticks rather than milliseconds because what it
            // guards against is alternation from one tick to the next.
            settle_ticks: 6,
        }
    }
}

/// Why a joint is not in the reconstruction.
///
/// Different faults with different repairs, and the panel used to show one dash
/// for all of them. "Twenty-three of twenty-six joints inferred" says something
/// is badly wrong and nothing about what: a camera that cannot see the legs, a
/// pose model that will not commit to them, a calibration that stopped the rays
/// meeting, and a geometry that cannot place them are four unrelated problems,
/// and the user's next move is different for each.
///
/// The last variant is the odd one: it is not a fault in the room but a joint
/// that keeps changing its answer, held back until it stops.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Missing {
    /// No camera reported a keypoint here at all. Move a camera, or aim it
    /// lower.
    Unseen,
    /// Cameras had a keypoint and none of them believed it enough to be used.
    /// The best confidence any of them managed is carried, because the useful
    /// question is how far below the gate it fell.
    Unsure { offered: usize, best: f64 },
    /// One usable ray, which fixes a direction and nothing else. A second
    /// camera has to be able to see this joint.
    OneRay,
    /// Rays were there and no pair of them came near enough to meeting to be
    /// worth solving from. `miss` is how near the nearest pair got, in metres,
    /// and the size of it is the whole diagnosis: a couple of centimetres is
    /// keypoint noise against a gate set too tight, twenty is two cameras
    /// looking at different legs of the same person, a metre is them looking at
    /// different people.
    ///
    /// Only reachable from three rays up. With exactly two this stage does not
    /// test agreement at all — there is nothing to outvote — so a badly
    /// calibrated pair does not fail here; it produces a point, confidently, in
    /// the wrong place, and only the uncertainty stands between a user and it.
    /// Even with a test, a pair is blind to any error along the epipolar
    /// direction: perturb one ray within the plane the two of them share and
    /// they still meet exactly, just somewhere else. That is the direction that
    /// moves the answer, so the third camera is not a refinement. It is the
    /// first one that can tell you the other two are wrong.
    Disagreed { rays: usize, miss: f64 },
    /// Solved, but to a position too uncertain to be worth reporting — the
    /// cameras that saw it are too close together, or too nearly in line with
    /// it, to say where along the ray it sits.
    Uncertain { sigma: f64 },
    /// Solved, and held back anyway because it has only just come back from one
    /// of the faults above and has not yet stayed. `ticks` is how many more
    /// consecutive solved ticks it owes.
    ///
    /// The one reason here that is not a complaint about the room. A count that
    /// stays high says joints are passing and failing their tests rather than
    /// settling on either, which is worth seeing on its own: it is the
    /// signature of thresholds being sat on, and it means the figures beside it
    /// are describing a body that keeps changing which joints it is made of.
    /// See [`crate::fusion::settle`].
    Settling { ticks: u32 },
}

impl Missing {
    pub fn label(self) -> &'static str {
        match self {
            Missing::Unseen => "unseen",
            Missing::Unsure { .. } => "unsure",
            Missing::OneRay => "one ray",
            Missing::Disagreed { .. } => "disagreed",
            Missing::Uncertain { .. } => "too uncertain",
            Missing::Settling { .. } => "settling",
        }
    }

    /// What to do about it, in one clause.
    pub fn remedy(self) -> &'static str {
        match self {
            Missing::Unseen => "no camera has it in shot",
            Missing::Unsure { .. } => "below the confidence gate",
            Missing::OneRay => "only one camera can see it",
            Missing::Disagreed { .. } => "the cameras disagree about where it is",
            Missing::Uncertain { .. } => "the cameras that see it cannot place it",
            Missing::Settling { .. } => "back, but not yet steady enough to use",
        }
    }
}

/// How the body divided up between measured and the reasons it was not.
///
/// The counts are what belongs on the panel: one joint's reason is a curiosity,
/// and the same reason for fifteen joints is the fault.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Tally {
    pub measured: usize,
    /// Median distance the nearest pair of rays missed by, over the joints that
    /// could not be solved at all, in metres. The size of it is the diagnosis.
    pub miss: f64,
    pub unseen: usize,
    pub unsure: usize,
    pub one_ray: usize,
    pub disagreed: usize,
    pub uncertain: usize,
    /// Solved this tick but held back as not yet steady. Unlike the rest this
    /// is a count of joints changing their minds rather than of a fault.
    pub settling: usize,
}

impl Tally {
    pub fn missing(&self) -> usize {
        self.unseen + self.unsure + self.one_ray + self.disagreed + self.uncertain + self.settling
    }

    /// The reason accounting for the most joints, when there are enough of them
    /// to be worth naming.
    pub fn commonest(&self) -> Option<(String, usize)> {
        [
            ("no camera has them in shot".to_owned(), self.unseen),
            (
                "their keypoints fall below the confidence gate".to_owned(),
                self.unsure,
            ),
            ("only one camera can see them".to_owned(), self.one_ray),
            (
                // The distance is the diagnosis, so it is in the sentence.
                format!(
                    "no two cameras agree where they are, missing each other by {:.0} cm \
                     at the nearest",
                    self.miss * 100.0
                ),
                self.disagreed,
            ),
            (
                "the cameras that see them cannot place them".to_owned(),
                self.uncertain,
            ),
            (
                "they keep passing and failing the same test".to_owned(),
                self.settling,
            ),
        ]
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .filter(|(_, count)| *count > 0)
    }
}

/// One joint as the cameras reconstructed it.
#[derive(Debug, Clone)]
pub struct FusedJoint {
    pub point: Point3<f64>,
    /// Standard deviation along the least constrained direction, in metres.
    pub sigma: f64,
    /// RMS angular reprojection residual over the rays used, in radians.
    pub residual: f64,
    /// Share of the answer each camera carried, summing to one.
    pub weights: Vec<(usize, f64)>,
    /// Cameras that saw the joint and were dropped for disagreeing.
    pub rejected: Vec<usize>,
}

impl FusedJoint {
    pub fn rays(&self) -> usize {
        self.weights.len()
    }

    /// The residual in degrees, which is the form worth showing.
    pub fn residual_degrees(&self) -> f64 {
        self.residual.to_degrees()
    }
}

/// A whole body at one instant of the fusion clock.
#[derive(Debug, Clone)]
pub struct Pose3d {
    /// The instant this reconstructs, which is behind real time by the
    /// alignment lag.
    pub at: Instant,
    /// How much worse the cameras agreed than their keypoints claimed they
    /// would, as a multiplier already applied to every joint.s uncertainty.
    ///
    /// One means the rays landed exactly as accurately as the pose models said
    /// they would. Anything much above it is the calibration, and it is the
    /// single most useful number about a room: it is measured continuously,
    /// from the user rather than from a checkerboard, and it is what says
    /// whether the cameras have been knocked since they were solved.
    pub disagreement: f64,
    joints: JointMap<FusedJoint>,
    /// Why each joint that is not here is not here.
    missing: JointMap<Missing>,
}

impl Pose3d {
    pub fn empty(at: Instant) -> Self {
        Self {
            at,
            disagreement: 1.0,
            joints: JointMap::default(),
            missing: JointMap::default(),
        }
    }

    pub fn get(&self, joint: Joint) -> Option<&FusedJoint> {
        self.joints.get(joint)
    }

    pub fn set(&mut self, joint: Joint, fused: FusedJoint) {
        self.joints.set(joint, fused);
        self.missing.clear(joint);
    }

    pub fn missing(&self, joint: Joint) -> Option<Missing> {
        self.missing.copied(joint)
    }

    fn miss(&mut self, joint: Joint, why: Missing) {
        self.missing.set(joint, why);
    }

    /// Takes back a joint that was set, giving a reason.
    ///
    /// Separate from [`Pose3d::miss`] because it has to undo the `set`, and
    /// because the only caller is the admission pass at the end of the fuse:
    /// nothing else in this stage decides against a joint it has already
    /// solved.
    fn hold_back(&mut self, joint: Joint, why: Missing) {
        self.joints.clear(joint);
        self.missing.set(joint, why);
    }

    /// How the body divided up between measured and the reasons it was not.
    pub fn tally(&self) -> Tally {
        let mut tally = Tally {
            measured: self.count(),
            ..Tally::default()
        };
        for why in self.missing.values() {
            let slot = match why {
                Missing::Unseen => &mut tally.unseen,
                Missing::Unsure { .. } => &mut tally.unsure,
                Missing::OneRay => &mut tally.one_ray,
                Missing::Disagreed { .. } => &mut tally.disagreed,
                Missing::Uncertain { .. } => &mut tally.uncertain,
                Missing::Settling { .. } => &mut tally.settling,
            };
            *slot += 1;
        }

        // Median rather than mean: one joint whose rays are a room apart is a
        // keypoint on somebody else, and it should not become the figure that
        // describes the other ten.
        let mut misses: Vec<f64> = self
            .missing
            .values()
            .filter_map(|why| match why {
                Missing::Disagreed { miss, .. } if miss.is_finite() => Some(*miss),
                _ => None,
            })
            .collect();
        if !misses.is_empty() {
            misses.sort_by(f64::total_cmp);
            tally.miss = misses[misses.len() / 2];
        }

        tally
    }

    pub fn iter(&self) -> impl Iterator<Item = (Joint, &FusedJoint)> + '_ {
        self.joints.iter()
    }

    pub fn count(&self) -> usize {
        self.joints.count()
    }

    pub fn is_empty(&self) -> bool {
        self.joints.is_empty()
    }

    /// How many of the lower-body joints came through, which is the number that
    /// decides whether the trackers can be built at all.
    pub fn lower_body(&self) -> usize {
        self.iter()
            .filter(|(joint, _)| joint.is_lower_body())
            .count()
    }
}

/// Reconstructs one pose from what each camera saw at the tick.
///
/// `views` pairs an index into `cameras` with that camera's keypoints already
/// resampled onto `at`. A camera with no bracketing frames belongs nowhere in
/// this list; it has nothing to say about this instant.
pub fn fuse(
    cameras: &[Camera],
    views: &[(usize, Aligned)],
    at: Instant,
    options: &FuseOptions,
    settling: &mut Settling,
) -> Pose3d {
    let mut pose = Pose3d::empty(at);
    let mut solved = Vec::with_capacity(Joint::ALL.len());

    for joint in Joint::ALL {
        // Counted before the gate as well as after it, because "no camera can
        // see this" and "no camera is sure enough about it" are different
        // faults and the user fixes them in different rooms.
        let mut offered = 0usize;
        let mut best = 0.0f64;
        let observations: Vec<Observation> = views
            .iter()
            .filter_map(|(index, aligned)| {
                let seen = aligned.get(joint)?;
                offered += 1;
                best = best.max(seen.confidence);
                if seen.confidence < options.min_confidence {
                    return None;
                }
                let camera = cameras.get(*index)?;
                Some(Observation::new(
                    *index,
                    camera,
                    seen.pixel,
                    seen.confidence,
                    seen.penalty,
                ))
            })
            .collect();

        // One ray fixes a direction and nothing else. Leaving the joint absent
        // is what lets the filter coast through the gap rather than following
        // a point that was never measured.
        if observations.len() < 2 {
            pose.miss(
                joint,
                match (offered, observations.len()) {
                    (0, _) => Missing::Unseen,
                    (_, 0) => Missing::Unsure { offered, best },
                    _ => Missing::OneRay,
                },
            );
            continue;
        }

        let Some(triangulated) = triangulate(cameras, &observations, options.inlier_threshold)
        else {
            pose.miss(
                joint,
                Missing::Disagreed {
                    rays: observations.len(),
                    miss: closest_approach(cameras, &observations).unwrap_or(f64::NAN),
                },
            );
            continue;
        };
        solved.push((joint, observations, triangulated));
    }

    // Every joint's uncertainty is scaled by this, so it has to be known before
    // any of them can be reported or thrown away.
    pose.disagreement = disagreement(solved.iter().map(|(_, _, t)| t));

    for (joint, observations, triangulated) in solved {
        let sigma = triangulated.sigma() * pose.disagreement;
        if !sigma.is_finite() || sigma > options.max_sigma {
            pose.miss(joint, Missing::Uncertain { sigma });
            continue;
        }

        let rejected = observations
            .iter()
            .map(|observation| observation.camera)
            .filter(|camera| !triangulated.inliers.contains(camera))
            .collect();

        pose.set(
            joint,
            FusedJoint {
                point: triangulated.point,
                sigma,
                residual: triangulated.rms_residual(),
                weights: triangulated.weights,
                rejected,
            },
        );
    }

    // Every test above is a threshold on a quantity that moves between ticks,
    // so a joint sitting on one of them passes and fails alternately rather
    // than settling. Run over the whole body rather than only the joints that
    // were set: a tick a joint missed is what starts it owing a dwell, and the
    // reason it missed does not matter to that.
    for joint in Joint::ALL {
        let solved = pose.get(joint).is_some();
        let owed = settling.wait(joint, solved, options.settle_ticks);
        if solved && owed > 0 {
            pose.hold_back(joint, Missing::Settling { ticks: owed });
        }
    }

    pose
}

/// How much worse the cameras agreed than their keypoints claimed they would,
/// as a multiplier on every uncertainty.
///
/// The covariance a triangulation reports is built entirely from the noise each
/// ray *claimed*, by way of the keypoint confidence the pose model attached to
/// it. It is a prediction of the error, never a measurement of one, and it is
/// wrong in a specific and damaging direction: three well spread rays pin a
/// point down beautifully whether or not the cameras they came from agree about
/// where anything is, so a room calibrated to three centimetres reports joints
/// good to five millimetres. Everything downstream believes it. The filter
/// weights a measurement by that number, so it follows the disagreement as fast
/// as it can; the panels print it, so the user is told their cameras are
/// excellent; and the limits that exist to withhold a joint nothing can place
/// never fire, because nothing ever exceeds them.
///
/// The residuals are the missing measurement. Scaling the covariance by the
/// ratio of the two is the standard a posteriori variance factor of a
/// least-squares adjustment, and it makes the reported uncertainty the one that
/// was observed rather than the one that was assumed.
///
/// Pooled over the whole body rather than computed per joint, because per joint
/// there is one degree of freedom with two rays and three with three, and the
/// variance of such an estimate is as large as the estimate. It would swing the
/// filter's gain around at random — a second source of shaking, introduced by
/// the fix for the first. Pooling is also the truer model: how far apart two
/// cameras think the room is is a property of the room, not of a knee.
fn disagreement<'a>(solved: impl Iterator<Item = &'a Triangulation>) -> f64 {
    /// A joint whose rays disagree by more than five times what they claimed
    /// is not evidence about the room. It is a keypoint on the wrong limb that
    /// the inlier test let through, and averaging it in would withhold the
    /// whole body on account of one bad ankle.
    const OUTLIER: f64 = 25.0;

    let mut chi_square = 0.0;
    let mut dof = 0.0;
    for triangulated in solved {
        if triangulated.dof <= 0.0 || triangulated.chi_square > OUTLIER * triangulated.dof {
            continue;
        }
        chi_square += triangulated.chi_square;
        dof += triangulated.dof;
    }

    if dof < 1.0 {
        return 1.0;
    }
    // Never below one. Rays that agree better than they claimed have been
    // lucky, not accurate, and a handful of joints is far too small a sample to
    // conclude otherwise from.
    (chi_square / dof).max(1.0).sqrt()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use nalgebra::Vector3;

    use super::*;
    use crate::geometry::camera::Intrinsics;
    use crate::geometry::lens::Lens;
    use crate::infer::traits::{Keypoint, Keypoints2d};
    use crate::pipeline::PoseFrame;

    fn corner(x: f64, z: f64) -> Camera {
        Camera::look_at(
            Intrinsics::from_fov(1280, 720, 70f64.to_radians()),
            Lens::default(),
            Point3::new(x, 2.4, z),
            Point3::new(0.0, 1.0, 0.0),
            Vector3::y(),
        )
    }

    fn room() -> Vec<Camera> {
        vec![corner(-1.8, -1.8), corner(1.8, -1.8), corner(1.8, 1.8)]
    }

    /// A few joints in positions a standing body would have them.
    fn body() -> Vec<(Joint, Point3<f64>)> {
        vec![
            (Joint::Hip, Point3::new(0.0, 0.95, 0.0)),
            (Joint::LeftKnee, Point3::new(-0.12, 0.50, 0.02)),
            (Joint::RightKnee, Point3::new(0.12, 0.50, -0.02)),
            (Joint::LeftAnkle, Point3::new(-0.13, 0.08, 0.0)),
            (Joint::RightAnkle, Point3::new(0.13, 0.08, 0.0)),
        ]
    }

    /// Projects the body into one camera and wraps it as an aligned view, with
    /// no interpolation penalty.
    fn view(cameras: &[Camera], index: usize, confidence: f32) -> (usize, Aligned) {
        let mut keypoints = Keypoints2d::default();
        for (joint, point) in body() {
            if let Some(pixel) = cameras[index].project(point) {
                keypoints.set(
                    joint,
                    Keypoint {
                        x: pixel.x as f32,
                        y: pixel.y as f32,
                        confidence,
                    },
                );
            }
        }
        (index, still(keypoints))
    }

    /// Wraps keypoints as an aligned view of a body that is not moving, so the
    /// interpolation costs nothing and the test measures only the geometry.
    fn still(keypoints: Keypoints2d) -> Aligned {
        let at = Instant::now();
        let frame = |captured_at| PoseFrame {
            seq: 0,
            captured_at,
            width: 1280,
            height: 720,
            detection: None,
            keypoints: keypoints.clone(),
        };
        super::super::align::align(
            &frame(at),
            &frame(at + Duration::from_millis(20)),
            at + Duration::from_millis(10),
        )
    }

    /// The same body, with one camera's keypoints landing a few pixels from
    /// where its calibration says they should — which is what a room that has
    /// drifted, or a camera that has been nudged, looks like from in here.
    fn nudged(cameras: &[Camera], index: usize, confidence: f32, pixels: f32) -> (usize, Aligned) {
        let mut keypoints = Keypoints2d::default();
        for (joint, point) in body() {
            if let Some(pixel) = cameras[index].project(point) {
                keypoints.set(
                    joint,
                    Keypoint {
                        x: pixel.x as f32 + pixels,
                        y: pixel.y as f32,
                        confidence,
                    },
                );
            }
        }
        (index, still(keypoints))
    }

    /// The uncertainty a joint is reported with has to be the one that was
    /// observed, not the one the pose models promised. Three well spread rays
    /// pin a point down beautifully whether or not the cameras they came from
    /// agree about where anything is, so the covariance alone will call a room
    /// calibrated to three centimetres accurate to five millimetres — and
    /// everything downstream believes it.
    #[test]
    fn cameras_that_disagree_report_uncertainty_that_says_so() {
        let cameras = room();

        let honest: Vec<_> = (0..3).map(|index| view(&cameras, index, 0.9)).collect();
        let agreeing = fuse(
            &cameras,
            &honest,
            Instant::now(),
            &FuseOptions::default(),
            &mut Settling::default(),
        );
        assert_eq!(
            agreeing.disagreement, 1.0,
            "rays that land exactly where they should claim nothing extra"
        );

        let mut skewed = honest.clone();
        skewed[2] = nudged(&cameras, 2, 0.9, 8.0);
        let disagreeing = fuse(
            &cameras,
            &skewed,
            Instant::now(),
            &FuseOptions::default(),
            &mut Settling::default(),
        );

        assert!(
            disagreeing.disagreement > 1.8,
            "one camera eight pixels out was reported as {:.2}x disagreement",
            disagreeing.disagreement
        );

        let calm = agreeing.get(Joint::Hip).expect("the hip was visible");
        let shaken = disagreeing
            .get(Joint::Hip)
            .expect("the hip was still visible");
        assert!(
            shaken.sigma > calm.sigma * 1.8,
            "the hip claimed {:.1} mm against {:.1} mm, which is not enough of a difference",
            shaken.sigma * 1000.0,
            calm.sigma * 1000.0
        );
    }

    /// One dash used to cover five unrelated faults, and the user's next move
    /// is different for each of them.
    #[test]
    fn a_joint_that_is_not_reconstructed_says_which_fault_it_is() {
        let cameras = room();
        let options = FuseOptions::default();

        // Nobody offers a keypoint for the wrists, because `body` has none.
        let seen: Vec<_> = (0..3).map(|index| view(&cameras, index, 0.9)).collect();
        let pose = fuse(
            &cameras,
            &seen,
            Instant::now(),
            &options,
            &mut Settling::default(),
        );
        assert_eq!(pose.missing(Joint::LeftWrist), Some(Missing::Unseen));

        // Every camera offers one, and none of them believes it.
        let timid: Vec<_> = (0..3).map(|index| view(&cameras, index, 0.1)).collect();
        let pose = fuse(
            &cameras,
            &timid,
            Instant::now(),
            &options,
            &mut Settling::default(),
        );
        assert!(
            matches!(
                pose.missing(Joint::Hip),
                Some(Missing::Unsure { offered: 3, best }) if (best - 0.1).abs() < 1e-6
            ),
            "expected three unsure offers, got {:?}",
            pose.missing(Joint::Hip)
        );

        // Only one camera can see it, which fixes a direction and nothing else.
        let pose = fuse(
            &cameras,
            &seen[..1],
            Instant::now(),
            &options,
            &mut Settling::default(),
        );
        assert_eq!(pose.missing(Joint::Hip), Some(Missing::OneRay));

        // Two cameras a hand.s width apart: they agree with each other about a
        // point neither of them can place along the line of sight, so it is
        // solved and then thrown away for being worthless.
        let grazing = vec![corner(-1.8, -1.8), corner(-1.7, -1.8)];
        let pair: Vec<_> = (0..2).map(|index| view(&grazing, index, 0.9)).collect();
        let pose = fuse(
            &grazing,
            &pair,
            Instant::now(),
            &options,
            &mut Settling::default(),
        );
        assert!(
            matches!(pose.missing(Joint::Hip), Some(Missing::Uncertain { .. })),
            "expected it to be too uncertain, got {:?}",
            pose.missing(Joint::Hip)
        );

        // And the counts are what goes on the panel.
        let pose = fuse(
            &cameras,
            &seen,
            Instant::now(),
            &options,
            &mut Settling::default(),
        );
        let tally = pose.tally();
        assert_eq!(tally.measured, body().len());
        assert_eq!(
            tally.measured + tally.missing(),
            Joint::ALL.len(),
            "every joint has to be accounted for exactly once"
        );
    }

    /// The point itself must not move. This changes what is claimed about the
    /// answer, not the answer.
    #[test]
    fn measuring_the_disagreement_does_not_move_the_body() {
        let cameras = room();
        let views: Vec<_> = (0..3).map(|index| view(&cameras, index, 0.9)).collect();
        let pose = fuse(
            &cameras,
            &views,
            Instant::now(),
            &FuseOptions::default(),
            &mut Settling::default(),
        );

        for (joint, truth) in body() {
            let fused = pose.get(joint).expect("every joint was visible");
            assert!((fused.point - truth).norm() < 1e-6, "{joint:?} moved");
        }
    }
    #[test]
    fn a_body_seen_by_three_cameras_comes_back_where_it_was() {
        let cameras = room();
        let views: Vec<_> = (0..3).map(|index| view(&cameras, index, 0.9)).collect();
        let pose = fuse(
            &cameras,
            &views,
            Instant::now(),
            &FuseOptions::default(),
            &mut Settling::default(),
        );

        assert_eq!(pose.count(), body().len());
        for (joint, truth) in body() {
            let fused = pose.get(joint).expect("every joint was visible");
            assert!(
                (fused.point - truth).norm() < 1e-6,
                "{joint:?} came back at {:?}, expected {truth:?}",
                fused.point
            );
            assert_eq!(fused.rays(), 3);
            assert!(fused.sigma < 0.01, "{joint:?} sigma {}", fused.sigma);
        }
    }

    #[test]
    fn one_camera_alone_reconstructs_nothing() {
        let cameras = room();
        let views = vec![view(&cameras, 0, 0.9)];
        let pose = fuse(
            &cameras,
            &views,
            Instant::now(),
            &FuseOptions::default(),
            &mut Settling::default(),
        );
        assert!(pose.is_empty());
    }

    /// A joint the model is barely reporting is not a measurement, and voting
    /// with it is worse than leaving the joint to the filter.
    #[test]
    fn keypoints_below_the_confidence_gate_do_not_vote() {
        let cameras = room();
        let views: Vec<_> = (0..3).map(|index| view(&cameras, index, 0.1)).collect();
        let pose = fuse(
            &cameras,
            &views,
            Instant::now(),
            &FuseOptions::default(),
            &mut Settling::default(),
        );
        assert!(pose.is_empty());
    }

    /// An occluded joint still gets a confident keypoint, just in the wrong
    /// place. It should be dropped and named, so the UI can say which camera.
    #[test]
    fn a_camera_that_disagrees_is_dropped_and_reported() {
        let cameras = room();
        let mut views: Vec<_> = (0..3).map(|index| view(&cameras, index, 0.9)).collect();

        // Move one camera's ankle a long way, as an occluding leg would.
        let mut broken = Keypoints2d::default();
        for (joint, point) in body() {
            if let Some(pixel) = cameras[2].project(point) {
                let shift = if joint == Joint::LeftAnkle {
                    120.0
                } else {
                    0.0
                };
                broken.set(
                    joint,
                    Keypoint {
                        x: pixel.x as f32 + shift,
                        y: pixel.y as f32,
                        confidence: 0.9,
                    },
                );
            }
        }
        views[2] = (2, still(broken));

        let pose = fuse(
            &cameras,
            &views,
            Instant::now(),
            &FuseOptions::default(),
            &mut Settling::default(),
        );
        let ankle = pose.get(Joint::LeftAnkle).expect("two cameras still agree");

        assert_eq!(ankle.rejected, vec![2]);
        assert_eq!(ankle.rays(), 2);
        assert!(
            (ankle.point - Point3::new(-0.13, 0.08, 0.0)).norm() < 1e-4,
            "the outlier moved the ankle to {:?}",
            ankle.point
        );

        // And the joints that camera got right are still worth three rays.
        assert_eq!(pose.get(Joint::Hip).unwrap().rays(), 3);
    }

    /// Cameras that cannot pin a joint down should have it withheld rather than
    /// reported with a number nobody can act on.
    #[test]
    fn a_joint_nothing_can_locate_is_withheld() {
        // Two cameras a hand's width apart: they agree with each other about a
        // point neither of them can place along the line of sight.
        let cameras = vec![corner(-1.8, -1.8), corner(-1.7, -1.8)];
        let views: Vec<_> = (0..2).map(|index| view(&cameras, index, 0.9)).collect();

        let pose = fuse(
            &cameras,
            &views,
            Instant::now(),
            &FuseOptions::default(),
            &mut Settling::default(),
        );
        assert!(
            pose.is_empty(),
            "a grazing pair should not be reported as a measurement"
        );
    }

    #[test]
    fn the_weights_say_which_camera_carried_the_joint() {
        let cameras = room();
        let views: Vec<_> = (0..3).map(|index| view(&cameras, index, 0.9)).collect();
        let pose = fuse(
            &cameras,
            &views,
            Instant::now(),
            &FuseOptions::default(),
            &mut Settling::default(),
        );

        let hip = pose.get(Joint::Hip).unwrap();
        let total: f64 = hip.weights.iter().map(|(_, weight)| weight).sum();
        assert!((total - 1.0).abs() < 1e-9);
        assert!(hip.weights.iter().all(|(_, weight)| *weight > 0.0));
    }

    /// A joint the cameras can only solve on alternate ticks used to alternate
    /// in the output, and the two answers are far apart: solved, it is where
    /// the cameras put it; unsolved, the fit invents it from the skeleton. The
    /// gap between those is the calibration error, so the joint does not
    /// degrade when it flips, it jumps.
    #[test]
    fn a_joint_that_can_only_be_solved_every_other_tick_is_held_out() {
        let cameras = room();
        let options = FuseOptions::default();
        let mut settling = Settling::default();

        let confident: Vec<_> = (0..3).map(|index| view(&cameras, index, 0.9)).collect();
        let timid: Vec<_> = (0..3).map(|index| view(&cameras, index, 0.1)).collect();

        let mut measured = 0;
        let mut settled_out = 0;
        for tick in 0..60 {
            let views = if tick % 2 == 0 { &confident } else { &timid };
            let pose = fuse(&cameras, views, Instant::now(), &options, &mut settling);

            if pose.get(Joint::Hip).is_some() {
                measured += 1;
            }
            if matches!(pose.missing(Joint::Hip), Some(Missing::Settling { .. })) {
                settled_out += 1;
            }
        }

        // Once, on the first tick, before anything had gone wrong. After that
        // it can never gather a run of solved ticks.
        assert_eq!(
            measured, 1,
            "an alternating joint reached the output {measured} times in sixty ticks"
        );
        assert_eq!(
            settled_out, 29,
            "the ticks it was solved but held back should be the other twenty-nine"
        );
    }

    /// And the same joint, solved every tick, is not held back at all: the
    /// dwell is there to stop a joint returning too eagerly, not to tax one
    /// that never left.
    #[test]
    fn a_joint_solved_every_tick_is_never_held_back() {
        let cameras = room();
        let options = FuseOptions::default();
        let mut settling = Settling::default();
        let views: Vec<_> = (0..3).map(|index| view(&cameras, index, 0.9)).collect();

        for tick in 0..60 {
            let pose = fuse(&cameras, &views, Instant::now(), &options, &mut settling);
            assert!(
                pose.get(Joint::Hip).is_some(),
                "the hip went missing on tick {tick} with nothing wrong"
            );
        }
    }
}
