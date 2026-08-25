//! Holding the reconstruction to something a body could do.
//!
//! Triangulation solves each joint on its own, which is what keeps one occluded
//! limb from spoiling the rest — and it means nothing stops the result being
//! anatomically impossible. A knee wanders four centimetres from its hip
//! between frames, a foot sinks through the floor, a leg bends backwards. None
//! of that is motion; all of it is keypoint noise, and roughly half of the
//! noise is in directions the body physically cannot go.
//!
//! So the fit is not smoothing. Smoothing trades lag for jitter and treats
//! every direction alike. This throws away only the component of the error that
//! is impossible, which costs no lag at all, and it is what lets a joint nobody
//! can currently see be placed from the ones who can rather than left to drift.
//!
//! The solver is sequential projection: each constraint is satisfied in turn,
//! repeatedly, with each joint moving in inverse proportion to how well it was
//! measured. A joint located to five millimetres barely moves; one located to
//! five centimetres gives way; one nobody saw goes wherever the constraints
//! put it. That weighting is what makes the difference between a fit that
//! cleans up the uncertain joints and one that drags the certain ones around.

use std::time::Instant;

use nalgebra::{Point3, Vector3};

use crate::models::{Joint, JointMap};

use super::bones::{BONES, Bone, MeasureOptions, Skeleton};
use super::fuse::Pose3d;

#[derive(Debug, Clone)]
pub struct FitOptions {
    /// Passes over the constraint set. Each one satisfies every constraint in
    /// turn, and later passes repair what earlier ones disturbed.
    pub passes: usize,
    /// Height of the floor in the world frame, which the calibration puts at
    /// zero.
    pub floor: f64,
    /// Uncertainty given to a joint nothing saw, in metres. Large enough that
    /// it yields to every joint that was actually measured.
    pub free_sigma: f64,
    /// Smallest uncertainty a joint is credited with, in metres.
    ///
    /// Without a floor here, one implausibly confident joint would become
    /// immovable and every correction would land on its neighbours.
    pub min_sigma: f64,
    /// Whether to keep knees from bending backwards.
    pub limit_knees: bool,
    /// Rules for reading the measured skeleton.
    pub measure: MeasureOptions,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            passes: 12,
            floor: 0.0,
            free_sigma: 0.5,
            min_sigma: 0.003,
            limit_knees: true,
            measure: MeasureOptions::default(),
        }
    }
}

/// One joint after the fit.
#[derive(Debug, Clone, Copy)]
pub struct FittedJoint {
    pub point: Point3<f64>,
    /// The uncertainty it came in with, in metres.
    pub sigma: f64,
    /// True when no camera saw it and the constraints placed it.
    pub inferred: bool,
    /// How far the fit moved it from where the cameras put it, in metres. Zero
    /// for an inferred joint, which had nowhere to be moved from.
    pub correction: f64,
}

/// A whole body after the fit.
#[derive(Debug, Clone)]
pub struct Fitted {
    pub at: Instant,
    joints: JointMap<FittedJoint>,
}

impl Fitted {
    pub fn empty(at: Instant) -> Self {
        Self {
            at,
            joints: JointMap::default(),
        }
    }

    pub fn get(&self, joint: Joint) -> Option<FittedJoint> {
        self.joints.copied(joint)
    }

    pub fn position(&self, joint: Joint) -> Option<Point3<f64>> {
        self.get(joint).map(|fitted| fitted.point)
    }

    pub fn set(&mut self, joint: Joint, fitted: FittedJoint) {
        self.joints.set(joint, fitted);
    }

    pub fn iter(&self) -> impl Iterator<Item = (Joint, FittedJoint)> + '_ {
        self.joints.iter().map(|(joint, fitted)| (joint, *fitted))
    }

    pub fn count(&self) -> usize {
        self.joints.count()
    }

    pub fn is_empty(&self) -> bool {
        self.joints.is_empty()
    }

    /// Joints that were placed by the constraints rather than seen.
    pub fn inferred(&self) -> usize {
        self.iter().filter(|(_, fitted)| fitted.inferred).count()
    }

    /// Largest distance the fit had to move an observed joint, in metres. A
    /// large value means the measured skeleton and the cameras disagree.
    pub fn worst_correction(&self) -> f64 {
        self.iter()
            .map(|(_, fitted)| fitted.correction)
            .fold(0.0, f64::max)
    }
}

/// Runs the fit, carrying the previous result forward so that a joint which
/// goes out of sight has somewhere to start from.
#[derive(Debug, Clone, Default)]
pub struct Fitter {
    options: FitOptions,
    previous: Option<Fitted>,
}

/// A joint while it is being solved.
#[derive(Debug, Clone, Copy)]
struct Node {
    point: Point3<f64>,
    /// Where the cameras put it, if they did.
    observed: Option<Point3<f64>>,
    sigma: f64,
    /// How freely this joint moves under a correction. Proportional to the
    /// square of its uncertainty, so the answer settles where the measurements
    /// are strongest.
    give: f64,
}

impl Fitter {
    pub fn new(options: FitOptions) -> Self {
        Self {
            options,
            previous: None,
        }
    }

    pub fn options(&self) -> &FitOptions {
        &self.options
    }

    pub fn set_options(&mut self, options: FitOptions) {
        self.options = options;
    }

    /// Forgets the previous pose, so nothing carries across a break in
    /// tracking.
    pub fn reset(&mut self) {
        self.previous = None;
    }

    pub fn fit(&mut self, pose: &Pose3d, skeleton: &Skeleton) -> Fitted {
        let mut nodes = self.seed(pose);

        for _ in 0..self.options.passes {
            for bone in BONES {
                self.enforce_length(&mut nodes, *bone, skeleton);
            }
            if self.options.limit_knees {
                for side in [
                    (Joint::LeftHip, Joint::LeftKnee, Joint::LeftAnkle),
                    (Joint::RightHip, Joint::RightKnee, Joint::RightAnkle),
                ] {
                    self.limit_knee(&mut nodes, side);
                }
            }
            // Last, so the floor is the one constraint that comes out exactly
            // satisfied. A foot a centimetre underground is the single most
            // visible failure in VR.
            self.hold_above_floor(&mut nodes);
        }

        let mut fitted = Fitted::empty(pose.at);
        for joint in Joint::ALL {
            let Some(node) = nodes[joint.index()] else {
                continue;
            };

            // How far the fit had to drag the joint from where the cameras put
            // it. A large value means the measured skeleton and the cameras
            // disagree, which is worth surfacing rather than silently
            // resolving — but the joint is still reported as fitted, because
            // reverting one joint and not its neighbours would hand downstream
            // a body that satisfies nothing.
            let correction = node
                .observed
                .map(|observed| (node.point - observed).norm())
                .unwrap_or(0.0);

            fitted.set(
                joint,
                FittedJoint {
                    point: node.point,
                    sigma: node.sigma,
                    inferred: node.observed.is_none(),
                    correction,
                },
            );
        }

        self.previous = Some(fitted.clone());
        fitted
    }

    /// Starting positions for the solve.
    ///
    /// A joint nothing saw starts from where it was last time, which is what
    /// makes it continuous rather than a jump; the constraints then move it to
    /// where the joints around it say it must be. A joint that was never seen
    /// and has no history stays absent — there is no honest place to put it.
    fn seed(&self, pose: &Pose3d) -> Vec<Option<Node>> {
        let mut nodes: Vec<Option<Node>> = (0..Joint::ALL.len()).map(|_| None).collect();

        for joint in Joint::ALL {
            nodes[joint.index()] = match pose.get(joint) {
                Some(fused) => {
                    let sigma = fused.sigma.max(self.options.min_sigma);
                    Some(Node {
                        point: fused.point,
                        observed: Some(fused.point),
                        sigma,
                        give: sigma * sigma,
                    })
                }
                None => self
                    .previous
                    .as_ref()
                    .and_then(|previous| previous.get(joint))
                    .map(|carried| Node {
                        point: carried.point,
                        observed: None,
                        sigma: self.options.free_sigma,
                        give: self.options.free_sigma * self.options.free_sigma,
                    }),
            };
        }

        nodes
    }

    /// Pulls two joints to the measured distance apart.
    fn enforce_length(&self, nodes: &mut [Option<Node>], bone: Bone, skeleton: &Skeleton) {
        let Some(measured) = skeleton.get(bone) else {
            return;
        };
        if !measured.is_settled(&self.options.measure) {
            return;
        }
        let (Some(from), Some(to)) = (nodes[bone.from.index()], nodes[bone.to.index()]) else {
            return;
        };

        let along = to.point - from.point;
        let distance = along.norm();
        if distance < 1e-9 {
            return;
        }

        let total = from.give + to.give;
        if total < 1e-18 {
            return;
        }

        let correction = along * ((distance - measured.length) / distance);
        nodes[bone.from.index()].as_mut().unwrap().point += correction * (from.give / total);
        nodes[bone.to.index()].as_mut().unwrap().point -= correction * (to.give / total);
    }

    /// Lifts anything that has sunk through the floor.
    fn hold_above_floor(&self, nodes: &mut [Option<Node>]) {
        for node in nodes.iter_mut().flatten() {
            if node.point.y < self.options.floor {
                node.point.y = self.options.floor;
            }
        }
    }

    /// Keeps a knee from ending up behind the line between its hip and ankle.
    ///
    /// A leg cannot bend that way, and it is a failure the cameras produce
    /// readily: from behind, a knee and the space in front of it look much
    /// alike, and a pose model asked which way the leg folds will sometimes
    /// answer wrongly. Left alone the result is a leg that snaps between two
    /// mirror-image bends as the user turns.
    fn limit_knee(&self, nodes: &mut [Option<Node>], leg: (Joint, Joint, Joint)) {
        let (hip_joint, knee_joint, ankle_joint) = leg;
        let (Some(hip), Some(knee), Some(ankle)) = (
            nodes[hip_joint.index()],
            nodes[knee_joint.index()],
            nodes[ankle_joint.index()],
        ) else {
            return;
        };
        let Some(forward) = self.facing(nodes) else {
            return;
        };

        // How far the knee sits from the hip-to-ankle line, and which way.
        let axis = ankle.point - hip.point;
        let length = axis.norm();
        if length < 1e-6 {
            return;
        }
        let axis = axis / length;
        let offset = knee.point - hip.point;
        let sideways = offset - axis * offset.dot(&axis);

        let bend = sideways.dot(&forward);
        if bend >= 0.0 {
            return;
        }

        // Reflecting the knee across the line, rather than flattening it onto
        // it, keeps the leg bent by as much as it was — the amount of bend was
        // never in doubt, only its direction.
        nodes[knee_joint.index()].as_mut().unwrap().point -= forward * (2.0 * bend);
    }

    /// Which way the body is facing, from the line across its hips.
    ///
    /// The world frame is right-handed with +Y up, so up crossed with the
    /// body's right-hand direction points out of its front.
    fn facing(&self, nodes: &[Option<Node>]) -> Option<Vector3<f64>> {
        let left = nodes[Joint::LeftHip.index()]?.point;
        let right = nodes[Joint::RightHip.index()]?.point;

        let across = right - left;
        let forward = Vector3::y().cross(&across);
        let length = forward.norm();
        // Hips seen edge-on say nothing about which way the body faces, and a
        // guess here would flip a leg rather than leave it alone.
        (length > 1e-3).then(|| forward / length)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion::bones::{BoneLength, BoneMeter};
    use crate::fusion::fuse::FusedJoint;

    fn joint(point: Point3<f64>, sigma: f64) -> FusedJoint {
        FusedJoint {
            point,
            sigma,
            residual: 0.0,
            weights: vec![(0, 0.5), (1, 0.5)],
            rejected: Vec::new(),
        }
    }

    /// A standing body with a known anatomy.
    fn standing() -> Vec<(Joint, Point3<f64>)> {
        vec![
            (Joint::Hip, Point3::new(0.0, 0.95, 0.0)),
            (Joint::LeftHip, Point3::new(-0.12, 0.95, 0.0)),
            (Joint::RightHip, Point3::new(0.12, 0.95, 0.0)),
            (Joint::LeftKnee, Point3::new(-0.12, 0.51, 0.0)),
            (Joint::RightKnee, Point3::new(0.12, 0.51, 0.0)),
            (Joint::LeftAnkle, Point3::new(-0.12, 0.09, 0.0)),
            (Joint::RightAnkle, Point3::new(0.12, 0.09, 0.0)),
        ]
    }

    fn pose_from(joints: &[(Joint, Point3<f64>)], sigma: f64) -> Pose3d {
        let mut pose = Pose3d::empty(Instant::now());
        for (name, point) in joints {
            pose.set(*name, joint(*point, sigma));
        }
        pose
    }

    /// The skeleton the standing body implies, measured properly.
    fn measured() -> Skeleton {
        let mut meter = BoneMeter::default();
        for _ in 0..200 {
            meter.observe(&pose_from(&standing(), 0.004));
        }
        meter.finish()
    }

    #[test]
    fn a_body_that_already_fits_is_left_alone() {
        let skeleton = measured();
        let mut fitter = Fitter::default();
        let fitted = fitter.fit(&pose_from(&standing(), 0.005), &skeleton);

        assert_eq!(fitted.count(), standing().len());
        for (name, truth) in standing() {
            let point = fitted.position(name).unwrap();
            assert!(
                (point - truth).norm() < 1e-6,
                "{name:?} moved to {point:?} from {truth:?}"
            );
        }
    }

    /// The point of the fit: noise that would lengthen a bone is impossible, so
    /// it can be removed outright rather than smoothed away over time.
    #[test]
    fn a_stretched_limb_is_pulled_back_to_its_measured_length() {
        let skeleton = measured();
        let mut joints = standing();
        // Push the left ankle 6 cm further from the knee than the shin allows.
        joints
            .iter_mut()
            .find(|(name, _)| *name == Joint::LeftAnkle)
            .unwrap()
            .1
            .y -= 0.06;

        let mut fitter = Fitter::default();
        let fitted = fitter.fit(&pose_from(&joints, 0.02), &skeleton);

        let knee = fitted.position(Joint::LeftKnee).unwrap();
        let ankle = fitted.position(Joint::LeftAnkle).unwrap();
        let shin = skeleton
            .length(Bone::new(Joint::LeftKnee, Joint::LeftAnkle))
            .unwrap();

        assert!(
            ((ankle - knee).norm() - shin).abs() < 0.005,
            "the shin came out at {} against a measured {shin}",
            (ankle - knee).norm()
        );
    }

    /// The joint that was measured well should stay put and the uncertain one
    /// should give way, not the other way round.
    #[test]
    fn the_uncertain_joint_is_the_one_that_moves() {
        let skeleton = measured();
        let mut joints = standing();
        joints
            .iter_mut()
            .find(|(name, _)| *name == Joint::LeftAnkle)
            .unwrap()
            .1
            .y -= 0.06;

        let mut pose = pose_from(&joints, 0.004);
        // The ankle is the joint nobody could place well.
        pose.set(Joint::LeftAnkle, joint(Point3::new(-0.12, 0.03, 0.0), 0.05));

        let mut fitter = Fitter::default();
        let fitted = fitter.fit(&pose, &skeleton);

        let knee_moved = fitted.get(Joint::LeftKnee).unwrap().correction;
        let ankle_moved = fitted.get(Joint::LeftAnkle).unwrap().correction;

        assert!(
            ankle_moved > 10.0 * knee_moved,
            "the ankle moved {ankle_moved} and the knee {knee_moved}"
        );
    }

    #[test]
    fn nothing_is_left_below_the_floor() {
        let skeleton = measured();
        let joints: Vec<_> = standing()
            .into_iter()
            .map(|(name, mut point)| {
                point.y -= 0.15;
                (name, point)
            })
            .collect();

        let mut fitter = Fitter::default();
        let fitted = fitter.fit(&pose_from(&joints, 0.02), &skeleton);

        for (name, joint) in fitted.iter() {
            assert!(
                joint.point.y >= -1e-9,
                "{name:?} ended up at {}",
                joint.point.y
            );
        }
    }

    /// A joint that goes out of sight should be placed by its neighbours, not
    /// frozen where it was last seen.
    #[test]
    fn an_unseen_joint_is_placed_by_the_joints_around_it() {
        let skeleton = measured();
        let mut fitter = Fitter::default();
        fitter.fit(&pose_from(&standing(), 0.004), &skeleton);

        // The user takes a step: everything moves half a metre except the left
        // knee, which nothing can see any more.
        let moved: Vec<_> = standing()
            .into_iter()
            .filter(|(name, _)| *name != Joint::LeftKnee)
            .map(|(name, mut point)| {
                point.z -= 0.5;
                (name, point)
            })
            .collect();

        let fitted = fitter.fit(&pose_from(&moved, 0.004), &skeleton);
        let knee = fitted.get(Joint::LeftKnee).expect("it should be filled in");

        assert!(knee.inferred);
        assert!(
            knee.point.z < -0.35,
            "the knee stayed behind at z = {}",
            knee.point.z
        );

        let hip = fitted.position(Joint::LeftHip).unwrap();
        let thigh = skeleton
            .length(Bone::new(Joint::LeftHip, Joint::LeftKnee))
            .unwrap();
        assert!(
            ((knee.point - hip).norm() - thigh).abs() < 0.01,
            "the inferred knee sits {} from its hip, not {thigh}",
            (knee.point - hip).norm()
        );
    }

    #[test]
    fn a_joint_that_was_never_seen_is_not_invented() {
        let skeleton = measured();
        let mut fitter = Fitter::default();

        let partial: Vec<_> = standing()
            .into_iter()
            .filter(|(name, _)| *name != Joint::LeftKnee)
            .collect();
        let fitted = fitter.fit(&pose_from(&partial, 0.004), &skeleton);

        assert!(fitted.get(Joint::LeftKnee).is_none());
    }

    /// A leg cannot bend backwards. The cameras will nonetheless say it does,
    /// and left alone the leg snaps between two mirror-image bends.
    #[test]
    fn a_backwards_knee_is_folded_the_right_way() {
        let skeleton = measured();
        let mut joints = standing();
        // Bend the left leg, with the knee behind the body instead of in front.
        for (name, point) in joints.iter_mut() {
            match name {
                Joint::LeftKnee => *point = Point3::new(-0.12, 0.55, 0.14),
                Joint::LeftAnkle => *point = Point3::new(-0.12, 0.18, 0.0),
                _ => {}
            }
        }

        let mut fitter = Fitter::default();
        let fitted = fitter.fit(&pose_from(&joints, 0.02), &skeleton);
        let knee = fitted.position(Joint::LeftKnee).unwrap();

        // The body faces -Z, so a knee in front of the hip-ankle line has
        // negative z here.
        assert!(
            knee.z < 0.0,
            "the knee stayed behind the leg at z = {}",
            knee.z
        );
    }

    #[test]
    fn a_forwards_knee_is_left_where_it_is() {
        let skeleton = measured();
        let mut joints = standing();
        for (name, point) in joints.iter_mut() {
            match name {
                Joint::LeftKnee => *point = Point3::new(-0.12, 0.55, -0.14),
                Joint::LeftAnkle => *point = Point3::new(-0.12, 0.18, 0.0),
                _ => {}
            }
        }

        let mut fitter = Fitter::default();
        let fitted = fitter.fit(&pose_from(&joints, 0.004), &skeleton);
        let knee = fitted.position(Joint::LeftKnee).unwrap();

        assert!((knee - Point3::new(-0.12, 0.55, -0.14)).norm() < 0.02);
    }

    /// Without a measurement there is nothing to hold the body to, and the fit
    /// must pass the reconstruction through rather than invent a skeleton.
    #[test]
    fn an_unmeasured_body_is_not_reshaped() {
        let mut fitter = Fitter::default();
        let pose = pose_from(&standing(), 0.005);
        let fitted = fitter.fit(&pose, &Skeleton::default());

        for (name, truth) in standing() {
            assert!((fitted.position(name).unwrap() - truth).norm() < 1e-9);
        }
    }

    /// A length that never settled is a number out of noise, and holding the
    /// body to it is worse than holding it to nothing.
    #[test]
    fn an_unsettled_length_is_not_enforced() {
        let skeleton = Skeleton {
            bones: vec![BoneLength {
                bone: Bone::new(Joint::LeftKnee, Joint::LeftAnkle),
                length: 0.20,
                samples: 4,
                scatter: 0.15,
            }],
            measured_at: None,
        };

        let mut fitter = Fitter::default();
        let fitted = fitter.fit(&pose_from(&standing(), 0.005), &skeleton);

        let knee = fitted.position(Joint::LeftKnee).unwrap();
        let ankle = fitted.position(Joint::LeftAnkle).unwrap();
        assert!(
            ((ankle - knee).norm() - 0.42).abs() < 1e-6,
            "the shin was dragged to {}",
            (ankle - knee).norm()
        );
    }

    /// A skeleton that disagrees with the cameras is a real failure — a body
    /// measured while the calibration was wrong, most likely. The fit cannot
    /// resolve it, but it must not hide it either.
    #[test]
    fn a_hopeless_disagreement_is_reported() {
        let skeleton = Skeleton {
            bones: vec![BoneLength {
                bone: Bone::new(Joint::LeftHip, Joint::LeftKnee),
                length: 1.40,
                samples: 4000,
                scatter: 0.001,
            }],
            measured_at: None,
        };

        let mut fitter = Fitter::default();
        let fitted = fitter.fit(&pose_from(&standing(), 0.005), &skeleton);

        assert!(
            fitted.worst_correction() > 0.3,
            "a metre of disagreement was reported as {}",
            fitted.worst_correction()
        );
        // And a body that fits reports nothing, so the number means something.
        let mut honest = Fitter::default();
        assert!(
            honest
                .fit(&pose_from(&standing(), 0.005), &measured())
                .worst_correction()
                < 1e-6
        );
    }
}
