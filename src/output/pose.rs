//! Turning a bag of joint positions into trackers with orientations.
//!
//! A tracker is a rigid body: a position and a rotation. The reconstruction is
//! neither — it is a set of independent points, each with its own uncertainty
//! and no orientation at all. Every orientation sent out is therefore inferred
//! from the positions of two or three joints, and the quality of a tracker's
//! rotation is the quality of the *worst* joint in the limb it was built from,
//! not of the joint it is nominally attached to.
//!
//! That matters more than it sounds. A foot's yaw comes from the line between
//! its heel and its toe — twenty centimetres apart, both at the far end of the
//! body from the cameras, both frequently occluded. Two centimetres of error in
//! either is six degrees of yaw. It is why a WholeBody model is the default:
//! with a COCO-17 layout there is no heel and no toe, and foot yaw has to fall
//! back to the shin, which cannot see the difference between standing and
//! standing with the feet turned out.

use std::time::Instant;

use nalgebra::{Isometry3, Matrix3, Point3, Rotation3, Translation3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};

use crate::fusion::filter::Filtered;
use crate::models::Joint;

/// Two axes closer to parallel than this cannot define a frame between them.
/// About two and a half degrees.
const MIN_SEPARATION: f64 = 0.045;

/// Shortest limb worth taking a direction from, in metres. Below this the
/// direction is mostly the noise in the two endpoints.
const MIN_LIMB: f64 = 0.04;

/// Where up the spine the chest tracker sits, as a fraction of hip to neck.
///
/// The sternum rather than the collarbone: a chest tracker mounted at the neck
/// makes an avatar's upper body pivot about the wrong point, and there is no
/// keypoint at the sternum to use instead.
const CHEST_HEIGHT: f64 = 0.75;

/// A tracker Optra can drive.
///
/// Eight, which is what VRChat's OSC tracker API accepts. The lower six are the
/// ones this application exists for; the elbows are here because the machinery
/// is identical and a user with a headset that loses its controllers behind
/// their back may want them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrackerRole {
    Hip,
    Chest,
    LeftKnee,
    RightKnee,
    LeftFoot,
    RightFoot,
    LeftElbow,
    RightElbow,
}

impl TrackerRole {
    /// Every role, in the order they are assigned tracker indices.
    ///
    /// Trunk down to the extremities, and left before right within a pair —
    /// which is also the order they matter in. See [`assign`] for what the
    /// order is used for.
    ///
    /// [`assign`]: super::sink::assign
    pub const ALL: [TrackerRole; 8] = [
        TrackerRole::Hip,
        TrackerRole::Chest,
        TrackerRole::LeftKnee,
        TrackerRole::RightKnee,
        TrackerRole::LeftFoot,
        TrackerRole::RightFoot,
        TrackerRole::LeftElbow,
        TrackerRole::RightElbow,
    ];

    pub fn label(self) -> &'static str {
        match self {
            TrackerRole::Hip => "hip",
            TrackerRole::Chest => "chest",
            TrackerRole::LeftKnee => "left knee",
            TrackerRole::RightKnee => "right knee",
            TrackerRole::LeftFoot => "left foot",
            TrackerRole::RightFoot => "right foot",
            TrackerRole::LeftElbow => "left elbow",
            TrackerRole::RightElbow => "right elbow",
        }
    }

    /// The three that make full-body tracking work at all.
    ///
    /// Hips and both feet are the minimum VRChat will calibrate against, and
    /// they are also the three this application is most able to place well:
    /// they are the joints a camera looking at a standing person can see.
    pub fn is_essential(self) -> bool {
        matches!(
            self,
            TrackerRole::Hip | TrackerRole::LeftFoot | TrackerRole::RightFoot
        )
    }

    /// Joints this tracker's pose is built from.
    ///
    /// Used to say why a tracker is missing without repeating the derivation,
    /// and to judge how much of the body a chosen set of trackers depends on.
    pub fn needs(self) -> &'static [Joint] {
        match self {
            TrackerRole::Hip => &[Joint::LeftHip, Joint::RightHip],
            TrackerRole::Chest => &[Joint::LeftShoulder, Joint::RightShoulder, Joint::Hip],
            TrackerRole::LeftKnee => &[Joint::LeftHip, Joint::LeftKnee, Joint::LeftAnkle],
            TrackerRole::RightKnee => &[Joint::RightHip, Joint::RightKnee, Joint::RightAnkle],
            TrackerRole::LeftFoot => &[Joint::LeftKnee, Joint::LeftAnkle],
            TrackerRole::RightFoot => &[Joint::RightKnee, Joint::RightAnkle],
            TrackerRole::LeftElbow => &[Joint::LeftShoulder, Joint::LeftElbow, Joint::LeftWrist],
            TrackerRole::RightElbow => {
                &[Joint::RightShoulder, Joint::RightElbow, Joint::RightWrist]
            }
        }
    }
}

/// One joint as the output stage sees it.
#[derive(Debug, Clone, Copy)]
pub struct PostureJoint {
    pub point: Point3<f64>,
    pub sigma: f64,
    pub inferred: bool,
}

/// A body at one instant, ready to be turned into trackers.
///
/// Separate from [`Filtered`] because it is the *extrapolated* body: every
/// position in it has been carried forward to the instant the trackers are
/// being sent for, which is not the instant fusion reconstructed.
#[derive(Debug, Clone)]
pub struct Posture {
    pub at: Instant,
    joints: Vec<Option<PostureJoint>>,
}

impl Posture {
    pub fn empty(at: Instant) -> Self {
        Self {
            at,
            joints: (0..Joint::ALL.len()).map(|_| None).collect(),
        }
    }

    /// Carries a filtered pose forward to `at`.
    ///
    /// The output stage sends faster than fusion runs, so most sends have no
    /// new reconstruction behind them and would otherwise repeat the last one —
    /// a tracker that moves in sixty-hertz steps while claiming to update at
    /// ninety. Each joint already carries a velocity and knows which instant
    /// its own prediction was made for, so the remaining distance is the only
    /// thing left to travel.
    ///
    /// Extrapolating each joint on its own does not preserve bone lengths
    /// exactly. Over the tens of milliseconds involved the error is under a
    /// millimetre, and every orientation built from these positions is
    /// orthonormalised anyway, so nothing downstream depends on it.
    ///
    /// How far ahead it reaches is not a fixed horizon. A reconstruction
    /// describes an instant already some way in the past — the fusion clock
    /// deliberately runs behind so that every camera has delivered — and that
    /// distance is *measured*, not configured. So the lead is the frame's own
    /// age plus the delay still to come, which means a frame sent again while
    /// waiting for the next one is predicted further forward each time rather
    /// than repeating itself.
    pub fn predicted(filtered: &Filtered, at: Instant) -> Self {
        let mut posture = Posture::empty(at);

        // Signed: a send instant behind the reconstruction is not a negative
        // duration but a request to interpolate back towards it.
        let age = if at >= filtered.at {
            at.duration_since(filtered.at).as_secs_f64()
        } else {
            -filtered.at.duration_since(at).as_secs_f64()
        };

        for (joint, state) in filtered.iter() {
            posture.set(
                joint,
                PostureJoint {
                    // `lead` is the configured horizon plus whatever smoothing
                    // lag this joint is owed, measured from the frame's own
                    // instant, so the age is all that has to be added.
                    point: state.extrapolate(age + state.lead, filtered.limit),
                    sigma: state.sigma,
                    inferred: state.inferred,
                },
            );
        }

        posture
    }

    pub fn get(&self, joint: Joint) -> Option<PostureJoint> {
        self.joints[joint.index()]
    }

    pub fn point(&self, joint: Joint) -> Option<Point3<f64>> {
        self.get(joint).map(|joint| joint.point)
    }

    pub fn set(&mut self, joint: Joint, state: PostureJoint) {
        self.joints[joint.index()] = Some(state);
    }

    pub fn count(&self) -> usize {
        self.joints.iter().filter(|joint| joint.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// The midpoint of two joints, when both are there.
    fn between(&self, left: Joint, right: Joint) -> Option<Point3<f64>> {
        let (left, right) = (self.point(left)?, self.point(right)?);
        Some(Point3::from((left.coords + right.coords) * 0.5))
    }

    /// Where the pelvis is, whether or not the model has a joint for it.
    ///
    /// Halpe-26 has a hip midpoint; COCO-17 does not, and the midpoint of the
    /// two hips is what it would have been.
    fn pelvis(&self) -> Option<Point3<f64>> {
        self.point(Joint::Hip)
            .or_else(|| self.between(Joint::LeftHip, Joint::RightHip))
    }

    /// The top of the spine, whether or not the model has a joint for it.
    fn shoulders(&self) -> Option<Point3<f64>> {
        self.point(Joint::Neck)
            .or_else(|| self.between(Joint::LeftShoulder, Joint::RightShoulder))
    }

    /// The body's own right, from the hips.
    ///
    /// Every tracker whose own limb cannot say which way it is facing falls
    /// back to this — a straight leg has no bend plane, and a foot with no heel
    /// or toe keypoint has no direction of its own at all.
    fn facing(&self) -> Option<Vector3<f64>> {
        let across = self.point(Joint::RightHip)? - self.point(Joint::LeftHip)?;
        (across.norm() > MIN_LIMB).then_some(across)
    }

    /// The worst uncertainty among the joints a tracker was built from, and
    /// whether any of them was placed by the fit rather than seen.
    fn quality(&self, joints: &[Joint]) -> (f64, bool) {
        joints
            .iter()
            .filter_map(|joint| self.get(*joint))
            .fold((0.0f64, false), |(sigma, inferred), joint| {
                (sigma.max(joint.sigma), inferred || joint.inferred)
            })
    }

    /// Builds one tracker, or says nothing rather than guessing.
    ///
    /// A tracker that cannot be derived is left out of the frame entirely. The
    /// alternative — sending the last known pose, or an identity rotation —
    /// puts a foot on the floor pointing north and leaves the user wondering
    /// which part of the system is lying to them.
    pub fn derive(&self, role: TrackerRole) -> Option<TrackerPose> {
        let (position, rotation) = match role {
            TrackerRole::Hip => {
                let pelvis = self.pelvis()?;
                let up = self.shoulders()? - pelvis;
                (pelvis, frame(up, self.facing()?)?)
            }
            TrackerRole::Chest => {
                let pelvis = self.pelvis()?;
                let up = self.shoulders()? - pelvis;
                let across = self.point(Joint::RightShoulder)? - self.point(Joint::LeftShoulder)?;
                (pelvis + up * CHEST_HEIGHT, frame(up, across)?)
            }
            TrackerRole::LeftKnee => self.bend(
                Joint::LeftHip,
                Joint::LeftKnee,
                Joint::LeftAnkle,
                Fold::Backwards,
            )?,
            TrackerRole::RightKnee => self.bend(
                Joint::RightHip,
                Joint::RightKnee,
                Joint::RightAnkle,
                Fold::Backwards,
            )?,
            TrackerRole::LeftFoot => self.foot(
                Joint::LeftKnee,
                Joint::LeftAnkle,
                Joint::LeftHeel,
                Joint::LeftBigToe,
            )?,
            TrackerRole::RightFoot => self.foot(
                Joint::RightKnee,
                Joint::RightAnkle,
                Joint::RightHeel,
                Joint::RightBigToe,
            )?,
            TrackerRole::LeftElbow => self.bend(
                Joint::LeftShoulder,
                Joint::LeftElbow,
                Joint::LeftWrist,
                Fold::Forwards,
            )?,
            TrackerRole::RightElbow => self.bend(
                Joint::RightShoulder,
                Joint::RightElbow,
                Joint::RightWrist,
                Fold::Forwards,
            )?,
        };

        let (sigma, inferred) = self.quality(role.needs());
        Some(TrackerPose {
            role,
            pose: Isometry3::from_parts(Translation3::from(position.coords), rotation),
            sigma,
            inferred,
        })
    }

    /// A joint in the middle of a limb that bends: knees and elbows.
    ///
    /// Up the limb towards the body, with the bend plane deciding which way it
    /// faces. A straight limb has no bend plane — the two segments are
    /// collinear and their cross product is noise — so it falls back to the
    /// hips. That is the common case, not the exception: a knee is straight for
    /// most of the time anyone is standing.
    fn bend(
        &self,
        upper: Joint,
        middle: Joint,
        lower: Joint,
        fold: Fold,
    ) -> Option<(Point3<f64>, UnitQuaternion<f64>)> {
        let middle_point = self.point(middle)?;
        let up = self.point(upper)? - middle_point;

        let across = self
            .point(lower)
            .map(|lower| up.cross(&(lower - middle_point)) * fold.sign())
            .filter(|normal| normal.norm() > MIN_SEPARATION * up.norm())
            .or_else(|| self.facing())?;

        Some((middle_point, frame(up, across)?))
    }

    /// A foot: at the ankle, up the shin, pointing where the toes point.
    ///
    /// Falling back to the hips when there is no heel or toe keypoint is a real
    /// loss and not a small one — it is the difference between a foot that
    /// turns and a foot welded to the pelvis — but it is still better than a
    /// yaw taken from the shin, which barely rotates when the foot does.
    fn foot(
        &self,
        knee: Joint,
        ankle: Joint,
        heel: Joint,
        toe: Joint,
    ) -> Option<(Point3<f64>, UnitQuaternion<f64>)> {
        let ankle_point = self.point(ankle)?;
        let up = self.point(knee)? - ankle_point;

        let sole = self
            .point(heel)
            .zip(self.point(toe))
            .map(|(heel, toe)| toe - heel)
            .filter(|sole| sole.norm() > MIN_LIMB);

        let right = match sole {
            Some(forward) => forward.cross(&up),
            None => self.facing()?,
        };

        Some((ankle_point, frame(up, right)?))
    }
}

/// Which way a hinge joint closes.
///
/// A knee and an elbow are the same three points and the same cross product,
/// and the plane normal that comes out points opposite ways for the two of
/// them: the kneecap is on the front of the body and the point of the elbow is
/// on the back. Both trackers still have to face forwards.
#[derive(Debug, Clone, Copy)]
enum Fold {
    /// Knees: the lower segment swings behind the upper one.
    Backwards,
    /// Elbows: the lower segment swings in front of it.
    Forwards,
}

impl Fold {
    fn sign(self) -> f64 {
        match self {
            Fold::Backwards => 1.0,
            Fold::Forwards => -1.0,
        }
    }
}

/// One tracker at one instant.
#[derive(Debug, Clone)]
pub struct TrackerPose {
    pub role: TrackerRole,
    /// Tracker-to-world, in the standing universe: right-handed, +Y up, metres.
    /// Each sink converts to whatever its consumer wants.
    pub pose: Isometry3<f64>,
    /// Worst uncertainty among the joints this was built from, in metres.
    pub sigma: f64,
    /// True when any of those joints was placed by the fit rather than seen.
    pub inferred: bool,
}

/// An orientation from an up axis and a rough idea of right.
///
/// The convention is OpenVR's, which is also the one the rest of this
/// application uses: +X right, +Y up, and the device looking down its own -Z.
/// `reference` only has to be roughly right; the component of it along `up` is
/// removed, so the caller may hand over a hip axis on a leaning body without
/// having to square it up first.
///
/// Returns nothing rather than something arbitrary when the two axes are too
/// close to parallel to span a plane. A tracker missing for a frame is a
/// visible, diagnosable fault; a tracker with a rotation picked out of a
/// degenerate cross product is a limb that snaps to a random pose and stays
/// there.
fn frame(up: Vector3<f64>, reference: Vector3<f64>) -> Option<UnitQuaternion<f64>> {
    let up = up.try_normalize(1e-9)?;
    let reference = reference.try_normalize(1e-9)?;

    let right = reference - up * reference.dot(&up);
    let right = right.try_normalize(MIN_SEPARATION)?;

    // Right-handed, so right x up is the device's own +Z, which points
    // backwards out of a device that looks down -Z.
    let back = right.cross(&up);

    Some(UnitQuaternion::from_rotation_matrix(
        &Rotation3::from_matrix_unchecked(Matrix3::from_columns(&[right, up, back])),
    ))
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_PI_2;

    use super::*;

    /// A body standing upright at the origin, facing -Z, in a Halpe-26 layout.
    fn standing() -> Posture {
        let mut posture = Posture::empty(Instant::now());
        for (joint, point) in [
            (Joint::Neck, Point3::new(0.0, 1.45, 0.0)),
            (Joint::LeftShoulder, Point3::new(-0.18, 1.42, 0.0)),
            (Joint::RightShoulder, Point3::new(0.18, 1.42, 0.0)),
            (Joint::LeftElbow, Point3::new(-0.20, 1.15, 0.0)),
            (Joint::RightElbow, Point3::new(0.20, 1.15, 0.0)),
            (Joint::LeftWrist, Point3::new(-0.21, 0.90, 0.0)),
            (Joint::RightWrist, Point3::new(0.21, 0.90, 0.0)),
            (Joint::Hip, Point3::new(0.0, 0.95, 0.0)),
            (Joint::LeftHip, Point3::new(-0.10, 0.95, 0.0)),
            (Joint::RightHip, Point3::new(0.10, 0.95, 0.0)),
            (Joint::LeftKnee, Point3::new(-0.10, 0.52, 0.0)),
            (Joint::RightKnee, Point3::new(0.10, 0.52, 0.0)),
            (Joint::LeftAnkle, Point3::new(-0.10, 0.09, 0.0)),
            (Joint::RightAnkle, Point3::new(0.10, 0.09, 0.0)),
            (Joint::LeftHeel, Point3::new(-0.10, 0.05, 0.05)),
            (Joint::RightHeel, Point3::new(0.10, 0.05, 0.05)),
            (Joint::LeftBigToe, Point3::new(-0.10, 0.03, -0.15)),
            (Joint::RightBigToe, Point3::new(0.10, 0.03, -0.15)),
        ] {
            posture.set(
                joint,
                PostureJoint {
                    point,
                    sigma: 0.01,
                    inferred: false,
                },
            );
        }
        posture
    }

    /// Where a tracker's own -Z points, in world coordinates.
    fn facing(pose: &TrackerPose) -> Vector3<f64> {
        pose.pose.rotation * -Vector3::z()
    }

    #[test]
    fn a_body_facing_forward_produces_trackers_facing_forward() {
        let posture = standing();

        for role in TrackerRole::ALL {
            let tracker = posture
                .derive(role)
                .unwrap_or_else(|| panic!("{} could not be derived", role.label()));

            let forward = facing(&tracker);
            assert!(
                forward.z < -0.9,
                "{} faces {forward:?}, which is not forward",
                role.label()
            );

            let up = tracker.pose.rotation * Vector3::y();
            assert!(up.y > 0.9, "{} has up pointing {up:?}", role.label());
        }
    }

    /// The whole point of a hip tracker: an avatar turns where the body turns,
    /// and the body's yaw is not the headset's.
    #[test]
    fn turning_the_hips_turns_the_hip_tracker() {
        let mut posture = standing();
        let turn = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), FRAC_PI_2);
        let pelvis = Vector3::new(0.0, 0.95, 0.0);

        for joint in [Joint::LeftHip, Joint::RightHip] {
            let point = posture.point(joint).unwrap();
            posture.set(
                joint,
                PostureJoint {
                    point: Point3::from(turn * (point.coords - pelvis) + pelvis),
                    sigma: 0.01,
                    inferred: false,
                },
            );
        }

        let hip = posture.derive(TrackerRole::Hip).unwrap();
        let forward = facing(&hip);
        // Turned a quarter turn about +Y: what was -Z is now -X.
        assert!(
            forward.x < -0.9,
            "the hips turned but the tracker faces {forward:?}"
        );
    }

    /// A foot points where the toes point, not where the shin does. This is the
    /// case that separates a WholeBody model from a COCO-17 one.
    #[test]
    fn a_foot_turned_out_reads_as_turned_out() {
        let mut posture = standing();
        let turn = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), -FRAC_PI_2);
        let ankle = Vector3::new(-0.10, 0.09, 0.0);

        for joint in [Joint::LeftHeel, Joint::LeftBigToe] {
            let point = posture.point(joint).unwrap();
            posture.set(
                joint,
                PostureJoint {
                    point: Point3::from(turn * (point.coords - ankle) + ankle),
                    sigma: 0.01,
                    inferred: false,
                },
            );
        }

        let foot = posture.derive(TrackerRole::LeftFoot).unwrap();
        let forward = facing(&foot);
        assert!(
            forward.x > 0.85,
            "the foot was turned out and the tracker faces {forward:?}"
        );

        // The other foot did not move, and must not have followed.
        let other = facing(&posture.derive(TrackerRole::RightFoot).unwrap());
        assert!(
            other.z < -0.9,
            "the right foot followed the left to {other:?}"
        );
    }

    /// Without heel and toe keypoints the foot has no yaw of its own, and falls
    /// back to the hips rather than dropping out.
    #[test]
    fn a_foot_with_no_toes_still_produces_a_tracker() {
        let mut posture = standing();
        posture.joints[Joint::LeftHeel.index()] = None;
        posture.joints[Joint::LeftBigToe.index()] = None;

        let foot = posture
            .derive(TrackerRole::LeftFoot)
            .expect("a foot with a shin and a hip axis is still placeable");
        let forward = facing(&foot);
        assert!(forward.z < -0.9, "the fallback faces {forward:?}");
    }

    /// A bent knee faces where the bend faces, which is the case the fallback
    /// to the hips exists to stand in for.
    #[test]
    fn a_bent_knee_takes_its_facing_from_the_bend() {
        let mut posture = standing();
        // Ankle swung back, as it is halfway through a stride.
        posture.set(
            Joint::LeftAnkle,
            PostureJoint {
                point: Point3::new(-0.10, 0.20, 0.35),
                sigma: 0.01,
                inferred: false,
            },
        );

        let knee = posture.derive(TrackerRole::LeftKnee).unwrap();
        let forward = facing(&knee);
        assert!(
            forward.z < -0.8,
            "a knee bent backwards should still face forwards, and faces {forward:?}"
        );
    }

    /// An elbow folds the other way from a knee, off the same three points and
    /// the same cross product. Both trackers still have to face forwards, and
    /// the sign that makes one right makes the other backwards.
    #[test]
    fn a_bent_elbow_faces_the_same_way_as_a_bent_knee() {
        let mut posture = standing();
        // Forearm raised in front, as when holding a controller.
        posture.set(
            Joint::LeftWrist,
            PostureJoint {
                point: Point3::new(-0.20, 1.15, -0.25),
                sigma: 0.01,
                inferred: false,
            },
        );

        let elbow = posture.derive(TrackerRole::LeftElbow).unwrap();
        let forward = facing(&elbow);
        assert!(
            forward.z < -0.8,
            "a bent elbow should still face forwards, and faces {forward:?}"
        );
    }

    #[test]
    fn a_tracker_takes_the_worst_uncertainty_of_its_limb() {
        let mut posture = standing();
        posture.set(
            Joint::LeftKnee,
            PostureJoint {
                point: posture.point(Joint::LeftKnee).unwrap(),
                sigma: 0.07,
                inferred: true,
            },
        );

        let foot = posture.derive(TrackerRole::LeftFoot).unwrap();
        assert!((foot.sigma - 0.07).abs() < 1e-9, "sigma was {}", foot.sigma);
        assert!(
            foot.inferred,
            "a foot built on an inferred knee is itself inferred"
        );

        // The other leg shares nothing with it and must be unaffected.
        let other = posture.derive(TrackerRole::RightFoot).unwrap();
        assert!(!other.inferred);
    }

    /// Half a body is a real state — the cameras see the legs and the user's
    /// arms are behind their back — and it must produce the trackers it can
    /// rather than nothing.
    #[test]
    fn missing_joints_drop_only_the_trackers_that_needed_them() {
        let mut posture = standing();
        for joint in [
            Joint::LeftShoulder,
            Joint::RightShoulder,
            Joint::LeftElbow,
            Joint::RightElbow,
            Joint::LeftWrist,
            Joint::RightWrist,
            Joint::Neck,
        ] {
            posture.joints[joint.index()] = None;
        }

        for role in [
            TrackerRole::LeftFoot,
            TrackerRole::RightFoot,
            TrackerRole::LeftKnee,
            TrackerRole::RightKnee,
        ] {
            assert!(
                posture.derive(role).is_some(),
                "{} needs nothing above the hips",
                role.label()
            );
        }

        // The hips need the spine to know which way is up, and the chest and
        // elbows need the arms.
        for role in [
            TrackerRole::Hip,
            TrackerRole::Chest,
            TrackerRole::LeftElbow,
            TrackerRole::RightElbow,
        ] {
            assert!(
                posture.derive(role).is_none(),
                "{} was derived from joints that are not there",
                role.label()
            );
        }
    }

    /// The output stage sends faster than fusion runs, so most sends are of a
    /// frame that has already been sent. Each one has to reach further ahead
    /// than the last, or the trackers move in fusion-rate steps while claiming
    /// a higher rate.
    #[test]
    fn a_frame_sent_again_is_predicted_further_ahead() {
        use std::time::Duration;

        use crate::fusion::filter::{FilterOptions, FilteredJoint};

        let options = FilterOptions::default();
        let at = Instant::now();
        let mut filtered = Filtered::empty(at, options.horizon);
        filtered.limit = options.max_prediction;
        filtered.set(
            Joint::LeftAnkle,
            FilteredJoint {
                point: Point3::origin(),
                velocity: Vector3::new(0.0, 0.0, -2.0),
                predicted: Point3::origin(),
                lead: 0.05,
                sigma: 0.01,
                inferred: false,
            },
        );

        let sooner = Posture::predicted(&filtered, at + Duration::from_millis(10));
        let later = Posture::predicted(&filtered, at + Duration::from_millis(21));

        let sooner = sooner.point(Joint::LeftAnkle).unwrap();
        let later = later.point(Joint::LeftAnkle).unwrap();

        // Eleven milliseconds further on at two metres a second.
        let moved = sooner.z - later.z;
        assert!(
            (moved - 0.022).abs() < 1e-6,
            "the second send moved the ankle {moved} m, not 22 mm"
        );
        // And both are ahead of where the joint was measured.
        assert!(
            sooner.z < -0.11,
            "the first send barely predicted: {sooner:?}"
        );
    }

    /// A velocity that has gone wrong should show as a foot that stopped
    /// tracking, not one that left the room — so the limit is measured from
    /// the joint, not accumulated per send.
    #[test]
    fn prediction_stops_rather_than_running_away() {
        use std::time::Duration;

        use crate::fusion::filter::{FilterOptions, FilteredJoint};

        let options = FilterOptions::default();
        let at = Instant::now();
        let mut filtered = Filtered::empty(at, options.horizon);
        filtered.limit = options.max_prediction;
        filtered.set(
            Joint::LeftAnkle,
            FilteredJoint {
                point: Point3::origin(),
                velocity: Vector3::new(0.0, 0.0, -40.0),
                predicted: Point3::origin(),
                lead: 0.05,
                sigma: 0.01,
                inferred: false,
            },
        );

        let posture = Posture::predicted(&filtered, at + Duration::from_millis(500));
        let ankle = posture.point(Joint::LeftAnkle).unwrap();
        assert!(
            ankle.coords.norm() <= options.max_prediction + 1e-9,
            "a runaway velocity threw the ankle {} m",
            ankle.coords.norm()
        );
    }

    #[test]
    fn two_parallel_axes_do_not_make_a_frame() {
        assert!(frame(Vector3::y(), Vector3::y()).is_none());
        assert!(frame(Vector3::y(), Vector3::new(0.0, 1.0, 0.001)).is_none());
        assert!(frame(Vector3::zeros(), Vector3::x()).is_none());
        assert!(frame(Vector3::y(), Vector3::x()).is_some());
    }
}
