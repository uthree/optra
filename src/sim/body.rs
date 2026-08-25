//! The body the simulated room contains, and how it moves.
//!
//! Two things come out of here and they have to agree exactly: the joint
//! positions that stand in for ground truth, and the surface the renderer
//! draws. They agree because there is only one source for both — a posture is
//! built by forward kinematics from unit directions, so every bone is exactly
//! its stated length at every instant, and the mesh is hung on the result.
//!
//! A body assembled from independent sine waves would have limbs that stretch,
//! and an accuracy figure measured against it would be measuring something no
//! reconstruction could reproduce.

use nalgebra::{Point3, Rotation3, Unit, Vector3};

use crate::models::Joint;

/// The skeleton, in metres.
///
/// These are the lengths the ground truth is built from. The default is an
/// average adult of about 1.72 m; the proportions follow the usual
/// anthropometric fractions of standing height rather than being invented, so
/// that a pose model trained on people has something the right shape to look
/// at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Anatomy {
    /// Ankle joint above the floor.
    pub ankle_height: f64,
    /// Ankle to knee.
    pub shin: f64,
    /// Knee to hip.
    pub thigh: f64,
    /// Half the distance between the two hip joints.
    pub hip_half_width: f64,
    /// Pelvis centre to the base of the neck.
    pub spine: f64,
    /// Base of the neck to the centre of the head.
    pub neck: f64,
    /// Half the distance between the two shoulder joints.
    pub shoulder_half_width: f64,
    /// Shoulder to elbow.
    pub upper_arm: f64,
    /// Elbow to wrist.
    pub forearm: f64,
    /// Radius of the head, which also places the face keypoints.
    pub head_radius: f64,
    /// Heel behind the ankle.
    pub heel_back: f64,
    /// Heel above the floor.
    pub heel_height: f64,
    /// Big toe ahead of the ankle.
    pub toe_forward: f64,
    /// Toes above the floor.
    pub toe_height: f64,
    /// Half the distance between the big and small toe points.
    pub toe_half_width: f64,
}

impl Default for Anatomy {
    fn default() -> Self {
        Self {
            ankle_height: 0.07,
            shin: 0.41,
            thigh: 0.42,
            hip_half_width: 0.09,
            spine: 0.50,
            neck: 0.22,
            shoulder_half_width: 0.19,
            upper_arm: 0.30,
            forearm: 0.26,
            head_radius: 0.10,
            heel_back: 0.06,
            heel_height: 0.04,
            toe_forward: 0.14,
            toe_height: 0.03,
            toe_half_width: 0.035,
        }
    }
}

impl Anatomy {
    /// Standing height, floor to the top of the head.
    pub fn standing_height(&self) -> f64 {
        self.ankle_height + self.shin + self.thigh + self.spine + self.neck + self.head_radius
    }

    /// Height of the pelvis with the legs straight, which is where a posture
    /// starts before any of the walk is applied.
    pub fn hip_height(&self) -> f64 {
        self.ankle_height + self.shin + self.thigh
    }
}

/// Where every joint is at one instant, with the frame the body stands in.
///
/// The frame travels with the body: `facing` is the direction it walks and
/// `right` is its right-hand side. Everything the mesh needs to be oriented —
/// which way the feet point, which way the face looks — comes from these rather
/// than from the joint positions, because a straight limb has no orientation of
/// its own.
#[derive(Debug, Clone)]
pub struct Posture {
    joints: [Option<Point3<f64>>; Joint::ALL.len()],
    pub facing: Vector3<f64>,
    pub right: Vector3<f64>,
}

impl Posture {
    fn new(facing: Vector3<f64>) -> Self {
        let facing = facing.normalize();
        Self {
            joints: [None; Joint::ALL.len()],
            facing,
            right: facing.cross(&Vector3::y()).normalize(),
        }
    }

    pub fn get(&self, joint: Joint) -> Option<Point3<f64>> {
        self.joints[joint.index()]
    }

    fn set(&mut self, joint: Joint, position: Point3<f64>) {
        self.joints[joint.index()] = Some(position);
    }

    pub fn iter(&self) -> impl Iterator<Item = (Joint, Point3<f64>)> + '_ {
        Joint::ALL
            .iter()
            .filter_map(|joint| self.get(*joint).map(|point| (*joint, point)))
    }

    pub fn count(&self) -> usize {
        self.joints.iter().filter(|joint| joint.is_some()).count()
    }
}

/// A walk around the middle of the room.
///
/// The figure circles rather than crossing, so every camera in the room sees it
/// from every side within one lap, and the legs swing about the hips so that
/// the lower body moves independently of the body as a whole. A body that only
/// translated would let a reconstruction with the limbs completely wrong still
/// look right.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Walk {
    /// Radius of the circle walked, in metres.
    pub radius: f64,
    /// How fast the figure goes round, in radians per second.
    pub turn: f64,
    /// Strides per second.
    pub cadence: f64,
    /// Hip swing amplitude, in radians.
    pub swing: f64,
    /// Knee fold amplitude, in radians. The knee only ever folds one way,
    /// which is what keeps the shin on the side of the leg a shin can be on.
    pub knee: f64,
    /// Shoulder swing amplitude, in radians.
    pub arm: f64,
    /// How far the arms are held from the body, in radians.
    pub splay: f64,
    /// Rise and fall of the pelvis over a stride, in metres.
    pub bob: f64,
}

impl Default for Walk {
    fn default() -> Self {
        Self {
            radius: 1.05,
            turn: 0.45,
            cadence: 0.85,
            swing: 0.42,
            knee: 0.55,
            arm: 0.30,
            splay: 0.12,
            bob: 0.025,
        }
    }
}

impl Walk {
    /// The posture at `t` seconds from the start of the walk.
    pub fn posture(&self, anatomy: &Anatomy, t: f64) -> Posture {
        let around = self.turn * t;
        let mut posture = Posture::new(Vector3::new(around.cos(), 0.0, -around.sin()));
        let (facing, right) = (posture.facing, posture.right);
        let up = Vector3::y();

        // Two footfalls to a stride, so the pelvis rises and falls at twice the
        // stride rate. This is the only part of the body not driven by an
        // angle, and it is here because a walk with a rigid pelvis height looks
        // like a puppet on a rail.
        let stride = std::f64::consts::TAU * self.cadence * t;
        let pelvis = Point3::new(
            self.radius * around.sin(),
            anatomy.hip_height() - self.bob + self.bob * (2.0 * stride).cos().abs(),
            self.radius * around.cos(),
        );
        posture.set(Joint::Hip, pelvis);

        let side_axis = Unit::new_normalize(right);
        let facing_axis = Unit::new_normalize(facing);

        for (side, hip, knee, ankle, heel, big_toe, small_toe, phase) in [
            (
                -1.0,
                Joint::LeftHip,
                Joint::LeftKnee,
                Joint::LeftAnkle,
                Joint::LeftHeel,
                Joint::LeftBigToe,
                Joint::LeftSmallToe,
                0.0,
            ),
            (
                1.0,
                Joint::RightHip,
                Joint::RightKnee,
                Joint::RightAnkle,
                Joint::RightHeel,
                Joint::RightBigToe,
                Joint::RightSmallToe,
                std::f64::consts::PI,
            ),
        ] {
            let hip_point = pelvis + right * (anatomy.hip_half_width * side);
            let swing = self.swing * (stride + phase).sin();
            // Folded away from the straight leg by a non-negative amount, so
            // the shin never bends through the knee.
            let fold = self.knee * 0.5 * (1.0 - (stride + phase).cos());

            let thigh = Rotation3::from_axis_angle(&side_axis, swing) * (-up);
            let knee_point = hip_point + thigh * anatomy.thigh;
            let shank = Rotation3::from_axis_angle(&side_axis, swing - fold) * (-up);
            let ankle_point = knee_point + shank * anatomy.shin;

            posture.set(hip, hip_point);
            posture.set(knee, knee_point);
            posture.set(ankle, ankle_point);

            // The foot keeps its own plane rather than following the shin: a
            // walking foot is level for most of its stride, and the heel and
            // toe points are what the foot trackers are built from.
            let ground = Point3::new(ankle_point.x, 0.0, ankle_point.z);
            posture.set(
                heel,
                ground - facing * anatomy.heel_back + up * anatomy.heel_height,
            );
            posture.set(
                big_toe,
                ground + facing * anatomy.toe_forward - right * (anatomy.toe_half_width * side)
                    + up * anatomy.toe_height,
            );
            posture.set(
                small_toe,
                ground
                    + facing * (anatomy.toe_forward * 0.88)
                    + right * (anatomy.toe_half_width * side)
                    + up * anatomy.toe_height,
            );
        }

        let neck = pelvis + up * anatomy.spine;
        let head = neck + up * anatomy.neck;
        posture.set(Joint::Neck, neck);
        posture.set(Joint::Head, head);

        let r = anatomy.head_radius;
        posture.set(Joint::Nose, head + facing * (r * 0.95) - up * (r * 0.15));
        for (side, eye, ear) in [
            (-1.0, Joint::LeftEye, Joint::LeftEar),
            (1.0, Joint::RightEye, Joint::RightEar),
        ] {
            posture.set(
                eye,
                head + facing * (r * 0.80) + right * (r * 0.35 * side) + up * (r * 0.15),
            );
            posture.set(
                ear,
                head - facing * (r * 0.15) + right * (r * 0.95 * side) + up * (r * 0.05),
            );
        }

        for (side, shoulder, elbow, wrist, phase) in [
            (
                -1.0,
                Joint::LeftShoulder,
                Joint::LeftElbow,
                Joint::LeftWrist,
                std::f64::consts::PI,
            ),
            (
                1.0,
                Joint::RightShoulder,
                Joint::RightElbow,
                Joint::RightWrist,
                0.0,
            ),
        ] {
            let shoulder_point = neck + right * (anatomy.shoulder_half_width * side);
            let swing = self.arm * (stride + phase).sin();

            // Held away from the body so the arms do not pass through the
            // torso, which would give the silhouette a waist a person does not
            // have. Rotating about the walking direction tilts a hanging arm
            // towards the body's left for a positive angle, so the sign follows
            // the side the arm is on.
            let splay = Rotation3::from_axis_angle(&facing_axis, -self.splay * side);
            let upper = splay * (Rotation3::from_axis_angle(&side_axis, swing) * (-up));
            let elbow_point = shoulder_point + upper * anatomy.upper_arm;
            // The elbow carries a permanent slight bend, forwards, which is
            // where a relaxed arm sits and what keeps the wrist off the thigh.
            let lower = splay * (Rotation3::from_axis_angle(&side_axis, swing + 0.35) * (-up));
            let wrist_point = elbow_point + lower * anatomy.forearm;

            posture.set(shoulder, shoulder_point);
            posture.set(elbow, elbow_point);
            posture.set(wrist, wrist_point);
        }

        posture
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn walked(t: f64) -> Posture {
        Walk::default().posture(&Anatomy::default(), t)
    }

    #[test]
    fn a_posture_carries_every_canonical_joint() {
        let posture = walked(1.0);
        for joint in Joint::ALL {
            assert!(
                posture.get(joint).is_some(),
                "{joint:?} is missing from the simulated body"
            );
        }
        assert_eq!(posture.count(), Joint::ALL.len());
    }

    /// The whole point of forward kinematics here: whatever the walk does, the
    /// bones stay the length they were declared. An accuracy figure measured
    /// against a body whose legs changed length would mean nothing.
    #[test]
    fn the_bones_keep_their_length_through_the_walk() {
        let anatomy = Anatomy::default();
        let pairs = [
            (Joint::LeftHip, Joint::LeftKnee, anatomy.thigh),
            (Joint::LeftKnee, Joint::LeftAnkle, anatomy.shin),
            (Joint::RightHip, Joint::RightKnee, anatomy.thigh),
            (Joint::RightKnee, Joint::RightAnkle, anatomy.shin),
            (Joint::Hip, Joint::Neck, anatomy.spine),
            (Joint::Neck, Joint::Head, anatomy.neck),
            (Joint::LeftShoulder, Joint::LeftElbow, anatomy.upper_arm),
            (Joint::LeftElbow, Joint::LeftWrist, anatomy.forearm),
        ];

        for step in 0..200 {
            let posture = walked(step as f64 * 0.05);
            for (a, b, expected) in pairs {
                let length = (posture.get(b).unwrap() - posture.get(a).unwrap()).norm();
                assert!(
                    (length - expected).abs() < 1e-9,
                    "{a:?}-{b:?} is {length} m, not {expected} m"
                );
            }
        }
    }

    #[test]
    fn the_body_stays_the_right_way_up_and_above_the_floor() {
        let anatomy = Anatomy::default();
        for step in 0..200 {
            let posture = walked(step as f64 * 0.05);
            let head = posture.get(Joint::Head).unwrap();
            let hip = posture.get(Joint::Hip).unwrap();
            assert!(head.y > hip.y, "the head should be above the pelvis");
            for (joint, point) in posture.iter() {
                assert!(point.y > -1e-9, "{joint:?} went through the floor");
            }
            assert!(
                (head.y + anatomy.head_radius - anatomy.standing_height()).abs() < 0.1,
                "the figure should stand about its own height tall"
            );
        }
    }

    /// A knee that folds the wrong way puts the shin in front of the leg, and a
    /// pose model looking at it would be right to disagree with the truth.
    #[test]
    fn the_knee_only_folds_backwards() {
        for step in 0..200 {
            let posture = walked(step as f64 * 0.05);
            for (hip, knee, ankle) in [
                (Joint::LeftHip, Joint::LeftKnee, Joint::LeftAnkle),
                (Joint::RightHip, Joint::RightKnee, Joint::RightAnkle),
            ] {
                let thigh = posture.get(knee).unwrap() - posture.get(hip).unwrap();
                let shin = posture.get(ankle).unwrap() - posture.get(knee).unwrap();
                let bend = thigh.cross(&shin).dot(&posture.right);
                assert!(
                    bend <= 1e-9,
                    "the knee bent forwards by {bend} at step {step}"
                );
            }
        }
    }

    #[test]
    fn the_arms_hang_outside_the_shoulders() {
        for step in 0..200 {
            let posture = walked(step as f64 * 0.05);
            let hip = posture.get(Joint::Hip).unwrap();
            for (shoulder, wrist, side) in [
                (Joint::LeftShoulder, Joint::LeftWrist, -1.0),
                (Joint::RightShoulder, Joint::RightWrist, 1.0),
            ] {
                let across = |joint| (posture.get(joint).unwrap() - hip).dot(&posture.right) * side;
                assert!(
                    across(wrist) > across(shoulder),
                    "the arm should splay away from the body, not into it"
                );
            }
        }
    }

    /// The face keypoints are what tell a pose model which way the body is
    /// turned, so they have to be on the front of the head.
    #[test]
    fn the_face_is_on_the_front_of_the_head() {
        let posture = walked(2.3);
        let head = posture.get(Joint::Head).unwrap();
        let ahead = |joint| (posture.get(joint).unwrap() - head).dot(&posture.facing);

        assert!(ahead(Joint::Nose) > 0.05);
        assert!(ahead(Joint::LeftEye) > 0.0);
        assert!(ahead(Joint::RightEye) > 0.0);
        assert!(ahead(Joint::LeftEar) < 0.0);
    }

    #[test]
    fn the_left_side_is_on_the_left() {
        let posture = walked(0.7);
        let hip = posture.get(Joint::Hip).unwrap();
        let across = |joint| (posture.get(joint).unwrap() - hip).dot(&posture.right);

        assert!(across(Joint::LeftHip) < 0.0);
        assert!(across(Joint::RightHip) > 0.0);
        assert!(across(Joint::LeftShoulder) < 0.0);
        assert!(across(Joint::RightShoulder) > 0.0);
    }

    #[test]
    fn the_toes_are_ahead_of_the_heel() {
        for step in 0..100 {
            let posture = walked(step as f64 * 0.1);
            for (heel, toe) in [
                (Joint::LeftHeel, Joint::LeftBigToe),
                (Joint::RightHeel, Joint::RightBigToe),
            ] {
                let along =
                    (posture.get(toe).unwrap() - posture.get(heel).unwrap()).dot(&posture.facing);
                assert!(along > 0.1, "the foot should point forwards");
            }
        }
    }

    #[test]
    fn the_walk_goes_round_the_room() {
        let walk = Walk::default();
        let anatomy = Anatomy::default();
        let quarter = std::f64::consts::FRAC_PI_2 / walk.turn;

        let first = walk.posture(&anatomy, 0.0).get(Joint::Hip).unwrap();
        let later = walk.posture(&anatomy, quarter).get(Joint::Hip).unwrap();
        assert!(
            (first - later).norm() > walk.radius,
            "a quarter lap is a long way"
        );

        for step in 0..200 {
            let hip = walk
                .posture(&anatomy, step as f64 * 0.05)
                .get(Joint::Hip)
                .unwrap();
            assert!((hip.coords.xz().norm() - walk.radius).abs() < 1e-9);
        }
    }
}
