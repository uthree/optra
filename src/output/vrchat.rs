//! VRChat's OSC tracker input.
//!
//! VRChat accepts up to eight trackers on `/tracking/trackers/{1..8}/position`
//! and `/rotation`, plus a head reference on `/tracking/trackers/head/...`.
//! The head is not optional in practice: VRChat places the trackers relative to
//! it, so sending trackers without it puts the body somewhere arbitrary. It is
//! the same headset pose Optra calibrated the cameras against, which is why
//! sending it costs nothing and makes the two agree by construction.
//!
//! Which tracker is which is not part of the protocol. VRChat works it out from
//! where they sit on the body during its own in-game calibration, so the
//! indices only have to stay still between calibrations.

use anyhow::Result;
use nalgebra::{Isometry3, UnitQuaternion, Vector3};

use super::osc::OscSender;
use super::pose::TrackerRole;
use super::sink::{TrackerFrame, TrackerSink};

/// Where VRChat listens unless it has been told otherwise.
pub const DEFAULT_TARGET: &str = "127.0.0.1:9000";

pub struct VrchatOsc {
    osc: OscSender,
    /// Roles in index order, so a send does not have to sort anything.
    indices: Vec<(u8, TrackerRole)>,
}

impl VrchatOsc {
    pub fn open(target: &str, indices: Vec<(u8, TrackerRole)>) -> Result<Self> {
        Ok(Self {
            osc: OscSender::open(target)?,
            indices,
        })
    }

    fn send_pose(&mut self, address: &str, pose: &Isometry3<f64>) -> Result<()> {
        self.send_triple(&format!("{address}/position"), unity_position(pose))?;
        self.send_triple(&format!("{address}/rotation"), unity_euler(&pose.rotation))
    }

    fn send_triple(&mut self, address: &str, value: Vector3<f64>) -> Result<()> {
        self.osc.send_triple(address, value.x, value.y, value.z)
    }
}

impl TrackerSink for VrchatOsc {
    fn name(&self) -> &str {
        "VRChat OSC"
    }

    fn target(&self) -> String {
        self.osc.target().to_owned()
    }

    fn send(&mut self, frame: &TrackerFrame) -> Result<()> {
        // The head first: it is the frame the rest are read against, and a
        // consumer that applies trackers against last frame's head has every
        // tracker wrong by however far the user's head moved.
        if let Some(head) = frame.head {
            self.send_pose("/tracking/trackers/head", &head)?;
        }

        for (index, role) in self.indices.clone() {
            let Some(tracker) = frame.trackers.iter().find(|tracker| tracker.role == role) else {
                // Nothing is sent for a tracker that could not be derived.
                // VRChat holds the last pose, which is right for the single
                // frame an occlusion lasts and is why `lost` exists for the
                // ones that last longer — there is no way to say "gone" here,
                // so the panel says it instead.
                continue;
            };

            self.send_pose(&format!("/tracking/trackers/{index}"), &tracker.pose)?;
        }

        Ok(())
    }
}

/// Optra's world position in Unity's left-handed frame.
///
/// Both are +Y up and metres; the difference is the handedness, and mirroring
/// Z is the conversion. OpenVR's standing universe has -Z forward, Unity has
/// +Z forward, and the same mirror covers both.
fn unity_position(pose: &Isometry3<f64>) -> Vector3<f64> {
    let translation = pose.translation.vector;
    Vector3::new(translation.x, translation.y, -translation.z)
}

/// Optra's world rotation as Unity Euler angles, in degrees.
///
/// Two conversions in one. First the handedness: mirroring an axis conjugates
/// a rotation, and mirroring Z leaves rotation about Z alone while reversing
/// rotation about X and Y — so the quaternion's x and y components negate and
/// its z does not.
///
/// Then Unity's Euler convention, which is intrinsic Z, then X, then Y —
/// composing to `Ry * Rx * Rz` when a matrix is applied to a column vector.
/// This is not a convention anything can be talked out of: it is what VRChat
/// hands to `Transform.eulerAngles`, so it is what has to come out of here.
fn unity_euler(rotation: &UnitQuaternion<f64>) -> Vector3<f64> {
    let mirrored = UnitQuaternion::new_unchecked(nalgebra::Quaternion::new(
        rotation.w,
        -rotation.i,
        -rotation.j,
        rotation.k,
    ));
    let m = mirrored.to_rotation_matrix();

    // R[1][2] is -sin(x), and it is the only entry that isolates one angle.
    let sin_x = -m[(1, 2)];
    let x = sin_x.clamp(-1.0, 1.0).asin();

    // Gimbal lock: with the X rotation at a right angle, cos(x) is zero and Y
    // and Z become the same axis. Any split of the sum is as correct as any
    // other, so it all goes to Y and Z is pinned at zero — an arbitrary answer
    // chosen deliberately beats two arbitrary answers that jitter against each
    // other from frame to frame.
    let (y, z) = if sin_x.abs() > 0.9999 {
        (f64::atan2(-m[(2, 0)], m[(0, 0)]), 0.0)
    } else {
        (
            f64::atan2(m[(0, 2)], m[(2, 2)]),
            f64::atan2(m[(1, 0)], m[(1, 1)]),
        )
    };

    Vector3::new(x.to_degrees(), y.to_degrees(), z.to_degrees())
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_PI_2;

    use nalgebra::{Point3, Translation3};

    use super::*;

    fn about(actual: f64, expected: f64) -> bool {
        (actual - expected).abs() < 1e-6
    }

    /// The one conversion a user would notice immediately: walking away from
    /// the play space origin must not walk the avatar towards it.
    #[test]
    fn forward_in_the_room_is_forward_in_unity() {
        // Optra's world: -Z is forward. Unity's: +Z is.
        let pose = Isometry3::from_parts(
            Translation3::new(0.5, 1.0, -2.0),
            UnitQuaternion::identity(),
        );
        let position = unity_position(&pose);
        assert!(about(position.x, 0.5));
        assert!(about(position.y, 1.0));
        assert!(about(position.z, 2.0), "z came out {}", position.z);
    }

    #[test]
    fn an_upright_tracker_has_no_rotation() {
        let euler = unity_euler(&UnitQuaternion::identity());
        assert!(euler.norm() < 1e-9, "identity came out as {euler:?}");
    }

    /// Yaw is the angle that matters most and the one a handedness mistake
    /// silently reverses. A body turning to its own left turns the same way in
    /// both frames; the *sign* of the angle about the up axis flips, because
    /// the frames disagree about which way is positive.
    #[test]
    fn a_quarter_turn_left_is_a_quarter_turn_left() {
        // Turning left in a right-handed +Y-up frame is positive about Y.
        let left = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), FRAC_PI_2);
        let euler = unity_euler(&left);

        assert!(about(euler.x, 0.0), "pitch appeared: {euler:?}");
        assert!(about(euler.z, 0.0), "roll appeared: {euler:?}");
        assert!(about(euler.y, -90.0), "yaw came out {}", euler.y);
    }

    /// Leaning forward is leaning forward. Pitch is about the left-right axis,
    /// which the Z mirror does reverse.
    #[test]
    fn leaning_forward_reads_as_leaning_forward() {
        // Positive about +X in a right-handed frame tips the top backwards.
        let back = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.3);
        assert!(about(unity_euler(&back).x, -0.3f64.to_degrees()));
    }

    /// Every rotation must survive the trip, whatever axes it mixes. A
    /// convention that only works on the three cardinal turns is a convention
    /// that fails the moment somebody bends a knee.
    #[test]
    fn any_rotation_round_trips_through_the_euler_convention() {
        for (axis, angle) in [
            (Vector3::new(1.0, 2.0, 3.0), 0.7),
            (Vector3::new(-2.0, 0.5, 1.0), 2.4),
            (Vector3::new(0.1, -1.0, 0.2), -1.9),
        ] {
            let rotation =
                UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(axis), angle);
            let euler = unity_euler(&rotation);

            // Rebuild it the way Unity would: intrinsic Z, then X, then Y.
            let rebuilt = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), euler.y.to_radians())
                * UnitQuaternion::from_axis_angle(&Vector3::x_axis(), euler.x.to_radians())
                * UnitQuaternion::from_axis_angle(&Vector3::z_axis(), euler.z.to_radians());

            // Against the mirrored rotation, which is what the Euler angles
            // describe — not against the original.
            let mirrored = UnitQuaternion::new_unchecked(nalgebra::Quaternion::new(
                rotation.w,
                -rotation.i,
                -rotation.j,
                rotation.k,
            ));

            let error = rebuilt.angle_to(&mirrored).to_degrees();
            assert!(
                error < 1e-6,
                "{axis:?} at {angle} came back {error} degrees off"
            );
        }
    }

    /// Straight up is where the Euler extraction divides by a cosine that has
    /// gone to zero. It has to come out of that with an answer, not a NaN.
    #[test]
    fn looking_straight_up_does_not_produce_nonsense() {
        let up = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), FRAC_PI_2);
        let euler = unity_euler(&up);
        assert!(euler.iter().all(|angle| angle.is_finite()), "{euler:?}");
        assert!(about(euler.z, 0.0), "roll should be pinned, got {euler:?}");
    }

    /// The socket work, without a consumer: a sink pointed at a port nobody is
    /// listening on must still send. UDP has no one to complain to, and a sink
    /// that treated that as an error would report a fault every frame for the
    /// entirely normal case of VRChat not being open yet.
    #[test]
    fn sending_into_the_void_is_not_an_error() {
        let mut sink = VrchatOsc::open(
            "127.0.0.1:39999",
            super::super::sink::assign(&[TrackerRole::Hip, TrackerRole::LeftFoot]),
        )
        .expect("a loopback socket");

        let frame = TrackerFrame {
            at: std::time::Instant::now(),
            lead: 0.06,
            trackers: vec![super::super::pose::TrackerPose {
                role: TrackerRole::Hip,
                pose: Isometry3::from_parts(
                    Translation3::from(Point3::new(0.0, 0.95, 0.0).coords),
                    UnitQuaternion::identity(),
                ),
                sigma: 0.01,
                inferred: false,
            }],
            lost: vec![TrackerRole::LeftFoot],
            head: Some(Isometry3::identity()),
        };

        sink.send(&frame).expect("UDP does not wait to be heard");
    }
}
