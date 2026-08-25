//! VirtualMotionTracker: virtual SteamVR devices, driven over OSC.
//!
//! VMT is a SteamVR driver that registers virtual trackers and takes their
//! poses from OSC messages. It is the route to everything that reads SteamVR
//! rather than VRChat's own OSC — other social platforms, recording tools,
//! and VRChat itself if the user would rather have real SteamVR devices.
//!
//! Poses go out in driver coordinates, which is Optra's world frame unchanged:
//! right-handed, +Y up, metres. No conversion, unlike the VRChat sink — which
//! is worth saying out loud, because a bug that only shows up in one of the two
//! backends is a bug in a conversion, and this file has none to blame.
//!
//! What it does have is the room matrix. VMT places devices in the runtime's
//! *raw* space, and everything here is in the standing universe — the two
//! differ by whatever SteamVR's room setup did, which is exactly the floor
//! height and play-space centre a user would notice being wrong. VMT keeps that
//! transform as a setting of its own; Optra can read the true one from OpenVR
//! and send it, which is one fewer thing to have configured correctly
//! elsewhere.

use std::net::{ToSocketAddrs, UdpSocket};

use anyhow::{Context, Result};
use nalgebra::Isometry3;
use rosc::{OscMessage, OscPacket, OscType, encoder};

use super::pose::TrackerRole;
use super::sink::{TrackerFrame, TrackerSink};

/// Where VMT listens unless it has been told otherwise.
pub const DEFAULT_TARGET: &str = "127.0.0.1:39570";

/// Room space, in driver coordinates: the address that takes a right-handed
/// pose and puts it through VMT's room matrix.
const ROOM_DRIVER: &str = "/VMT/Room/Driver";

/// Sets the room matrix for this run only, leaving whatever the user has saved
/// in VMT alone. Optra has no business making a permanent change to another
/// application's configuration.
const SET_ROOM_MATRIX: &str = "/VMT/SetRoomMatrix/Temporary";

pub struct Vmt {
    socket: UdpSocket,
    target: String,
    indices: Vec<(u8, TrackerRole)>,
    buffer: Vec<u8>,
}

impl Vmt {
    /// Opens a connection and, if a room transform is given, tells VMT about
    /// it.
    ///
    /// `standing_to_raw` comes from OpenVR's own room setup. Passing `None`
    /// leaves VMT using whatever it already has, which is right for a user who
    /// has configured it themselves and wrong-looking by the height of their
    /// floor for one who has not.
    pub fn open(
        target: &str,
        indices: Vec<(u8, TrackerRole)>,
        standing_to_raw: Option<Isometry3<f64>>,
    ) -> Result<Self> {
        let socket = UdpSocket::bind("0.0.0.0:0").context("could not open a UDP socket")?;
        let resolved = target
            .to_socket_addrs()
            .with_context(|| format!("{target} is not an address"))?
            .next()
            .with_context(|| format!("{target} resolved to nothing"))?;
        socket
            .connect(resolved)
            .with_context(|| format!("could not point a socket at {target}"))?;

        let mut sink = Self {
            socket,
            target: target.to_owned(),
            indices,
            buffer: Vec::with_capacity(192),
        };

        if let Some(room) = standing_to_raw {
            sink.set_room_matrix(&room)?;
        }

        Ok(sink)
    }

    /// Hands VMT a row-major 3x4, which is how OpenVR writes a transform and
    /// how VMT reads one.
    fn set_room_matrix(&mut self, room: &Isometry3<f64>) -> Result<()> {
        let rotation = room.rotation.to_rotation_matrix();
        let translation = room.translation.vector;

        let mut args = Vec::with_capacity(12);
        for row in 0..3 {
            for column in 0..3 {
                args.push(OscType::Float(rotation[(row, column)] as f32));
            }
            args.push(OscType::Float(translation[row] as f32));
        }

        self.emit(SET_ROOM_MATRIX, args)
    }

    fn emit(&mut self, address: &str, args: Vec<OscType>) -> Result<()> {
        let packet = OscPacket::Message(OscMessage {
            addr: address.to_owned(),
            args,
        });

        self.buffer.clear();
        // Encoding into a `Vec` cannot fail; the error type is `Infallible`.
        encoder::encode_into(&packet, &mut self.buffer).ok();
        self.socket
            .send(&self.buffer)
            .with_context(|| format!("could not send {address}"))?;
        Ok(())
    }

    /// One device, at an offset from now.
    ///
    /// `enable` is what makes a device appear in SteamVR at all, so switching
    /// it off is how a tracker Optra has lost stops pretending: the device
    /// disappears rather than standing wherever it was last seen. That is the
    /// honest report, and it is also the one a user can act on.
    fn device(&mut self, index: u8, enable: bool, lead: f64, pose: &Isometry3<f64>) -> Result<()> {
        let position = pose.translation.vector;
        let rotation = pose.rotation;

        self.emit(
            ROOM_DRIVER,
            vec![
                OscType::Int(index as i32),
                OscType::Int(enable as i32),
                // Positive is the future: the poses have already been predicted
                // forward, and SteamVR has to be told so or it predicts them
                // again on top.
                OscType::Float(lead as f32),
                OscType::Float(position.x as f32),
                OscType::Float(position.y as f32),
                OscType::Float(position.z as f32),
                OscType::Float(rotation.i as f32),
                OscType::Float(rotation.j as f32),
                OscType::Float(rotation.k as f32),
                OscType::Float(rotation.w as f32),
            ],
        )
    }
}

impl TrackerSink for Vmt {
    fn name(&self) -> &str {
        "VMT"
    }

    fn target(&self) -> String {
        self.target.clone()
    }

    fn send(&mut self, frame: &TrackerFrame) -> Result<()> {
        for (index, role) in self.indices.clone() {
            match frame.trackers.iter().find(|tracker| tracker.role == role) {
                Some(tracker) => self.device(index, true, frame.lead, &tracker.pose)?,
                // A tracker missing for a single frame holds still: sending
                // nothing leaves VMT with the last pose, which is what a
                // momentary occlusion should look like. One that has been gone
                // long enough to call lost is switched off instead.
                None if frame.lost.contains(&role) => {
                    self.device(index, false, 0.0, &Isometry3::identity())?
                }
                None => {}
            }
        }

        Ok(())
    }

    /// Switches every device off on the way out.
    ///
    /// Without this the trackers stay in SteamVR, frozen at their last pose,
    /// until VMT is restarted — and a user who closed Optra to fix something
    /// would be looking at trackers that appear to still be working.
    fn close(&mut self) -> Result<()> {
        for (index, _) in self.indices.clone() {
            self.device(index, false, 0.0, &Isometry3::identity())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use nalgebra::{Translation3, UnitQuaternion, Vector3};

    use super::super::pose::TrackerPose;
    use super::super::sink::assign;
    use super::*;

    fn sink() -> Vmt {
        Vmt::open(
            "127.0.0.1:39998",
            assign(&[TrackerRole::Hip, TrackerRole::LeftFoot]),
            None,
        )
        .expect("a loopback socket")
    }

    fn frame(trackers: Vec<TrackerPose>, lost: Vec<TrackerRole>) -> TrackerFrame {
        TrackerFrame {
            at: Instant::now(),
            lead: 0.06,
            trackers,
            lost,
            head: None,
        }
    }

    fn tracker(role: TrackerRole) -> TrackerPose {
        TrackerPose {
            role,
            pose: Isometry3::from_parts(
                Translation3::new(0.0, 0.95, -1.0),
                UnitQuaternion::identity(),
            ),
            sigma: 0.01,
            inferred: false,
        }
    }

    #[test]
    fn sending_into_the_void_is_not_an_error() {
        let mut sink = sink();
        sink.send(&frame(
            vec![tracker(TrackerRole::Hip)],
            vec![TrackerRole::LeftFoot],
        ))
        .expect("UDP does not wait to be heard");
    }

    #[test]
    fn closing_does_not_fail_with_nothing_listening() {
        let mut sink = sink();
        sink.close().expect("the goodbye is best effort");
    }

    /// The room matrix is a 3x4 in the same layout OpenVR uses, and a
    /// transposed rotation is a mistake that looks almost right — the floor
    /// height would be correct and every turn would go the wrong way.
    #[test]
    fn the_room_matrix_goes_out_row_major() {
        let room = Isometry3::from_parts(
            Translation3::new(0.1, -1.2, 0.3),
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.4),
        );

        let expected = room.rotation.to_rotation_matrix();
        let mut sink = sink();
        sink.set_room_matrix(&room).expect("a loopback socket");

        // Decode what actually went on the wire rather than trusting the
        // builder: the layout is the whole point of the test.
        let packet = rosc::decoder::decode_udp(&sink.buffer)
            .expect("we just encoded this")
            .1;
        let OscPacket::Message(message) = packet else {
            panic!("a bundle came out of a message");
        };

        assert_eq!(message.addr, SET_ROOM_MATRIX);
        assert_eq!(message.args.len(), 12);

        let floats: Vec<f32> = message
            .args
            .iter()
            .map(|arg| match arg {
                OscType::Float(value) => *value,
                other => panic!("{other:?} is not a float"),
            })
            .collect();

        for row in 0..3 {
            for column in 0..3 {
                let sent = floats[row * 4 + column];
                let want = expected[(row, column)] as f32;
                assert!(
                    (sent - want).abs() < 1e-6,
                    "row {row} column {column} came out {sent}, not {want}"
                );
            }
        }
        assert!((floats[3] - 0.1).abs() < 1e-6);
        assert!((floats[7] + 1.2).abs() < 1e-6);
        assert!((floats[11] - 0.3).abs() < 1e-6);
    }

    /// Driver coordinates are Optra's world frame, so a pose must go out
    /// untouched. This is the sink where a conversion would be the bug.
    #[test]
    fn a_pose_goes_out_unconverted() {
        let mut sink = sink();
        sink.send(&frame(vec![tracker(TrackerRole::Hip)], Vec::new()))
            .expect("a loopback socket");

        let OscPacket::Message(message) = rosc::decoder::decode_udp(&sink.buffer)
            .expect("we just encoded this")
            .1
        else {
            panic!("a bundle came out of a message");
        };

        assert_eq!(message.addr, ROOM_DRIVER);
        assert_eq!(message.args[0], OscType::Int(1));
        assert_eq!(message.args[1], OscType::Int(1));
        assert_eq!(message.args[3], OscType::Float(0.0));
        assert_eq!(message.args[4], OscType::Float(0.95));
        assert_eq!(
            message.args[5],
            OscType::Float(-1.0),
            "forward was mirrored, which VMT does not want"
        );
    }
}
