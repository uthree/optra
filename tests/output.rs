//! The output stage from a reconstructed body to bytes on a socket.
//!
//! Everything downstream of here is somebody else's process, and there is no
//! way to test against it from a build machine. What can be tested is that the
//! bytes leaving Optra say what this project believes they should say — so the
//! sinks send to a real loopback socket and the test reads back what arrived
//! and decodes it, rather than inspecting the sink's own idea of what it did.

use std::net::UdpSocket;
use std::time::{Duration, Instant};

use nalgebra::{Point3, UnitQuaternion, Vector3};
use rosc::{OscPacket, OscType};

use optra::fusion::filter::{FilterOptions, Filtered, FilteredJoint};
use optra::models::Joint;
use optra::output::sink::{TrackerFrame, TrackerSink, assign};
use optra::output::vmt::Vmt;
use optra::output::vrchat::VrchatOsc;
use optra::output::{Posture, TrackerRole};

/// A consumer: a socket on a port the OS picked, so tests can run at once.
fn listener() -> (UdpSocket, String) {
    let socket = UdpSocket::bind("127.0.0.1:0").expect("a loopback socket");
    socket
        .set_read_timeout(Some(Duration::from_millis(500)))
        .expect("a read timeout");
    let address = socket.local_addr().expect("its own address").to_string();
    (socket, address)
}

/// Everything waiting on the socket, decoded.
fn drain(socket: &UdpSocket) -> Vec<(String, Vec<OscType>)> {
    let mut messages = Vec::new();
    let mut buffer = [0u8; 2048];

    socket
        .set_read_timeout(Some(Duration::from_millis(200)))
        .expect("a read timeout");

    while let Ok(read) = socket.recv(&mut buffer) {
        let (_, packet) = rosc::decoder::decode_udp(&buffer[..read]).expect("valid OSC");
        let OscPacket::Message(message) = packet else {
            panic!("a bundle arrived where a message was sent");
        };
        messages.push((message.addr, message.args));
    }

    messages
}

fn floats(args: &[OscType]) -> Vec<f32> {
    args.iter()
        .map(|arg| match arg {
            OscType::Float(value) => *value,
            other => panic!("{other:?} is not a float"),
        })
        .collect()
}

/// A body standing upright at the origin, facing -Z, one metre in front of the
/// play space centre and walking forwards at a metre a second.
fn walking(at: Instant) -> Filtered {
    let options = FilterOptions::default();
    let mut filtered = Filtered::empty(at, options.horizon);
    filtered.limit = options.max_prediction;

    let body = [
        (Joint::Neck, Point3::new(0.0, 1.45, -1.0)),
        (Joint::LeftShoulder, Point3::new(-0.18, 1.42, -1.0)),
        (Joint::RightShoulder, Point3::new(0.18, 1.42, -1.0)),
        (Joint::Hip, Point3::new(0.0, 0.95, -1.0)),
        (Joint::LeftHip, Point3::new(-0.10, 0.95, -1.0)),
        (Joint::RightHip, Point3::new(0.10, 0.95, -1.0)),
        (Joint::LeftKnee, Point3::new(-0.10, 0.52, -1.0)),
        (Joint::RightKnee, Point3::new(0.10, 0.52, -1.0)),
        (Joint::LeftAnkle, Point3::new(-0.10, 0.09, -1.0)),
        (Joint::RightAnkle, Point3::new(0.10, 0.09, -1.0)),
        (Joint::LeftHeel, Point3::new(-0.10, 0.05, -0.95)),
        (Joint::RightHeel, Point3::new(0.10, 0.05, -0.95)),
        (Joint::LeftBigToe, Point3::new(-0.10, 0.03, -1.15)),
        (Joint::RightBigToe, Point3::new(0.10, 0.03, -1.15)),
    ];

    for (joint, point) in body {
        filtered.set(
            joint,
            FilteredJoint {
                point,
                velocity: Vector3::new(0.0, 0.0, -1.0),
                predicted: point,
                lead: 0.05,
                sigma: 0.01,
                inferred: false,
            },
        );
    }

    filtered
}

/// The whole chain, in the configuration a first-time user would have: hips
/// and both feet, into VRChat.
#[test]
fn a_standing_body_arrives_at_vrchat_as_three_trackers_and_a_head() {
    let (socket, address) = listener();
    let roles = [
        TrackerRole::Hip,
        TrackerRole::LeftFoot,
        TrackerRole::RightFoot,
    ];
    let mut sink = VrchatOsc::open(&address, assign(&roles)).expect("a socket");

    let at = Instant::now();
    let posture = Posture::predicted(&walking(at), at, 1.0);
    let trackers: Vec<_> = roles
        .iter()
        .filter_map(|role| posture.derive(*role))
        .collect();
    assert_eq!(trackers.len(), 3, "a standing body places all three");

    sink.send(&TrackerFrame {
        at,
        lead: 0.06,
        trackers,
        lost: Vec::new(),
        head: Some(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(0.0, 1.6, -1.0),
            UnitQuaternion::identity(),
        )),
    })
    .expect("a loopback send");

    let messages = drain(&socket);
    let addresses: Vec<&str> = messages.iter().map(|(addr, _)| addr.as_str()).collect();

    // Head first, then one position and one rotation per tracker.
    assert_eq!(
        addresses,
        vec![
            "/tracking/trackers/head/position",
            "/tracking/trackers/head/rotation",
            "/tracking/trackers/1/position",
            "/tracking/trackers/1/rotation",
            "/tracking/trackers/2/position",
            "/tracking/trackers/2/rotation",
            "/tracking/trackers/3/position",
            "/tracking/trackers/3/rotation",
        ]
    );

    // The hip is tracker one, at hip height, and one metre out along Unity's
    // +Z where Optra had it one metre along -Z.
    let hip = floats(&messages[2].1);
    assert_eq!(hip.len(), 3);
    assert!(
        (hip[1] - 0.95).abs() < 0.02,
        "hip height came out {}",
        hip[1]
    );
    assert!(
        (hip[2] - 1.0).abs() < 0.1,
        "forward came out {}, so the handedness is wrong",
        hip[2]
    );

    // Standing square on, so every angle is near zero — and in degrees, which
    // is the units mistake that would otherwise pass every other check here.
    let rotation = floats(&messages[3].1);
    assert!(
        rotation.iter().all(|angle| angle.abs() < 5.0),
        "an upright body came out at {rotation:?}"
    );

    // Feet are at ankle height, not on the floor and not at the hip.
    for index in [4, 6] {
        let foot = floats(&messages[index].1);
        assert!(
            (0.02..0.25).contains(&foot[1]),
            "a foot came out at {} m",
            foot[1]
        );
    }
}

/// The same body into VMT, which wants the frame it is already in.
#[test]
fn the_same_body_arrives_at_vmt_unconverted() {
    let (socket, address) = listener();
    let roles = [TrackerRole::Hip, TrackerRole::LeftFoot];
    let mut sink = Vmt::open(&address, assign(&roles), None).expect("a socket");

    let at = Instant::now();
    let posture = Posture::predicted(&walking(at), at, 1.0);
    let trackers: Vec<_> = roles
        .iter()
        .filter_map(|role| posture.derive(*role))
        .collect();

    sink.send(&TrackerFrame {
        at,
        lead: 0.06,
        trackers,
        lost: Vec::new(),
        head: None,
    })
    .expect("a loopback send");

    let messages = drain(&socket);
    assert_eq!(messages.len(), 2, "one message per device");

    for (address, args) in &messages {
        assert_eq!(address, "/VMT/Room/Driver");
        assert_eq!(args.len(), 10);
        assert_eq!(args[1], OscType::Int(1), "the device should be enabled");
    }

    // The hip, still one metre along -Z: no mirror, unlike VRChat's.
    let hip = floats(&messages[0].1[2..]);
    assert!(
        hip[3] < -0.9,
        "VMT was handed a mirrored position: {}",
        hip[3]
    );
    // Quaternion, not Euler: an identity rotation is (0, 0, 0, 1).
    assert!((hip[7] - 1.0).abs() < 0.05, "w came out {}", hip[7]);
}

/// Sending faster than fusion reconstructs has to actually move the body, or
/// the extra sends are noise on the wire.
#[test]
fn consecutive_sends_of_one_frame_move_the_trackers() {
    let (socket, address) = listener();
    let roles = [TrackerRole::Hip];
    let mut sink = VrchatOsc::open(&address, assign(&roles)).expect("a socket");

    let at = Instant::now();
    let filtered = walking(at);

    let mut sent = Vec::new();
    for step in [0u64, 11, 22] {
        let now = at + Duration::from_millis(step);
        let posture = Posture::predicted(&filtered, now, 1.0);
        sink.send(&TrackerFrame {
            at: now,
            lead: 0.06,
            trackers: posture.derive(TrackerRole::Hip).into_iter().collect(),
            lost: Vec::new(),
            head: None,
        })
        .expect("a loopback send");

        let messages = drain(&socket);
        sent.push(floats(&messages[0].1)[2]);
    }

    // Walking towards Unity's +Z at a metre a second: eleven milliseconds is
    // eleven millimetres, and the same pose three times would be zero.
    for pair in sent.windows(2) {
        let moved = pair[1] - pair[0];
        assert!(
            (moved - 0.011).abs() < 1e-4,
            "consecutive sends moved the hip {moved} m, not 11 mm"
        );
    }
}

/// A tracker that has been gone long enough to call lost is switched off in
/// SteamVR rather than left standing where it was.
#[test]
fn a_lost_tracker_is_disabled_in_vmt() {
    let (socket, address) = listener();
    let roles = [TrackerRole::Hip, TrackerRole::LeftFoot];
    let mut sink = Vmt::open(&address, assign(&roles), None).expect("a socket");

    let at = Instant::now();
    let posture = Posture::predicted(&walking(at), at, 1.0);

    sink.send(&TrackerFrame {
        at,
        lead: 0.06,
        trackers: posture.derive(TrackerRole::Hip).into_iter().collect(),
        lost: vec![TrackerRole::LeftFoot],
        head: None,
    })
    .expect("a loopback send");

    let messages = drain(&socket);
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].1[0], OscType::Int(1));
    assert_eq!(messages[0].1[1], OscType::Int(1), "the hip is still there");
    assert_eq!(messages[1].1[0], OscType::Int(2));
    assert_eq!(
        messages[1].1[1],
        OscType::Int(0),
        "the lost foot should have been switched off"
    );
}

/// Quitting must not leave trackers behind, working as far as SteamVR can see.
#[test]
fn closing_the_vmt_sink_switches_every_device_off() {
    let (socket, address) = listener();
    let roles = [
        TrackerRole::Hip,
        TrackerRole::LeftFoot,
        TrackerRole::RightFoot,
    ];
    let mut sink = Vmt::open(&address, assign(&roles), None).expect("a socket");

    sink.close().expect("a loopback send");

    let messages = drain(&socket);
    assert_eq!(messages.len(), 3);
    for (index, (address, args)) in messages.iter().enumerate() {
        assert_eq!(address, "/VMT/Room/Driver");
        assert_eq!(args[0], OscType::Int(index as i32 + 1));
        assert_eq!(args[1], OscType::Int(0));
    }
}

/// A body the cameras have half of still sends what it has. This is the normal
/// state for the lower-body case Optra exists for: nothing above the waist is
/// ever reconstructed well, and the feet must not wait for it.
#[test]
fn a_body_with_no_upper_half_still_sends_its_feet() {
    let at = Instant::now();
    let mut filtered = walking(at);

    // Wipe everything the cameras would lose behind a desk.
    let full = Filtered::empty(at, filtered.horizon);
    let mut lower = full;
    lower.limit = filtered.limit;
    for joint in [
        Joint::LeftHip,
        Joint::RightHip,
        Joint::LeftKnee,
        Joint::RightKnee,
        Joint::LeftAnkle,
        Joint::RightAnkle,
        Joint::LeftHeel,
        Joint::RightHeel,
        Joint::LeftBigToe,
        Joint::RightBigToe,
    ] {
        lower.set(joint, filtered.get(joint).expect("a walking body has legs"));
    }
    filtered = lower;

    let posture = Posture::predicted(&filtered, at, 1.0);

    assert!(posture.derive(TrackerRole::LeftFoot).is_some());
    assert!(posture.derive(TrackerRole::RightFoot).is_some());
    assert!(posture.derive(TrackerRole::LeftKnee).is_some());
    // The hips need the spine to know which way up they are, and there is none.
    assert!(posture.derive(TrackerRole::Hip).is_none());
}

/// The lever a user reaches for when the trackers shake: with the cap at zero
/// nothing is extrapolated at all, so what goes out is exactly where the
/// cameras last put the body. Late, but if it is also steady then the trouble
/// is the prediction and not the reconstruction — which is a thing worth being
/// able to establish in ten seconds rather than by argument.
#[test]
fn capping_the_lead_at_zero_sends_the_body_where_it_was() {
    let at = Instant::now();
    let filtered = walking(at);

    let unpredicted = Posture::predicted(&filtered, at + Duration::from_millis(120), 0.0);
    let hip = unpredicted.point(Joint::Hip).unwrap();
    let measured = filtered.position(Joint::Hip).unwrap();

    assert!(
        (hip - measured).norm() < 1e-12,
        "the cap was zero and the hip still moved {:?}",
        hip - measured
    );

    // And the same instant with a cap on does move it, or the test above would
    // pass for the wrong reason.
    let predicted = Posture::predicted(&filtered, at + Duration::from_millis(120), 0.15);
    let moved = (predicted.point(Joint::Hip).unwrap() - measured).norm();
    assert!(moved > 0.05, "the uncapped prediction only moved {moved} m");
}
