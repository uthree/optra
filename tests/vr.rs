//! Checks against a real SteamVR runtime.
//!
//! Ignored by default: they need SteamVR running with a headset. Run one with
//! `cargo test --test vr -- --ignored --nocapture <name>`.

use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use optra::config::VrConfig;
use optra::vr::{LinkState, Role, VrLink, api};
use optra::worker::Supervisor;

/// OpenVR is a process-wide singleton, so these tests must not overlap. The
/// harness runs tests in parallel by default, and a second connection in one
/// process is exactly what the runtime refuses.
static EXCLUSIVE: Mutex<()> = Mutex::new(());

fn exclusive() -> MutexGuard<'static, ()> {
    EXCLUSIVE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[test]
#[ignore = "requires a SteamVR installation"]
fn finds_the_runtime() {
    assert!(
        api::is_installed(),
        "no openvr_api.dll was found; set VR_OVERRIDE to the SteamVR directory if it is somewhere unusual"
    );
}

/// The one that matters. A wrongly transcribed function table would show up
/// here as a crash or as nonsense poses rather than as a compile error.
#[test]
#[ignore = "requires SteamVR to be running"]
fn reads_poses_from_a_running_runtime() {
    let _guard = exclusive();
    let runtime = api::Runtime::connect().expect("SteamVR should be running");
    println!("runtime: {}", runtime.path().display());

    let poses = runtime.poses(0.0);
    let mut connected = 0;

    for (index, pose) in poses.iter().enumerate() {
        if !pose.device_is_connected {
            continue;
        }
        connected += 1;

        let index = index as u32;
        let m = &pose.device_to_absolute_tracking.m;
        println!(
            "{index:>2}  class {}  role {:>2}  valid {}  at ({:.2}, {:.2}, {:.2})  {} / {}",
            runtime.device_class(index),
            runtime.controller_role(index),
            pose.pose_is_valid,
            m[0][3],
            m[1][3],
            m[2][3],
            runtime.model(index),
            runtime.serial(index),
        );

        // A pose the runtime calls valid must be a rigid transform. If the
        // table were misread this is where the numbers stop making sense.
        if pose.pose_is_valid {
            for row in m.iter() {
                let length = (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt();
                assert!(
                    (length - 1.0).abs() < 1e-3,
                    "device {index} has a row of length {length}, which is not a rotation"
                );
                assert!(
                    row[3].abs() < 50.0,
                    "device {index} is {} m from the origin",
                    row[3]
                );
            }
        }
    }

    assert!(connected > 0, "SteamVR reported no connected devices");
}

/// The whole link as the application runs it, including the history the
/// calibration recorder will read from.
#[test]
#[ignore = "requires SteamVR to be running"]
fn the_link_thread_records_a_usable_history() {
    let _guard = exclusive();

    // The application raises this at startup. Without it every fixed-rate loop
    // runs at the 64 Hz the Windows scheduler wakes threads at, and the test
    // would be measuring the scheduler rather than the link.
    let _timer = optra::worker::timing::TimerResolution::request();
    let mut supervisor = Supervisor::new();
    let mut link = VrLink::default();

    link.start(&VrConfig::default(), &mut supervisor);
    let channel = link
        .channel()
        .expect("the link should have started")
        .clone();

    // Give it a moment to connect and fill some history.
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline && channel.stats().state != LinkState::Connected {
        std::thread::sleep(Duration::from_millis(100));
    }
    std::thread::sleep(Duration::from_secs(1));

    let stats = channel.stats();
    println!(
        "state {:?}, {} devices, {:.1} Hz, {} samples",
        stats.state, stats.devices, stats.measured_hz, stats.samples
    );
    assert_eq!(stats.state, LinkState::Connected);
    assert!(
        stats.samples > 50,
        "only {} samples in a second; the link thread stalled",
        stats.samples
    );
    assert!(
        stats.measured_hz > 30.0,
        "sampling ran at only {:.1} Hz",
        stats.measured_hz
    );

    let snapshot = channel.latest().expect("a snapshot should have been taken");
    for device in &snapshot.devices {
        let p = device.pose.translation.vector;
        println!(
            "{:>10}  ({:.2}, {:.2}, {:.2})  tracking {}  {}",
            device.role.label(),
            p.x,
            p.y,
            p.z,
            device.tracking,
            device.model
        );
    }

    assert!(
        channel.is_tracking(Role::Head),
        "the headset should be tracking"
    );

    // A pose from half a second ago is what pairing a camera frame with the
    // headset actually asks for.
    let past = Instant::now() - Duration::from_millis(500);
    let pose = channel
        .pose_at(Role::Head, past)
        .expect("half a second ago is inside the history window");
    println!("head half a second ago: {:?}", pose.translation.vector);

    // Far outside the window, nothing should be invented.
    assert!(
        channel
            .pose_at(Role::Head, Instant::now() + Duration::from_secs(30))
            .is_none()
    );

    link.stop();
    supervisor.shutdown();
}
