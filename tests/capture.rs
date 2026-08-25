//! Capture pipeline tests.
//!
//! The synthetic source exists so that the multi-camera paths can be exercised
//! without owning multiple cameras; these tests are the first thing it buys.

use std::time::{Duration, Instant};

use optra::capture::{CameraState, CaptureManager};
use optra::config::{CameraConfig, SourceConfig};
use optra::worker::Supervisor;

fn synthetic(id: &str, seat: u32, fps: u32) -> CameraConfig {
    CameraConfig {
        id: id.to_owned(),
        label: format!("Synthetic {seat}"),
        enabled: true,
        source: SourceConfig::Synthetic { seat },
        width: 320,
        height: 240,
        fps,
        ..CameraConfig::default()
    }
}

/// Waits until `condition` holds, or gives up after `timeout`.
fn wait_for(timeout: Duration, mut condition: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if condition() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    condition()
}

/// Counts frames over a measured interval, which is the rate this test means.
///
/// Deliberately not `stats.measured_fps`. That one is smoothed at a constant
/// chosen to make a panel readable, and a test that asserts on it is asserting
/// partly about that constant. Counting frames over a clock is the quantity
/// itself, and it costs a sleep this test can afford.
fn rate_of(channel: &optra::capture::CameraChannel, over: Duration) -> f32 {
    let before = channel.stats().captured;
    let start = Instant::now();
    std::thread::sleep(over);
    let after = channel.stats().captured;
    (after - before) as f32 / start.elapsed().as_secs_f32()
}

#[test]
fn synthetic_cameras_stream_independently() {
    let mut supervisor = Supervisor::new();
    let mut capture = CaptureManager::default();

    let configs = [
        synthetic("a", 0, 30),
        synthetic("b", 1, 30),
        synthetic("c", 2, 15),
    ];
    capture.start(&configs, &mut supervisor);

    let ready = wait_for(Duration::from_secs(5), || {
        capture
            .channels()
            .iter()
            .all(|channel| channel.stats().captured >= 10)
    });
    assert!(ready, "cameras did not reach 10 frames within 5 s");

    for channel in capture.channels() {
        let stats = channel.stats();
        let frame = channel.peek().expect("a published frame");

        assert_eq!(stats.state, CameraState::Running);
        assert_eq!(
            stats.errors, 0,
            "camera {} reported errors",
            channel.config.id
        );
        assert_eq!(frame.width, channel.config.width);
        assert_eq!(frame.height, channel.config.height);
        assert_eq!(
            frame.rgb.len(),
            frame.width as usize * frame.height as usize * 3
        );

        // The source paces itself, so it must never run *faster* than it was
        // asked to. There is no floor here to match it: how fast the scene
        // renders is a property of the build and the machine, and an
        // unoptimised build manages about ten frames a second whatever size the
        // frame is — the cost is per-frame geometry, not fill, so it does not
        // shrink with the image. A floor would make this a test of the compiler
        // settings. The pacing itself is tested below, at a rate any build can
        // reach.
        let requested = channel.config.fps as f32;
        let measured = rate_of(channel, Duration::from_millis(400));
        assert!(
            measured < requested * 1.4,
            "camera {} free-ran at {measured:.1} fps against a requested {requested:.0}",
            channel.config.id,
        );
    }

    // Cameras placed in different corners must not produce identical images,
    // otherwise triangulation would have nothing to work with later.
    let a = capture.channel("a").unwrap().peek().unwrap();
    let b = capture.channel("b").unwrap().peek().unwrap();
    assert_ne!(a.rgb, b.rgb, "two seats rendered the same view");

    capture.stop();
    supervisor.shutdown();
}

/// The pacing claim on its own, at a rate slow enough that every build can
/// render it.
///
/// This is what the frame rate assertion above was trying to say and could not:
/// a source that ignores the rate it was given and hands over frames as fast as
/// it can breaks every timing assumption downstream, and one that sleeps too
/// long starves the fusion clock. Asking for five frames a second leaves an
/// unoptimised build twice the time it needs, so what is left being measured is
/// the pacing.
#[test]
fn a_synthetic_camera_holds_the_rate_it_is_asked_for() {
    let mut supervisor = Supervisor::new();
    let mut capture = CaptureManager::default();

    capture.start(&[synthetic("a", 0, 5)], &mut supervisor);
    assert!(
        wait_for(Duration::from_secs(5), || capture.channels()[0]
            .stats()
            .captured
            >= 2),
        "no frames arrived within 5 s"
    );

    let measured = rate_of(&capture.channels()[0], Duration::from_millis(1600));
    assert!(
        (measured - 5.0).abs() < 1.5,
        "asked for 5 fps and got {measured:.1}"
    );

    capture.stop();
    supervisor.shutdown();
}

#[test]
fn taking_a_frame_marks_it_consumed() {
    let mut supervisor = Supervisor::new();
    let mut capture = CaptureManager::default();

    capture.start(&[synthetic("a", 0, 30)], &mut supervisor);
    assert!(
        wait_for(Duration::from_secs(5), || capture.channels()[0]
            .stats()
            .captured
            >= 1),
        "no frame arrived within 5 s"
    );

    let channel = &capture.channels()[0];
    let taken = channel.take().expect("an unread frame");
    assert!(
        channel.take().is_none() || channel.peek().unwrap().seq != taken.seq,
        "the same frame was handed out twice"
    );

    capture.stop();
    supervisor.shutdown();
}

#[test]
fn disabled_cameras_do_not_start() {
    let mut supervisor = Supervisor::new();
    let mut capture = CaptureManager::default();

    let mut disabled = synthetic("a", 0, 30);
    disabled.enabled = false;
    capture.start(&[disabled, synthetic("b", 1, 30)], &mut supervisor);

    assert_eq!(capture.channels().len(), 1);
    assert_eq!(capture.channels()[0].config.id, "b");

    capture.stop();
    supervisor.shutdown();
}
