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

        // The source paces itself, so the measured rate should land near the
        // requested one rather than free-running.
        let requested = channel.config.fps as f32;
        assert!(
            stats.measured_fps > requested * 0.6 && stats.measured_fps < requested * 1.4,
            "camera {} measured {:.1} fps against a requested {requested:.0}",
            channel.config.id,
            stats.measured_fps
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
