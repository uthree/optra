//! Hardware tests. These need a real camera, so they are ignored by default.
//!
//! Run them with `cargo test --test webcam -- --ignored --nocapture`.

#![cfg(windows)]

use std::time::{Duration, Instant};

use optra::capture::source::{self, webcam};
use optra::capture::{CameraState, CaptureManager};
use optra::config::{CameraConfig, SourceConfig};
use optra::worker::Supervisor;

#[test]
#[ignore = "requires a connected camera"]
fn lists_connected_devices() {
    let devices = webcam::list_devices().expect("device enumeration");
    assert!(!devices.is_empty(), "no capture devices found");

    for device in &devices {
        println!("{} \n    {}", device.human_name(), device.misc());
    }
}

#[test]
#[ignore = "requires a connected camera"]
fn opens_the_first_device_and_streams() {
    let devices = webcam::list_devices().expect("device enumeration");
    let device = devices.first().expect("at least one capture device");

    let config = CameraConfig {
        id: "hw".to_owned(),
        label: device.human_name(),
        enabled: true,
        source: SourceConfig::Webcam {
            device_path: device.misc(),
            device_name: device.human_name(),
        },
        width: 1280,
        height: 720,
        fps: 30,
        ..CameraConfig::default()
    };

    let mut source = source::open(&config).expect("failed to open the camera");
    println!("negotiated: {}", source.negotiated());

    // Discard the first second: a camera ramps up its exposure and gain after
    // opening, and the rate during that period says nothing about steady state.
    let warmup = Instant::now();
    while warmup.elapsed() < Duration::from_secs(1) {
        source.next_frame().expect("failed to read a frame");
    }

    let started = Instant::now();
    let mut frames = 0;
    let mut decode_total = Duration::ZERO;
    while started.elapsed() < Duration::from_secs(3) {
        let frame = source.next_frame().expect("failed to read a frame");
        assert_eq!(
            frame.rgb.len(),
            frame.width as usize * frame.height as usize * 3
        );
        decode_total += frame.decode;
        frames += 1;
    }

    let elapsed = started.elapsed().as_secs_f32();
    println!(
        "{frames} frames in {elapsed:.1} s = {:.1} fps, decode {:.2} ms average",
        frames as f32 / elapsed,
        decode_total.as_secs_f32() * 1000.0 / frames as f32
    );
    assert!(frames > 10, "only {frames} frames in {elapsed:.1} s");
}

/// The capture thread has to work for a real device too, not just the
/// synthetic source: it is the path that has to survive COM initialization.
#[test]
#[ignore = "requires a connected camera"]
fn runs_a_real_camera_through_the_capture_thread() {
    let devices = webcam::list_devices().expect("device enumeration");
    let device = devices.first().expect("at least one capture device");

    let config = CameraConfig {
        id: "hw".to_owned(),
        label: device.human_name(),
        enabled: true,
        source: SourceConfig::Webcam {
            device_path: device.misc(),
            device_name: device.human_name(),
        },
        width: 1280,
        height: 720,
        fps: 30,
        ..CameraConfig::default()
    };

    let mut supervisor = Supervisor::new();
    let mut capture = CaptureManager::default();
    capture.start(&[config], &mut supervisor);

    let deadline = Instant::now() + Duration::from_secs(6);
    while Instant::now() < deadline && capture.channels()[0].stats().captured < 30 {
        std::thread::sleep(Duration::from_millis(50));
    }

    let stats = capture.channels()[0].stats();
    println!(
        "state {:?}, {} frames, {:.1} fps, decode {:.1} ms, negotiated {:?}",
        stats.state, stats.captured, stats.measured_fps, stats.decode_ms, stats.negotiated
    );
    if let Some(error) = &stats.last_error {
        println!("last error: {error}");
    }

    capture.stop();
    supervisor.shutdown();

    assert_eq!(stats.state, CameraState::Running);
    assert!(stats.captured >= 30, "only {} frames", stats.captured);
}

/// Diagnostic: what the device actually offers.
#[test]
#[ignore = "requires a connected camera"]
fn prints_supported_formats() {
    use nokhwa::Camera;
    use nokhwa::pixel_format::RgbFormat;
    use nokhwa::utils::{RequestedFormat, RequestedFormatType};

    let devices = webcam::list_devices().expect("device enumeration");
    let device = devices.first().expect("at least one capture device");

    let mut camera = Camera::new(
        device.index().clone(),
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate),
    )
    .expect("failed to open the camera");

    let mut formats = camera
        .compatible_camera_formats()
        .expect("failed to list formats");
    formats.sort_by_key(|f| {
        (
            f.format().to_string(),
            f.resolution().width(),
            f.resolution().height(),
            f.frame_rate(),
        )
    });
    for format in &formats {
        println!("{format}");
    }
    println!("{} formats", formats.len());
}

/// Diagnostic: what the device lets us control.
#[test]
#[ignore = "requires a connected camera"]
fn prints_camera_controls() {
    use nokhwa::Camera;
    use nokhwa::pixel_format::RgbFormat;
    use nokhwa::utils::{RequestedFormat, RequestedFormatType};

    let devices = webcam::list_devices().expect("device enumeration");
    let device = devices.first().expect("at least one capture device");

    let camera = Camera::new(
        device.index().clone(),
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate),
    )
    .expect("failed to open the camera");

    for control in camera.camera_controls().expect("failed to list controls") {
        println!(
            "{:?} | {:?} | flags {:?} | active {}",
            control.control(),
            control.description(),
            control.flag(),
            control.active()
        );
    }
}

/// The decisive question for capture quality: can exposure be pinned, and does
/// that actually restore the frame rate the format promises?
#[test]
#[ignore = "requires a connected camera"]
fn manual_exposure_restores_the_frame_rate() {
    use optra::capture::source::ControlSession;
    use optra::capture::source::controls::DeviceControls;
    use optra::config::ControlName;

    let devices = webcam::list_devices().expect("device enumeration");
    let device = devices.first().expect("at least one capture device");
    let path = device.misc();

    let controls = DeviceControls::open(&path).expect("failed to open the control session");
    for info in controls.list() {
        println!("{info:?}");
    }

    let exposure = controls
        .get(ControlName::Exposure)
        .expect("the device reports no exposure control");
    println!("exposure before: {exposure:?}");

    let config = CameraConfig {
        id: "hw".to_owned(),
        label: device.human_name(),
        enabled: true,
        source: SourceConfig::Webcam {
            device_path: path.clone(),
            device_name: device.human_name(),
        },
        width: 1280,
        height: 720,
        fps: 30,
        ..CameraConfig::default()
    };

    // 2^-6 s is about 1/64 s, comfortably shorter than a 30 fps frame period.
    let target = (-6).max(exposure.min);
    controls
        .set(ControlName::Exposure, target, false)
        .expect("failed to pin the exposure");
    println!("exposure after: {:?}", controls.get(ControlName::Exposure));

    let mut source = source::open(&config).expect("failed to open the camera");
    let warmup = Instant::now();
    while warmup.elapsed() < Duration::from_secs(1) {
        source.next_frame().expect("failed to read a frame");
    }

    let started = Instant::now();
    let mut frames = 0;
    while started.elapsed() < Duration::from_secs(3) {
        source.next_frame().expect("failed to read a frame");
        frames += 1;
    }
    let manual_fps = frames as f32 / started.elapsed().as_secs_f32();
    println!("manual exposure: {manual_fps:.1} fps");

    // Restore whatever the device was doing before, so the test is not
    // destructive to the user's camera settings.
    let _ = controls.set(ControlName::Exposure, exposure.value, exposure.auto);

    assert!(
        manual_fps > 25.0,
        "manual exposure gave only {manual_fps:.1} fps"
    );
}
