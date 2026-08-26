//! The startup self-check.
//!
//! Each of these is a setup that used to leave the Tracking panel silent with
//! the reason on another panel. What is asserted is not that a check fires but
//! that it names the thing the user has to go and touch.

use nalgebra::{Point3, Vector3};

use optra::calib::RoomCalibration;
use optra::calib::solve::CameraCalibration;
use optra::config::{CameraConfig, Config, SourceConfig};
use optra::geometry::camera::{Camera, Intrinsics};
use optra::geometry::lens::Lens;
use optra::models::manifest::Manifest;
use optra::startup::{self, Verdict};

/// A configured webcam, identified the way a real one is.
fn webcam(index: usize) -> CameraConfig {
    CameraConfig {
        id: format!("cam{index}"),
        label: format!("Camera {index}"),
        enabled: true,
        source: SourceConfig::Webcam {
            device_path: format!("\\\\?\\usb#vid_046d&pid_0825#{index}"),
            device_name: format!("Webcam {index}"),
        },
        ..CameraConfig::default()
    }
}

fn config_with(count: usize) -> Config {
    Config {
        cameras: (0..count).map(webcam).collect(),
        ..Config::default()
    }
}

fn paths(config: &Config) -> Vec<String> {
    config
        .cameras
        .iter()
        .filter_map(|camera| match &camera.source {
            SourceConfig::Webcam { device_path, .. } => Some(device_path.clone()),
            _ => None,
        })
        .collect()
}

/// A profile covering the first `count` cameras.
fn solved(count: usize) -> RoomCalibration {
    RoomCalibration {
        cameras: (0..count)
            .map(|index| CameraCalibration {
                id: format!("cam{index}"),
                camera: Camera::look_at(
                    Intrinsics::from_fov(1280, 720, 70f64.to_radians()),
                    Lens::default(),
                    Point3::new(index as f64, 2.4, -1.8),
                    Point3::new(0.0, 1.0, 0.0),
                    Vector3::y(),
                ),
                rms: 0.001,
                sightings: 400,
                coverage: 0.6,
                spread: 0.4,
                latency: None,
                range: 2.4,
                feet: 0.8,
            })
            .collect(),
        rigs: Vec::new(),
        rms: 0.001,
        rejected: 0,
        used: 400,
        precision: Some(0.01),
        floor_precision: None,
        solved_at: "2026-08-26T12:00:00+09:00".to_owned(),
    }
}

#[test]
fn a_camera_that_was_unplugged_is_named() {
    let config = config_with(3);
    let mut present = paths(&config);
    present.remove(1);

    let check = startup::cameras(&config, Some(&present));

    // Two cameras can still place a joint, so tracking runs — worse than it
    // did, which is the part nothing else in the application would mention.
    assert_eq!(check.verdict, Verdict::Warning);
    assert!(
        check.detail.contains("Webcam 1"),
        "the device name is the only thing a user can recognise: {}",
        check.detail
    );
}

#[test]
fn losing_the_second_camera_stops_tracking() {
    let config = config_with(2);
    let present = paths(&config)[..1].to_vec();

    let check = startup::cameras(&config, Some(&present));

    assert_eq!(check.verdict, Verdict::Blocked);
    assert!(check.detail.contains("Webcam 1"), "{}", check.detail);
}

#[test]
fn a_platform_that_cannot_be_asked_is_not_every_camera_missing() {
    let config = config_with(3);

    // `None` means the enumeration failed. Treating it as an empty list would
    // report three attached cameras as three unplugged ones, which is the
    // worst possible answer: it sends the user to check cables that are fine.
    let check = startup::cameras(&config, None);

    assert_eq!(check.verdict, Verdict::Warning);
    assert!(!check.detail.contains("Webcam"), "{}", check.detail);
}

#[test]
fn a_synthetic_camera_is_never_missing() {
    let config = Config {
        cameras: (0..2)
            .map(|index| CameraConfig {
                id: format!("cam{index}"),
                enabled: true,
                source: SourceConfig::Synthetic { seat: index as u32 },
                ..CameraConfig::default()
            })
            .collect(),
        ..Config::default()
    };

    assert_eq!(
        startup::cameras(&config, Some(&[])).verdict,
        Verdict::Ready,
        "nothing generated can be unplugged"
    );
}

#[test]
fn one_camera_cannot_place_anything() {
    let config = config_with(1);
    let check = startup::cameras(&config, Some(&paths(&config)));

    assert_eq!(check.verdict, Verdict::Blocked);
    assert!(
        check.fix.is_some(),
        "a block has to say what to do about it"
    );
}

#[test]
fn a_model_that_was_never_downloaded_blocks() {
    let config = config_with(2);
    let catalogue = Manifest::load().expect("the builtin catalogue always parses");

    // Nothing installed, which is the state of a fresh machine.
    let check = startup::models(&config, Some(&catalogue), &|_| false);

    assert_eq!(check.verdict, Verdict::Blocked);
    assert!(
        check.detail.contains(&config.inference.pose_model),
        "the model to install has to be named: {}",
        check.detail
    );
}

#[test]
fn a_camera_can_name_a_model_that_does_not_exist() {
    let mut config = config_with(2);
    config.cameras[1].pose_model = Some("a-model-nobody-published".to_owned());
    let catalogue = Manifest::load().unwrap();

    // Everything in the catalogue is installed, so the only thing left to
    // report is the one entry that is not in the catalogue at all.
    let check = startup::models(&config, Some(&catalogue), &|_| true);

    assert_eq!(check.verdict, Verdict::Blocked);
    assert!(
        check.detail.contains("a-model-nobody-published"),
        "{}",
        check.detail
    );
}

#[test]
fn a_camera_that_is_off_does_not_need_its_model() {
    let mut config = config_with(2);
    config.cameras[1].pose_model = Some("a-model-nobody-published".to_owned());
    config.cameras[1].enabled = false;
    let catalogue = Manifest::load().unwrap();

    assert_eq!(
        startup::models(&config, Some(&catalogue), &|_| true).verdict,
        Verdict::Ready
    );
}

#[test]
fn a_fresh_install_is_told_to_calibrate() {
    let config = config_with(2);
    let check = startup::room_profile(&config, None);

    assert_eq!(check.verdict, Verdict::Blocked);
    assert!(check.fix.unwrap().contains("Calibration"));
}

#[test]
fn a_profile_that_could_not_be_loaded_still_names_itself() {
    let mut config = config_with(2);
    config.room = Some("living-room".to_owned());

    // The name is kept in the config on purpose: a profile missing because a
    // folder was moved is a restore, not a recalibration.
    let check = startup::room_profile(&config, None);

    assert_eq!(check.verdict, Verdict::Blocked);
    assert!(check.detail.contains("living-room"), "{}", check.detail);
}

#[test]
fn a_camera_outside_the_profile_is_not_lost_quietly() {
    let mut config = config_with(3);
    config.room = Some("living-room".to_owned());

    // The third camera streams, finds a person, and contributes nothing,
    // because nothing knows where it is looking from. From the Cameras panel it
    // is indistinguishable from a camera that is working.
    let check = startup::room_profile(&config, Some(&solved(2)));

    assert_eq!(check.verdict, Verdict::Warning);
    assert!(check.detail.contains("Camera 2"), "{}", check.detail);
}

#[test]
fn a_profile_whose_cameras_are_switched_off_blocks() {
    let mut config = config_with(3);
    config.room = Some("living-room".to_owned());
    config.cameras[1].enabled = false;
    config.cameras[2].enabled = false;

    let check = startup::room_profile(&config, Some(&solved(3)));

    assert_eq!(check.verdict, Verdict::Blocked);
    assert!(check.detail.contains("living-room"), "{}", check.detail);
}

#[test]
fn a_room_that_matches_its_cameras_is_ready() {
    let mut config = config_with(3);
    config.room = Some("living-room".to_owned());

    assert_eq!(
        startup::room_profile(&config, Some(&solved(3))).verdict,
        Verdict::Ready
    );
}

#[test]
fn the_report_speaks_for_its_worst_check() {
    let report = optra::startup::Report {
        checks: vec![
            startup::cameras(&config_with(2), Some(&paths(&config_with(2)))),
            startup::room_profile(&config_with(2), None),
        ],
    };

    assert_eq!(report.verdict(), Verdict::Blocked);
    assert!(!report.is_clear());
    assert_eq!(report.problems().count(), 1);
}
