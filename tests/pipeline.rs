//! The capture-to-keypoints chain, end to end.
//!
//! ```text
//! cargo test --release --test pipeline -- --ignored --nocapture
//! ```

use std::time::{Duration, Instant};

use optra::capture::CaptureManager;
use optra::config::{CameraConfig, InferenceConfig, SourceConfig};
use optra::infer::ProviderChoice;
use optra::models::manifest::Manifest;
use optra::models::{Joint, store};
use optra::pipeline::Pipeline;
use optra::worker::Supervisor;

const TEST_IMAGE: &str =
    "https://raw.githubusercontent.com/open-mmlab/mmpose/main/tests/data/coco/000000000785.jpg";

/// Fetches the test photograph once and returns where it lives.
fn test_image() -> std::path::PathBuf {
    let path = optra::paths::models_dir()
        .expect("models directory")
        .join("test-person.jpg");

    if !path.exists() {
        let mut response = ureq::get(TEST_IMAGE)
            .call()
            .expect("the image should fetch");
        let mut bytes = Vec::new();
        std::io::Read::read_to_end(&mut response.body_mut().as_reader(), &mut bytes)
            .expect("the image should download");
        std::fs::write(&path, &bytes).expect("the image should be written");
    }
    path
}

fn install(id: &str) {
    let spec = Manifest::load()
        .expect("the catalogue")
        .into_iter()
        .find(|spec| spec.id == id)
        .unwrap_or_else(|| panic!("{id} is in the catalogue"));
    store::install(&spec, &mut |_| {}).expect("the model should install");
}

/// Two cameras showing the same person, which is the shape the fusion stage
/// will consume: independent capture, independent inference, matching results.
#[test]
#[ignore = "downloads models"]
fn two_cameras_produce_keypoints() {
    install("yolox-tiny-humanart-416");
    install("rtmpose-m-halpe26-256x192");

    let image = test_image().display().to_string();
    let cameras: Vec<CameraConfig> = ["left", "right"]
        .iter()
        .map(|id| CameraConfig {
            id: (*id).to_owned(),
            label: (*id).to_owned(),
            enabled: true,
            source: SourceConfig::Still {
                path: image.clone(),
            },
            fps: 15,
            ..CameraConfig::default()
        })
        .collect();

    let inference = InferenceConfig {
        provider: ProviderChoice::Cpu,
        detect_every: 2,
        ..InferenceConfig::default()
    };

    let mut supervisor = Supervisor::new();
    let mut capture = CaptureManager::default();
    let mut pipeline = Pipeline::default();

    capture.start(&cameras, &mut supervisor);
    pipeline.start(inference, &cameras, capture.channels(), &mut supervisor);

    // Loading two models and processing a few frames; generous because the
    // first session build dominates.
    let deadline = Instant::now() + Duration::from_secs(60);
    let ready = loop {
        let done = cameras.iter().all(|camera| {
            pipeline
                .channel(&camera.id)
                .map(|channel| channel.stats().processed >= 3)
                .unwrap_or(false)
        });
        if done || Instant::now() > deadline {
            break done;
        }
        std::thread::sleep(Duration::from_millis(100));
    };

    for camera in &cameras {
        let channel = pipeline.channel(&camera.id).expect("a pose channel");
        let stats = channel.stats();
        println!(
            "{}: {} processed, {} empty, {:.1} fps, {:.0} ms latency, backend {:?}",
            camera.id, stats.processed, stats.empty, stats.fps, stats.latency_ms, stats.backend
        );
        if let Some(error) = &stats.last_error {
            println!("  last error: {error}");
        }
    }

    assert!(
        ready,
        "the pipeline did not process three frames per camera"
    );

    for camera in &cameras {
        let channel = pipeline.channel(&camera.id).expect("a pose channel");
        let frame = channel.peek().expect("a published pose frame");

        assert!(frame.detection.is_some(), "{} found no person", camera.id);
        for joint in [
            Joint::LeftAnkle,
            Joint::RightAnkle,
            Joint::LeftHeel,
            Joint::Hip,
        ] {
            assert!(
                frame.keypoints.get(joint).is_some(),
                "{} is missing {joint:?}",
                camera.id
            );
        }

        // Both cameras see the same picture, so the same joint should land in
        // the same place; a mismatch means results were crossed between
        // cameras.
        let other = cameras.iter().find(|c| c.id != camera.id).unwrap();
        let other_frame = pipeline.channel(&other.id).unwrap().peek().unwrap();
        let a = frame.keypoints.get(Joint::LeftAnkle).unwrap();
        let b = other_frame.keypoints.get(Joint::LeftAnkle).unwrap();
        assert!(
            (a.x - b.x).abs() < 2.0 && (a.y - b.y).abs() < 2.0,
            "the two cameras disagree about a joint in an identical image"
        );
    }

    pipeline.stop();
    capture.stop();
    supervisor.shutdown();
}
