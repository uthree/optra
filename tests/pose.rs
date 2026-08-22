//! End-to-end inference against real models and a real photograph.
//!
//! These download models and a test image, so they are ignored by default:
//!
//! ```text
//! cargo test --release --test pose -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use optra::infer::arch;
use optra::infer::session::ProviderChoice;
use optra::infer::traits::{Detection, ImageView, Keypoints2d};
use optra::models::manifest::Manifest;
use optra::models::{Joint, ModelSpec, store};

/// A COCO validation photograph of one person, whole body visible.
const TEST_IMAGE: &str =
    "https://raw.githubusercontent.com/open-mmlab/mmpose/main/tests/data/coco/000000000785.jpg";

fn spec(id: &str) -> ModelSpec {
    Manifest::load()
        .expect("the catalogue")
        .into_iter()
        .find(|spec| spec.id == id)
        .unwrap_or_else(|| panic!("{id} is in the catalogue"))
}

fn install(id: &str) -> PathBuf {
    store::install(&spec(id), &mut |_| {}).expect("the model should install")
}

/// Fetches the test image once and keeps it next to the models.
fn test_image() -> (u32, u32, Vec<u8>) {
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

    let decoded = image::open(&path)
        .expect("the image should decode")
        .to_rgb8();
    let (width, height) = decoded.dimensions();
    (width, height, decoded.into_raw())
}

fn best(detections: &[Detection]) -> Detection {
    *detections
        .iter()
        .max_by(|a, b| a.score.total_cmp(&b.score))
        .expect("at least one detection")
}

#[test]
#[ignore = "downloads models"]
fn finds_a_person_and_their_keypoints() {
    let (width, height, rgb) = test_image();
    let view = ImageView::new(width, height, &rgb);
    println!("image: {width}x{height}");

    install("yolox-tiny-humanart-416");
    install("rtmpose-m-halpe26-256x192");

    let mut detector = arch::build_detector(&spec("yolox-tiny-humanart-416"), ProviderChoice::Cpu)
        .expect("the detector should load");
    let mut pose = arch::build_pose2d(&spec("rtmpose-m-halpe26-256x192"), ProviderChoice::Cpu)
        .expect("the pose model should load");

    let started = Instant::now();
    let detections = detector.detect(&[view]).expect("detection should run");
    println!(
        "detection: {} person(s) in {:.1} ms",
        detections[0].len(),
        started.elapsed().as_secs_f32() * 1000.0
    );
    assert!(!detections[0].is_empty(), "no person was detected");

    let person = best(&detections[0]);
    println!(
        "best box: ({:.0}, {:.0}) to ({:.0}, {:.0}) score {:.2}",
        person.x1, person.y1, person.x2, person.y2, person.score
    );
    assert!(person.score > 0.5);
    assert!(person.width() > 10.0 && person.height() > 10.0);
    assert!(person.x2 <= width as f32 && person.y2 <= height as f32);

    let started = Instant::now();
    let keypoints = pose
        .estimate(&[(view, person)])
        .expect("pose estimation should run");
    println!(
        "pose: {} keypoints in {:.1} ms",
        keypoints[0].count(),
        started.elapsed().as_secs_f32() * 1000.0
    );

    let people = &keypoints[0];
    for (joint, kp) in people.iter() {
        println!(
            "  {joint:?}: ({:.0}, {:.0}) {:.2}",
            kp.x, kp.y, kp.confidence
        );
    }

    assert_plausible(people, &person);
}

/// The keypoints have to be anatomically ordered and inside the person's box.
///
/// This is the check that catches a wrong keypoint layout, a mis-decoded SimCC
/// axis, or a broken coordinate mapping, all of which otherwise produce a
/// confident-looking skeleton made of nonsense.
fn assert_plausible(keypoints: &Keypoints2d, person: &Detection) {
    for joint in [
        Joint::Nose,
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
        assert!(
            keypoints.get(joint).is_some(),
            "{joint:?} is missing, so the layout or the decode is wrong"
        );
    }

    // Image y grows downward, so a standing person runs nose, hip, knee, ankle.
    let y = |joint: Joint| keypoints.get(joint).unwrap().y;
    assert!(
        y(Joint::Nose) < y(Joint::LeftHip),
        "the head is below the hips"
    );
    assert!(
        y(Joint::LeftHip) < y(Joint::LeftKnee),
        "the hip is below the knee"
    );
    assert!(
        y(Joint::LeftKnee) < y(Joint::LeftAnkle),
        "the knee is below the ankle"
    );
    assert!(
        y(Joint::RightKnee) < y(Joint::RightAnkle),
        "the knee is below the ankle"
    );

    // Toes are in front of heels, and both sit at or below the ankle.
    assert!(y(Joint::LeftAnkle) <= y(Joint::LeftHeel) + 20.0);
    assert!(y(Joint::LeftAnkle) <= y(Joint::LeftBigToe) + 20.0);

    // Everything should land within the detection, with a little slack for the
    // padding the crop adds.
    let margin = person.width().max(person.height()) * 0.35;
    for (joint, kp) in keypoints.iter() {
        assert!(
            kp.x > person.x1 - margin
                && kp.x < person.x2 + margin
                && kp.y > person.y1 - margin
                && kp.y < person.y2 + margin,
            "{joint:?} at ({:.0}, {:.0}) is outside the person's box",
            kp.x,
            kp.y
        );
    }
}

/// Several crops in one call must come back in order and match what a
/// one-at-a-time run produces, since the pipeline batches every camera together.
#[test]
#[ignore = "downloads models"]
fn batching_matches_running_one_at_a_time() {
    let (width, height, rgb) = test_image();
    let view = ImageView::new(width, height, &rgb);

    install("yolox-tiny-humanart-416");
    install("rtmpose-m-halpe26-256x192");

    let mut detector = arch::build_detector(&spec("yolox-tiny-humanart-416"), ProviderChoice::Cpu)
        .expect("the detector should load");
    let mut pose = arch::build_pose2d(&spec("rtmpose-m-halpe26-256x192"), ProviderChoice::Cpu)
        .expect("the pose model should load");

    let person = best(&detector.detect(&[view]).expect("detection")[0]);

    // A second, deliberately different box, so an order mix-up would show.
    let shifted = Detection {
        x1: person.x1 + person.width() * 0.1,
        y1: person.y1,
        x2: person.x2 + person.width() * 0.1,
        y2: person.y2,
        score: person.score,
    };

    let single_a = pose.estimate(&[(view, person)]).expect("pose")[0].clone();
    let single_b = pose.estimate(&[(view, shifted)]).expect("pose")[0].clone();
    let batched = pose
        .estimate(&[(view, person), (view, shifted)])
        .expect("pose");

    assert_eq!(batched.len(), 2);
    for (index, (batched, single)) in [(&batched[0], &single_a), (&batched[1], &single_b)]
        .into_iter()
        .enumerate()
    {
        assert_eq!(
            batched.count(),
            single.count(),
            "crop {index} produced a different keypoint count when batched"
        );
        for (joint, kp) in single.iter() {
            let other = batched
                .get(joint)
                .unwrap_or_else(|| panic!("crop {index} lost {joint:?} when batched"));
            assert!(
                (kp.x - other.x).abs() < 0.5 && (kp.y - other.y).abs() < 0.5,
                "crop {index} moved {joint:?} when batched"
            );
        }
    }
}
