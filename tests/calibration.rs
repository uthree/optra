//! The calibration procedure end to end, against a room whose answer is known.
//!
//! This is the sequence from the design document: record the walk, resect each
//! camera on its own, then refine every camera together with the offset between
//! the headset and the head keypoint. Nothing here needs a headset or a camera,
//! which is the point — the geometry can be wrong in ways that only show up as
//! a few centimetres of drift, and that is not something to discover with a
//! user standing in the room.

use nalgebra::{Isometry3, Point3, Translation3, UnitQuaternion, Vector3};

use optra::geometry::camera::{Camera, Intrinsics};
use optra::geometry::lens::Lens;
use optra::geometry::refine::{RefineOptions, Sighting, offset_observability, refine};
use optra::geometry::resection::{Correspondence, ResectionOptions, resect};

/// Where the head keypoint sits relative to the headset origin, in the
/// headset's own frame: up a little, back a little.
const HEAD_OFFSET: Vector3<f64> = Vector3::new(0.012, 0.055, 0.13);

/// Four cameras in the ceiling corners, deliberately unlike each other: the
/// room this software is for is whatever webcams the user already owned.
fn room() -> Vec<Camera> {
    let place = |x: f64, z: f64, width: u32, height: u32, fov: f64, lens: Lens| {
        Camera::look_at(
            Intrinsics::from_fov(width, height, fov.to_radians()),
            lens,
            Point3::new(x, 2.45, z),
            Point3::new(0.0, 1.0, 0.0),
            Vector3::y(),
        )
    };

    vec![
        place(-1.95, -1.95, 1280, 720, 70.0, Lens::default()),
        place(1.95, -1.95, 1920, 1080, 62.0, Lens::default()),
        place(
            1.95,
            1.95,
            640,
            480,
            95.0,
            Lens::RadialTangential {
                k1: -0.21,
                k2: 0.05,
                p1: 0.0003,
                p2: -0.0004,
            },
        ),
        place(-1.95, 1.95, 1280, 720, 78.0, Lens::default()),
    ]
}

/// The calibration walk: a path that covers floor area, changes height, and
/// turns the head, because all three are needed for the solve to be well posed.
fn walk() -> Vec<Isometry3<f64>> {
    (0..200)
        .map(|step| {
            let t = step as f64 * 0.09;
            Isometry3::from_parts(
                Translation3::new(
                    1.25 * t.sin(),
                    1.4 + 0.32 * (1.7 * t).sin(),
                    1.15 * (0.7 * t).cos(),
                ),
                UnitQuaternion::from_euler_angles(
                    0.22 * (1.3 * t).sin(),
                    0.85 * t,
                    0.1 * (0.6 * t).cos(),
                ),
            )
        })
        .collect()
}

fn inside(camera: &Camera, pixel: nalgebra::Point2<f64>) -> bool {
    pixel.x >= 0.0
        && pixel.y >= 0.0
        && pixel.x < camera.intrinsics.width as f64
        && pixel.y < camera.intrinsics.height as f64
}

#[test]
fn a_recorded_walk_calibrates_the_room() {
    let truth = room();
    let walk = walk();

    // What the cameras actually saw: the head keypoint, which is not where the
    // headset reports itself to be.
    let mut sightings: Vec<Sighting> = Vec::new();
    for anchor in &walk {
        let head = anchor * Point3::from(HEAD_OFFSET);
        for (index, camera) in truth.iter().enumerate() {
            let Some(pixel) = camera.project(head) else {
                continue;
            };
            if inside(camera, pixel) {
                sightings.push(Sighting {
                    camera: index,
                    rig: 0,
                    anchor: *anchor,
                    pixel,
                    weight: 1.0,
                });
            }
        }
    }
    assert!(
        sightings.len() > 400,
        "the walk should be widely visible, got {}",
        sightings.len()
    );

    assert!(
        offset_observability(&sightings, 0) > 0.3,
        "the walk turns the head enough to pin the offset down"
    );

    // Step one: each camera on its own, from the headset origin. The offset is
    // not known yet, so every correspondence is wrong by the same thirteen
    // centimetres in the headset's frame — which is why the inlier threshold
    // here has to be loose enough to absorb it. Tightening it would throw the
    // whole walk away rather than reject bad detections.
    let options = ResectionOptions {
        inlier_threshold: 0.08,
        ..ResectionOptions::default()
    };

    let mut seeded = Vec::new();
    for (index, camera) in truth.iter().enumerate() {
        let correspondences: Vec<Correspondence> = sightings
            .iter()
            .filter(|s| s.camera == index)
            .map(|s| Correspondence {
                world: Point3::from(s.anchor.translation.vector),
                pixel: s.pixel,
            })
            .collect();

        // A field of view guess, as a driver would report it, not the answer.
        let guess = Intrinsics::from_fov(
            camera.intrinsics.width,
            camera.intrinsics.height,
            85f64.to_radians(),
        );

        let resection = resect(&guess, camera.lens, &correspondences, &options)
            .unwrap_or_else(|| panic!("camera {index} should resect"));
        assert!(
            resection.is_well_conditioned(),
            "camera {index} saw a degenerate walk"
        );
        seeded.push(resection.camera);
    }

    // The seed is close but not right: the unmodelled head offset shows up as
    // several centimetres of camera position error.
    let seed_error = seeded
        .iter()
        .zip(&truth)
        .map(|(a, b)| (a.position() - b.position()).norm())
        .fold(0.0, f64::max);
    assert!(
        seed_error > 0.005,
        "the seed is expected to be biased, it was off by only {seed_error} m"
    );

    // Step two: everything at once, with the offset free.
    let result = refine(
        &seeded,
        &[Vector3::zeros()],
        &sightings,
        &RefineOptions::default(),
    );

    assert_eq!(result.rejected, 0, "clean data should lose no sightings");
    assert!(
        result.rms < 1e-5,
        "rms was {} rad over {} sightings",
        result.rms,
        sightings.len()
    );

    let position_error = result
        .cameras
        .iter()
        .zip(&truth)
        .map(|(a, b)| (a.position() - b.position()).norm())
        .fold(0.0, f64::max);
    let angle_error = result
        .cameras
        .iter()
        .zip(&truth)
        .map(|(a, b)| a.pose.rotation.angle_to(&b.pose.rotation))
        .fold(0.0, f64::max);

    assert!(
        position_error < 3e-3,
        "worst camera is {position_error} m from where it really is"
    );
    assert!(
        angle_error < 2e-3,
        "worst camera is {angle_error} rad from where it really points"
    );
    assert!(
        (result.offsets[0] - HEAD_OFFSET).norm() < 3e-3,
        "recovered the head offset as {:?}, expected {HEAD_OFFSET:?}",
        result.offsets[0]
    );

    for (index, camera) in result.per_camera.iter().enumerate() {
        assert!(
            camera.sightings > 50,
            "camera {index} contributed only {} sightings",
            camera.sightings
        );
        assert!(
            camera.worst < 1e-4,
            "camera {index} has a {} rad residual left",
            camera.worst
        );
    }
}

/// Half the walk missing from one camera, and a scattering of bad keypoints
/// everywhere: what a real recording looks like.
#[test]
fn a_messy_walk_still_calibrates() {
    let truth = room();
    let walk = walk();

    let mut sightings: Vec<Sighting> = Vec::new();
    for (frame, anchor) in walk.iter().enumerate() {
        let head = anchor * Point3::from(HEAD_OFFSET);
        for (index, camera) in truth.iter().enumerate() {
            // One camera is blocked for the first half of the walk.
            if index == 3 && frame < walk.len() / 2 {
                continue;
            }
            let Some(mut pixel) = camera.project(head) else {
                continue;
            };
            if !inside(camera, pixel) {
                continue;
            }
            // Roughly one keypoint in twelve lands somewhere else entirely.
            if (frame + index) % 12 == 0 {
                pixel.x += 90.0;
                pixel.y -= 70.0;
            }
            sightings.push(Sighting {
                camera: index,
                rig: 0,
                anchor: *anchor,
                pixel,
                weight: 1.0,
            });
        }
    }

    // Cameras roughly where a resection would have left them.
    let seeded: Vec<Camera> = room()
        .into_iter()
        .enumerate()
        .map(|(index, mut camera)| {
            let sign = if index % 2 == 0 { 1.0 } else { -1.0 };
            camera.intrinsics.fx *= 1.0 + 0.02 * sign;
            camera.intrinsics.fy *= 1.0 + 0.02 * sign;
            camera.pose.rotation =
                UnitQuaternion::from_scaled_axis(Vector3::new(0.01 * sign, -0.008, 0.005))
                    * camera.pose.rotation;
            camera.pose.translation.vector += Vector3::new(0.03 * sign, -0.02, 0.04);
            camera
        })
        .collect();

    let result = refine(
        &seeded,
        &[Vector3::zeros()],
        &sightings,
        &RefineOptions::default(),
    );

    assert!(
        result.rejected > 0,
        "the bad keypoints should have been thrown out"
    );
    assert!(
        result.rejected < sightings.len() / 5,
        "it threw out {} of {} sightings",
        result.rejected,
        sightings.len()
    );

    let position_error = result
        .cameras
        .iter()
        .zip(&truth)
        .map(|(a, b)| (a.position() - b.position()).norm())
        .fold(0.0, f64::max);
    assert!(
        position_error < 3e-3,
        "worst camera is {position_error} m from where it really is"
    );
    assert!(
        (result.offsets[0] - HEAD_OFFSET).norm() < 3e-3,
        "recovered the head offset as {:?}",
        result.offsets[0]
    );
}

/// The recorder-to-solver path, on a walk built from the same room as above.
///
/// The maths is already covered; what this exercises is the glue — rig
/// indexing, pairing each pixel with the pose at its own timestamp, and the
/// per-camera reporting that comes back out.
#[test]
fn a_recording_solves_into_a_room() {
    use std::time::{Duration, Instant};

    use optra::calib::recorder::{CameraTrail, Recording, Rig, Sample};
    use optra::calib::{SolveOptions, solve};
    use optra::config::{CameraConfig, LensKind};
    use optra::models::keypoints::Joint;
    use optra::vr::{Role, Track};

    let truth = room();
    let start = Instant::now();

    // Three rigs, each with its own offset from the device it hangs off.
    let rigs = vec![
        Rig {
            role: Role::Head,
            joint: Joint::Head,
        },
        Rig {
            role: Role::LeftHand,
            joint: Joint::LeftWrist,
        },
        Rig {
            role: Role::RightHand,
            joint: Joint::RightWrist,
        },
    ];
    let offsets = [
        HEAD_OFFSET,
        Vector3::new(-0.02, 0.03, 0.09),
        Vector3::new(0.02, 0.03, 0.09),
    ];

    // Where each device was, sampled far more often than the cameras run.
    let mut tracks = vec![Track::default(); rigs.len()];
    let mut anchors: Vec<Vec<(Instant, Isometry3<f64>)>> = vec![Vec::new(); rigs.len()];

    for (step, head) in walk().into_iter().enumerate() {
        let at = start + Duration::from_millis(step as u64 * 8);

        // The hands swing relative to the head, so their tracks are not a
        // translated copy of it.
        let phase = step as f64 * 0.19;
        let devices = [
            head,
            head * Isometry3::from_parts(
                Translation3::new(-0.25, -0.45 + 0.3 * phase.sin(), -0.2),
                UnitQuaternion::from_euler_angles(0.4 * phase.cos(), -0.6, 0.2),
            ),
            head * Isometry3::from_parts(
                Translation3::new(0.25, -0.45 + 0.3 * (phase + 1.7).sin(), -0.2),
                UnitQuaternion::from_euler_angles(0.4 * (phase + 1.1).cos(), 0.6, -0.2),
            ),
        ];

        for (rig, device) in devices.into_iter().enumerate() {
            tracks[rig].push(at, device);
            anchors[rig].push((at, device));
        }
    }

    // What each camera saw. Frames land between pose samples, which is the
    // normal case and the reason the recorder interpolates at all.
    let mut trails = Vec::new();
    for (index, camera) in truth.iter().enumerate() {
        let mut trail = CameraTrail::new(format!("cam{index}"));
        trail.width = camera.intrinsics.width;
        trail.height = camera.intrinsics.height;

        for (rig, samples) in anchors.iter().enumerate() {
            for (step, (at, _)) in samples.iter().enumerate() {
                // Every third pose sample carries a camera frame, offset by
                // three milliseconds so it never coincides with one.
                if step % 3 != 0 || step + 1 >= samples.len() {
                    continue;
                }
                let frame_at = *at + Duration::from_millis(3);
                let Some(anchor) = tracks[rig].at(frame_at) else {
                    continue;
                };

                let point = anchor * Point3::from(offsets[rig]);
                let Some(pixel) = camera.project(point) else {
                    continue;
                };
                if !inside(camera, pixel) {
                    continue;
                }

                trail.record(Sample {
                    at: frame_at,
                    rig,
                    pixel,
                    confidence: 0.9,
                });
            }
        }

        trails.push(trail);
    }

    let recording = Recording {
        rigs: rigs.clone(),
        tracks,
        cameras: trails,
        duration: Duration::from_secs(20),
    };

    assert!(
        recording.samples() > 500,
        "the synthetic walk should be well seen, got {}",
        recording.samples()
    );
    for progress in recording.observability() {
        assert!(
            progress.spread > 0.2,
            "{} barely turned during the walk: {:.3}",
            progress.rig.label(),
            progress.spread
        );
        assert!(
            progress.samples > 100,
            "{} was barely seen",
            progress.rig.label()
        );
    }

    let configs: Vec<CameraConfig> = (0..truth.len())
        .map(|index| CameraConfig {
            id: format!("cam{index}"),
            lens: if index == 2 {
                LensKind::Wide
            } else {
                LensKind::Standard
            },
            ..CameraConfig::default()
        })
        .collect();

    let solved = solve(&recording, &configs, &SolveOptions::default())
        .expect("a clean recording should solve");

    println!(
        "solved {} cameras, rms {:.4} deg, {} of {} sightings used",
        solved.cameras.len(),
        solved.rms_degrees(),
        solved.used,
        solved.used + solved.rejected
    );

    assert_eq!(solved.cameras.len(), truth.len());
    for (index, calibrated) in solved.cameras.iter().enumerate() {
        let error = (calibrated.camera.position() - truth[index].position()).norm();
        assert!(
            error < 5e-3,
            "{} is {error} m from where it really is",
            calibrated.id
        );
        assert!(
            calibrated.spread > 0.05,
            "{} was solved from near-planar correspondences",
            calibrated.id
        );
        assert!(calibrated.sightings > 100);

        // The frames in this recording carry no delay, and a solve that
        // invents one would shift every camera against a walk that never
        // happened that way.
        if let Some(latency) = calibrated.latency {
            assert!(
                latency.millis() < 5.0,
                "{} was given {:.1} ms of latency that is not there",
                calibrated.id,
                latency.millis()
            );
        }
    }

    for (index, (rig, offset)) in solved.rigs.iter().enumerate() {
        assert_eq!(*rig, rigs[index]);
        assert!(
            (offset - offsets[index]).norm() < 5e-3,
            "{} offset came out {offset:?}, expected {:?}",
            rig.label(),
            offsets[index]
        );
    }
}
