//! The calibration procedure end to end, against a room whose answer is known.
//!
//! This is the sequence from the design document: record the walk, resect each
//! camera on its own, then refine every camera together with the offset between
//! the headset and the head keypoint. Nothing here needs a headset or a camera,
//! which is the point — the geometry can be wrong in ways that only show up as
//! a few centimetres of drift, and that is not something to discover with a
//! user standing in the room.

use nalgebra::{Isometry3, Point3, Translation3, UnitQuaternion, Vector3};

use std::time::{Duration, Instant};

use optra::calib::recorder::{CameraTrail, Recording, Rig, Sample};
use optra::calib::{RoomCalibration, SolveOptions, solve};
use optra::config::{CameraConfig, LensKind};
use optra::geometry::camera::{Camera, Intrinsics};
use optra::geometry::lens::Lens;
use optra::geometry::refine::{RefineOptions, Sighting, offset_observability, refine};
use optra::geometry::resection::{Correspondence, ResectionOptions, resect};
use optra::models::keypoints::Joint;
use optra::vr::{Role, Track};

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
                // The pitch is the part that matters and the part a real user
                // forgets. Turning left and right leaves the vertical axis
                // fixed, and a shift of every camera along a fixed axis is
                // indistinguishable from a shift of the head offset — so a walk
                // with yaw alone, however much of it, cannot say how high the
                // cameras are.
                UnitQuaternion::from_euler_angles(
                    0.45 * (1.3 * t).sin(),
                    0.85 * t,
                    0.18 * (0.6 * t).cos(),
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

/// The rigs a walk is recorded against, and each one's offset from the device
/// it hangs off.
fn rigs() -> (Vec<Rig>, Vec<Vector3<f64>>) {
    (
        vec![
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
        ],
        vec![
            HEAD_OFFSET,
            Vector3::new(-0.02, 0.03, 0.09),
            Vector3::new(0.02, 0.03, 0.09),
        ],
    )
}

/// Records the walk as `cameras` would have seen it, with camera `index`
/// stamping each of its frames `delays[index]` later than it really exposed
/// them.
///
/// That is what a camera latency is: not a frame that arrives late, which
/// costs nothing, but a frame whose *timestamp* is later than the instant it
/// shows. The solver pairs each pixel with where the headset was at the
/// stamped time, so an uncorrected delay pairs every pixel with a pose from
/// after the shutter, and the room is solved against a walk that never
/// happened that way.
fn recorded_walk(cameras: &[Camera], delays: &[Duration]) -> Recording {
    let start = Instant::now();
    let (rigs, offsets) = rigs();

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
    for (index, camera) in cameras.iter().enumerate() {
        let delay = delays.get(index).copied().unwrap_or(Duration::ZERO);
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
                let exposed_at = *at + Duration::from_millis(3);
                let Some(anchor) = tracks[rig].at(exposed_at) else {
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
                    // The pixel is what the camera saw when it exposed; the
                    // stamp is when it got round to saying so.
                    at: exposed_at + delay,
                    rig,
                    pixel,
                    confidence: 0.9,
                });
            }
        }

        trails.push(trail);
    }

    Recording {
        rigs,
        tracks,
        cameras: trails,
        duration: Duration::from_secs(20),
    }
}

fn configs(count: usize) -> Vec<CameraConfig> {
    (0..count)
        .map(|index| CameraConfig {
            id: format!("cam{index}"),
            lens: if index == 2 {
                LensKind::Wide
            } else {
                LensKind::Standard
            },
            ..CameraConfig::default()
        })
        .collect()
}

/// Worst distance between a solved camera and where it really is, in metres.
fn worst_camera_error(solved: &RoomCalibration, truth: &[Camera]) -> f64 {
    solved
        .cameras
        .iter()
        .zip(truth)
        .map(|(calibrated, camera)| (calibrated.camera.position() - camera.position()).norm())
        .fold(0.0, f64::max)
}

/// The recorder-to-solver path, on a walk built from the same room as above.
///
/// The maths is already covered; what this exercises is the glue — rig
/// indexing, pairing each pixel with the pose at its own timestamp, and the
/// per-camera reporting that comes back out.
#[test]
fn a_recording_solves_into_a_room() {
    let truth = room();
    let recording = recorded_walk(&truth, &[]);
    let (rigs, offsets) = rigs();

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

    let solved = solve(&recording, &configs(truth.len()), &SolveOptions::default())
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

/// Cameras hand their frames over late, by different amounts, and the solve
/// measures that and then fits again against the corrected timestamps.
///
/// The second fit is glue rather than maths. The estimator has its own tests
/// for the search and the parabola, and none of them can say whether the
/// result is put back in the right camera's slot, whether the sightings are
/// re-paired against it, or whether the refinement restarts from the seeds
/// rather than from the answer it already had. Until now the only end-to-end
/// recording carried no delay at all, so this whole branch never ran.
#[test]
fn a_recording_with_late_cameras_solves_once_the_delay_is_measured() {
    let truth = room();
    let delays = [
        Duration::ZERO,
        Duration::from_millis(40),
        Duration::from_millis(90),
        Duration::from_millis(20),
    ];
    let recording = recorded_walk(&truth, &delays);
    let configs = configs(truth.len());

    let solved = solve(&recording, &configs, &SolveOptions::default())
        .expect("a late recording should still solve");

    for (index, calibrated) in solved.cameras.iter().enumerate() {
        let measured = calibrated
            .latency
            .unwrap_or_else(|| panic!("{} was given no latency at all", calibrated.id));
        let expected = delays[index].as_secs_f64() * 1000.0;
        println!(
            "{} is {expected:.0} ms late, measured {:.1} ms",
            calibrated.id,
            measured.millis()
        );
        assert!(
            (measured.millis() - expected).abs() < 8.0,
            "{} is {expected:.0} ms late and was measured at {:.1} ms",
            calibrated.id,
            measured.millis()
        );
    }

    for (index, calibrated) in solved.cameras.iter().enumerate() {
        println!(
            "  {} {:.1} mm, rms {:.4} deg, {} sightings, spread {:.3}",
            calibrated.id,
            (calibrated.camera.position() - truth[index].position()).norm() * 1000.0,
            calibrated.rms_degrees(),
            calibrated.sightings,
            calibrated.spread
        );
    }
    let corrected = worst_camera_error(&solved, &truth);
    println!(
        "worst camera {:.1} mm once the delays were corrected",
        corrected * 1000.0
    );
    assert!(
        corrected < 8e-3,
        "worst camera is {corrected} m out even after the delays were measured"
    );

    // What the second fit is worth. Solving the same recording while insisting
    // the cameras are prompt pairs every pixel with a pose from tens of
    // milliseconds after the shutter, and it used to converge on a room with a
    // camera nine metres from where it hangs — cheerfully, as an `Ok`.
    //
    // It no longer comes back at all, which is the better answer and is checked
    // here rather than the nine metres: the cameras that were fitted to a walk
    // they had not caught up with keep a fraction of their sightings and land
    // tens of degrees from the ones they keep, and that is now refused.
    let ignored = solve(
        &recording,
        &configs,
        &SolveOptions {
            estimate_latency: false,
            ..SolveOptions::default()
        },
    );
    let refused = ignored
        .as_ref()
        .err()
        .map(|error| error.to_string())
        .unwrap_or_else(|| {
            format!(
                "no refusal at all, and a room {:.0} mm out",
                ignored
                    .as_ref()
                    .map(|room| worst_camera_error(room, &truth) * 1000.0)
                    .unwrap_or(f64::NAN)
            )
        });
    println!("with the delays ignored: {refused}");
    assert!(
        ignored.is_err(),
        "a room solved as though three late cameras were prompt was returned \
         as usable: {refused}"
    );
}

/// Delays large enough that the first resection cannot be done at all.
///
/// The whole latency correction hangs off a chicken and egg. Measuring a
/// camera's delay needs a camera to reproject through, so the delay can only be
/// found after the first fit — and a camera more than about sixty milliseconds
/// late does not fit. Its pixels, paired with poses from sixty milliseconds
/// after the shutter, do not agree with any one camera, and the resection ends
/// with no consensus rather than with a bad answer.
///
/// That used to end the whole calibration, before the estimator ever ran, on a
/// message about correspondences that named neither the camera's problem nor
/// its cause. Three of these four cameras failed at eighty milliseconds and all
/// four at a hundred and ten, which is inside the range the estimator itself
/// calls plausible for a webcam.
///
/// The seed now searches for a delay that does resect, and the delays here are
/// spread across the range where that search is the only thing standing between
/// a solved room and an error message.
#[test]
fn cameras_too_late_to_resect_are_still_solved() {
    let truth = room();
    let configs = configs(truth.len());

    for delays in [
        [0, 80, 0, 0],
        [0, 0, 0, 110],
        [40, 90, 120, 60],
        // Every camera late by the same amount, which is what a room full of
        // one model of webcam looks like and what a per-camera search could be
        // forgiven for treating as no delay at all.
        [70, 70, 70, 70],
    ] {
        let delays: Vec<Duration> = delays.iter().map(|ms| Duration::from_millis(*ms)).collect();
        let recording = recorded_walk(&truth, &delays);
        let solved = solve(&recording, &configs, &SolveOptions::default())
            .unwrap_or_else(|error| panic!("{delays:?} should still solve, got: {error}"));

        for (index, calibrated) in solved.cameras.iter().enumerate() {
            let expected = delays[index].as_secs_f64() * 1000.0;
            let measured = calibrated
                .latency
                .map(|estimate| estimate.millis())
                .unwrap_or(f64::NAN);
            assert!(
                (measured - expected).abs() < 8.0,
                "{} is {expected:.0} ms late and was measured at {measured:.1} ms",
                calibrated.id
            );
        }

        let worst = worst_camera_error(&solved, &truth);
        println!("{delays:?} -> worst camera {:.1} mm", worst * 1000.0);
        assert!(
            worst < 0.01,
            "worst camera is {:.1} mm out with delays {delays:?}",
            worst * 1000.0
        );
    }
}

/// A room that did not solve is refused rather than returned.
///
/// The failure this guards against does not look like one. When a camera ends
/// up somewhere wrong, the refinement's outlier rejection throws away every
/// sighting that disagreed with it, so what comes back is a handful of
/// sightings with a small error over them — and `solve` returns `Ok`, the
/// wizard saves a profile, and the feet are a metre out for as long as it is in
/// force.
///
/// Both of the cases here were found by sweeping delays past what the tests
/// used to cover, and both came back `Ok`: one camera two and a half metres
/// from where it hangs, another a metre and a half.
#[test]
fn a_camera_that_did_not_solve_is_refused_rather_than_returned() {
    let truth = room();
    let configs = configs(truth.len());

    // The seed search does not run here, because resecting against prompt
    // timestamps *succeeds* — it just succeeds somewhere wrong, and the
    // latency estimator then finds its minimum at a delay of zero. Only two of
    // this camera's two hundred and one sightings survived the fit.
    let mut delays = vec![Duration::ZERO; 4];
    delays[2] = Duration::from_millis(80);
    let error = solve(
        &recorded_walk(&truth, &delays),
        &configs,
        &SolveOptions::default(),
    )
    .expect_err("a camera solved to nonsense should not come back as a room")
    .to_string();
    assert!(
        error.contains("cam2") && error.contains("sightings"),
        "the refusal should name the camera and what was wrong with it: {error}"
    );

    // Late enough that the delay is measured and then not applied. The room is
    // fitted at whatever delay the seed search landed on, and a fit ten
    // milliseconds away from the truth is self-consistent and half a metre out
    // — which is exactly what reprojection error cannot see, since a timing
    // error moves every sighting the same way.
    let mut delays = vec![Duration::ZERO; 4];
    delays[0] = Duration::from_millis(150);
    let error = solve(
        &recorded_walk(&truth, &delays),
        &configs,
        &SolveOptions::default(),
    )
    .expect_err("a room fitted at the wrong delay should not come back")
    .to_string();
    assert!(
        error.contains("cam0") && error.contains("behind"),
        "the refusal should say the delay was not the one the room was solved at: {error}"
    );
}

/// And the gate does not fire on rooms that did solve.
///
/// The thresholds separate populations four orders of magnitude apart — a
/// solved camera keeps every sighting and lands within hundredths of a degree,
/// a lost one keeps one per cent of them and lands tens of degrees out — but a
/// gate that refuses good rooms is worse than no gate, so the delays that do
/// work are checked as well as the ones that do not.
#[test]
fn a_room_that_solved_is_not_refused() {
    let truth = room();
    let configs = configs(truth.len());

    for delays in [
        [0, 0, 0, 0],
        [0, 40, 90, 20],
        [40, 90, 120, 60],
        [70, 70, 70, 70],
    ] {
        let delays: Vec<Duration> = delays.iter().map(|ms| Duration::from_millis(*ms)).collect();
        let solved = solve(
            &recorded_walk(&truth, &delays),
            &configs,
            &SolveOptions::default(),
        )
        .unwrap_or_else(|error| panic!("{delays:?} solves, and was refused: {error}"));
        assert!(
            worst_camera_error(&solved, &truth) < 0.01,
            "{delays:?} was accepted and is {:.1} mm out",
            worst_camera_error(&solved, &truth) * 1000.0
        );
    }
}
