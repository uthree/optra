//! The fusion chain, end to end, against a body whose position is known.
//!
//! Every piece of the stage has its own unit tests, and passing all of them
//! would still leave the obvious question unanswered: does a person walking
//! across a room come out of the other end in the right place? So this runs a
//! simulated walk past three unlike cameras — different resolutions, fields of
//! view, frame rates and delays — through the whole path a real one takes, and
//! compares the result against the body it started from.
//!
//! It is also the only test that can show the temporal alignment is worth
//! anything, because that requires cameras that disagree about the time.

use std::sync::Arc;
use std::time::{Duration, Instant};

use nalgebra::{Point2, Point3, Rotation3, Unit, Vector3};

use optra::fusion::align::align;
use optra::fusion::bones::BoneMeter;
use optra::fusion::filter::{FilterOptions, PoseFilter};
use optra::fusion::fit::Fitter;
use optra::fusion::fuse::{FuseOptions, fuse};
use optra::geometry::camera::{Camera, Intrinsics};
use optra::geometry::lens::Lens;
use optra::infer::traits::{Keypoint, Keypoints2d};
use optra::models::keypoints::Joint;
use optra::pipeline::{PoseChannel, PoseFrame};

/// The anatomy the simulated body is built from, in metres.
const THIGH: f64 = 0.44;
const SHIN: f64 = 0.42;
const HALF_HIPS: f64 = 0.12;
const SPINE: f64 = 0.50;
const NECK: f64 = 0.22;

/// Where every joint is at time `t`, in seconds from the start of the walk.
///
/// Built by forward kinematics from unit vectors, so the bones are exactly the
/// lengths above at every instant. A body assembled from independent sine waves
/// would have limbs that stretch, and the fit would be measured against
/// something no fit could reproduce.
fn body(t: f64) -> Vec<(Joint, Point3<f64>)> {
    let around = 0.5 * t;
    let centre = Point3::new(0.6 * around.sin(), 0.95, 0.6 * around.cos());
    let forward = Vector3::new(around.cos(), 0.0, -around.sin());
    let right = forward.cross(&Vector3::y());
    let axis = Unit::new_normalize(right);

    let stride = std::f64::consts::TAU * 1.6 * t;
    let mut joints = vec![
        (Joint::Hip, centre),
        (Joint::Neck, centre + Vector3::y() * SPINE),
        (Joint::Head, centre + Vector3::y() * (SPINE + NECK)),
    ];

    for (side, hip_joint, knee_joint, ankle_joint, offset) in [
        (-1.0, Joint::LeftHip, Joint::LeftKnee, Joint::LeftAnkle, 0.0),
        (
            1.0,
            Joint::RightHip,
            Joint::RightKnee,
            Joint::RightAnkle,
            std::f64::consts::PI,
        ),
    ] {
        let hip = centre + right * (HALF_HIPS * side);
        // The thigh swings about the hip and the knee only ever folds one way,
        // which is what keeps the leg on the side of the hip-to-ankle line that
        // a leg can be on.
        let swing = 0.45 * (stride + offset).sin();
        let bend = 0.35 * (1.0 - (stride + offset).cos());

        let thigh = Rotation3::from_axis_angle(&axis, swing) * (-Vector3::y());
        let knee = hip + thigh * THIGH;
        let shank = Rotation3::from_axis_angle(&axis, swing - bend) * (-Vector3::y());
        let ankle = knee + shank * SHIN;

        joints.push((hip_joint, hip));
        joints.push((knee_joint, knee));
        joints.push((ankle_joint, ankle));
    }

    joints
}

/// One camera in the simulated room.
struct Rig {
    camera: Camera,
    channel: Arc<PoseChannel>,
    /// How late this camera stamps its frames.
    latency: Duration,
    interval: Duration,
    /// Where in its own frame period it happens to sit, so no two cameras are
    /// in step.
    phase: Duration,
    next: u64,
    /// A joint this camera stops reporting after the given time, standing in
    /// for a limb that goes behind something.
    hidden: Option<(Joint, f64)>,
}

impl Rig {
    fn new(
        position: Point3<f64>,
        width: u32,
        height: u32,
        fov: f64,
        fps: f64,
        latency_ms: u64,
        phase_ms: u64,
    ) -> Self {
        Self {
            camera: Camera::look_at(
                Intrinsics::from_fov(width, height, fov.to_radians()),
                Lens::default(),
                position,
                Point3::new(0.0, 1.0, 0.0),
                Vector3::y(),
            ),
            channel: Arc::new(PoseChannel::default()),
            latency: Duration::from_millis(latency_ms),
            interval: Duration::from_secs_f64(1.0 / fps),
            phase: Duration::from_millis(phase_ms),
            next: 0,
            hidden: None,
        }
    }

    /// Publishes every frame this camera would have delivered by `now`.
    fn advance(&mut self, start: Instant, now: Instant, noise: &mut Noise) {
        loop {
            let exposed = start + self.phase + self.interval.mul_f64(self.next as f64);
            let stamped = exposed + self.latency;
            if stamped > now {
                return;
            }

            let seconds = exposed.duration_since(start).as_secs_f64();
            let mut keypoints = Keypoints2d::default();
            for (joint, point) in body(seconds) {
                if self
                    .hidden
                    .is_some_and(|(name, from)| name == joint && seconds >= from)
                {
                    continue;
                }
                let Some(pixel) = self.camera.project(point) else {
                    continue;
                };
                let jittered =
                    Point2::new(pixel.x + noise.next() * 0.8, pixel.y + noise.next() * 0.8);
                // A keypoint off the edge of the frame is one the model would
                // not have produced.
                if jittered.x < 0.0
                    || jittered.y < 0.0
                    || jittered.x >= self.camera.intrinsics.width as f64
                    || jittered.y >= self.camera.intrinsics.height as f64
                {
                    continue;
                }
                keypoints.set(
                    joint,
                    Keypoint {
                        x: jittered.x as f32,
                        y: jittered.y as f32,
                        confidence: 0.9,
                    },
                );
            }

            self.channel.publish(PoseFrame {
                seq: self.next,
                captured_at: stamped,
                width: self.camera.intrinsics.width,
                height: self.camera.intrinsics.height,
                detection: None,
                keypoints,
            });
            self.next += 1;
        }
    }
}

fn room() -> Vec<Rig> {
    vec![
        Rig::new(Point3::new(-1.9, 2.4, -1.9), 1280, 720, 74.0, 30.0, 0, 0),
        Rig::new(Point3::new(1.9, 2.4, -1.9), 1920, 1080, 62.0, 60.0, 40, 7),
        Rig::new(Point3::new(1.9, 2.4, 1.9), 640, 480, 96.0, 25.0, 90, 23),
    ]
}

/// What one run of the chain produced.
struct Outcome {
    /// RMS distance from the truth over every reconstructed joint, in metres.
    raw: f64,
    /// The same after the fit and the filter.
    filtered: f64,
    /// The same for the predicted position, against the truth one horizon
    /// ahead.
    predicted: f64,
    /// What the error against that same target would have been with no
    /// prediction at all — the smoothed position, handed over as though it were
    /// current. This is what the prediction has to beat to be worth having.
    stale: f64,
    ticks: usize,
    joints: usize,
    /// Ticks where a joint was placed by the fit rather than seen.
    inferred: usize,
    /// RMS error of the joint the cameras were made to lose, in metres.
    hidden_error: f64,
}

/// Runs the walk through the whole chain.
///
/// `honour_latency` is what the comparison hangs on: with it off, every camera
/// is treated as though it delivered instantly, which is what a fusion stage
/// that ignored the calibration's latency measurement would do.
fn walk(honour_latency: bool, hide: Option<Joint>) -> Outcome {
    const RATE: Duration = Duration::from_micros(16_667);
    const SLACK: Duration = Duration::from_millis(40);
    const SETTLE: Duration = Duration::from_secs(2);
    const LENGTH: Duration = Duration::from_secs(7);
    /// When the joint that gets hidden goes out of sight.
    const OCCLUDED_FROM: f64 = 4.0;

    let mut rigs = room();
    if let Some(joint) = hide {
        // Two of the three lose it, which is one short of what triangulating
        // anything needs. The fit is then the only thing that can place it.
        for rig in rigs.iter_mut().take(2) {
            rig.hidden = Some((joint, OCCLUDED_FROM));
        }
    }
    let cameras: Vec<Camera> = rigs.iter().map(|rig| rig.camera.clone()).collect();
    let lags: Vec<Duration> = rigs
        .iter()
        .map(|rig| {
            if honour_latency {
                rig.latency
            } else {
                Duration::ZERO
            }
        })
        .collect();
    let lag = lags.iter().copied().max().unwrap_or_default() + SLACK;

    let start = Instant::now() + Duration::from_secs(30);
    let mut noise = Noise(0xFA57_1234);

    let options = FuseOptions::default();
    let filter_options = FilterOptions::default();
    let mut meter = BoneMeter::default();
    let mut fitter = Fitter::default();
    let mut filter = PoseFilter::new(filter_options.clone());
    let mut skeleton = meter.finish();

    let mut outcome = Outcome {
        raw: 0.0,
        filtered: 0.0,
        predicted: 0.0,
        stale: 0.0,
        ticks: 0,
        joints: 0,
        inferred: 0,
        hidden_error: 0.0,
    };
    let (mut raw_sum, mut filtered_sum, mut predicted_sum) = (0.0, 0.0, 0.0);
    let mut stale_sum = 0.0;
    let mut counted = 0usize;
    let (mut hidden_sum, mut hidden_counted) = (0.0, 0usize);

    let mut now = start;
    let mut tick = 0u64;
    while now < start + LENGTH {
        now = start + RATE.mul_f64(tick as f64);
        tick += 1;

        for (index, rig) in rigs.iter_mut().enumerate() {
            let _ = index;
            rig.advance(start, now, &mut noise);
        }

        let Some(at) = now.checked_sub(lag) else {
            continue;
        };
        if at < start {
            continue;
        }

        let mut views = Vec::new();
        for (index, rig) in rigs.iter().enumerate() {
            let sampled_at = at + lags[index];
            let Some((before, after)) = rig.channel.bracket(sampled_at) else {
                continue;
            };
            let aligned = align(&before, &after, sampled_at);
            if !aligned.is_empty() {
                views.push((index, aligned));
            }
        }

        let elapsed = at.duration_since(start);
        let reconstruction = fuse(&cameras, &views, at, &options);

        meter.observe(&reconstruction);
        if tick.is_multiple_of(60) {
            skeleton = meter.finish();
        }

        let fitted = fitter.fit(&reconstruction, &skeleton);
        let smoothed = filter.push(&fitted);

        // The first couple of seconds are the filters and the measurement
        // settling, which is not what is being judged.
        if elapsed < SETTLE {
            continue;
        }

        let truth = body(elapsed.as_secs_f64());
        let ahead = body((elapsed + filter_options.horizon).as_secs_f64());

        for (joint, expected) in &truth {
            let Some(fused) = reconstruction.get(*joint) else {
                continue;
            };
            raw_sum += (fused.point - expected).norm_squared();

            if let Some(point) = smoothed.position(*joint) {
                filtered_sum += (point - expected).norm_squared();
            }
            if let Some((_, target)) = ahead.iter().find(|(name, _)| name == joint) {
                if let Some(point) = smoothed.predicted(*joint) {
                    predicted_sum += (point - target).norm_squared();
                }
                if let Some(point) = smoothed.position(*joint) {
                    stale_sum += (point - target).norm_squared();
                }
            }
            counted += 1;
        }

        // The joint the cameras were made to lose is judged on its own, after
        // it goes: it is the only one the fit has to place unaided.
        if let Some(joint) = hide.filter(|_| elapsed.as_secs_f64() > OCCLUDED_FROM + 0.5)
            && let (Some(placed), Some((_, expected))) = (
                fitted.get(joint),
                truth.iter().find(|(name, _)| *name == joint),
            )
        {
            if placed.inferred {
                outcome.inferred += 1;
            }
            hidden_sum += (placed.point - expected).norm_squared();
            hidden_counted += 1;
        }

        outcome.ticks += 1;
        outcome.joints = outcome.joints.max(reconstruction.count());
    }

    let counted = counted.max(1) as f64;
    outcome.raw = (raw_sum / counted).sqrt();
    outcome.filtered = (filtered_sum / counted).sqrt();
    outcome.predicted = (predicted_sum / counted).sqrt();
    outcome.stale = (stale_sum / counted).sqrt();
    outcome.hidden_error = (hidden_sum / hidden_counted.max(1) as f64).sqrt();
    outcome
}

#[test]
fn a_walk_past_three_unlike_cameras_is_reconstructed() {
    let outcome = walk(true, None);

    // Printed rather than only asserted: these four numbers are how every
    // change to the filter gets judged, and reading them off a passing run
    // beats provoking a failure to see them.
    eprintln!(
        "raw {:.1} cm  filtered {:.1} cm  predicted {:.1} cm  stale {:.1} cm",
        outcome.raw * 100.0,
        outcome.filtered * 100.0,
        outcome.predicted * 100.0,
        outcome.stale * 100.0
    );

    assert!(
        outcome.ticks > 200,
        "only {} ticks produced anything",
        outcome.ticks
    );
    assert_eq!(
        outcome.joints, 9,
        "every joint of the simulated body should come through"
    );
    assert!(
        outcome.raw < 0.01,
        "the reconstruction was off by {:.1} cm",
        outcome.raw * 100.0
    );
    // Larger than the reconstruction, and rightly so: this is the smoothing
    // deliberately trailing a body whose legs are moving at a couple of metres
    // per second. The prediction is what pays it back, and the test below is
    // where that is judged.
    assert!(
        outcome.filtered < 0.04,
        "the finished skeleton was off by {:.1} cm",
        outcome.filtered * 100.0
    );
}

/// Cameras hand their frames over late, and by different amounts. Fusing them
/// as though they had not is the failure this whole stage exists to avoid, so
/// it is worth showing that it would in fact be a failure.
#[test]
fn ignoring_the_camera_delays_is_much_worse() {
    let aligned = walk(true, None);
    let ignored = walk(false, None);

    assert!(
        ignored.raw > 5.0 * aligned.raw,
        "aligned {:.1} cm against ignored {:.1} cm",
        aligned.raw * 100.0,
        ignored.raw * 100.0
    );
}

/// Everything here is late. The prediction is what makes the output land where
/// the body will be rather than where it was.
#[test]
fn the_prediction_lands_ahead_of_the_measurement() {
    let outcome = walk(true, None);

    assert!(
        outcome.predicted < 0.07,
        "the prediction was off by {:.1} cm",
        outcome.predicted * 100.0
    );
    // The comparison that matters: handing over the current answer as though
    // it were the future one, which is what no prediction looks like.
    //
    // The bar was half and is now three fifths, which is a real loss and worth
    // naming. Weighing each velocity against how well it is known costs about
    // two centimetres here — this walk is simulated with millimetre-accurate
    // joints, so its velocities are far better determined than a real room's,
    // and the caution buys least exactly where it is measured. What it buys is
    // a body that does not vibrate when it is standing still, which is the
    // state a user spends most of their time in and the one that made the first
    // build unusable. See `filter::tests::a_still_joint_is_predicted_still`.
    assert!(
        outcome.predicted < 0.6 * outcome.stale,
        "predicted {:.1} cm against {:.1} cm with no prediction",
        outcome.predicted * 100.0,
        outcome.stale * 100.0
    );
}

/// The body is assembled from unit vectors, so its bones are exactly constant.
/// Recovering them from the cameras is what the fit rests on.
#[test]
fn the_bones_are_measured_from_the_walk() {
    use optra::fusion::bones::Bone;

    let mut rigs = room();
    let cameras: Vec<Camera> = rigs.iter().map(|rig| rig.camera.clone()).collect();
    let start = Instant::now() + Duration::from_secs(30);
    let mut noise = Noise(0x1234_5678);
    let mut meter = BoneMeter::default();

    for tick in 0..360 {
        let now = start + Duration::from_micros(16_667).mul_f64(tick as f64);
        for rig in rigs.iter_mut() {
            rig.advance(start, now, &mut noise);
        }

        let Some(at) = now.checked_sub(Duration::from_millis(130)) else {
            continue;
        };
        let mut views = Vec::new();
        for (index, rig) in rigs.iter().enumerate() {
            let sampled_at = at + rig.latency;
            if let Some((before, after)) = rig.channel.bracket(sampled_at) {
                views.push((index, align(&before, &after, sampled_at)));
            }
        }
        meter.observe(&fuse(&cameras, &views, at, &FuseOptions::default()));
    }

    let skeleton = meter.finish();
    for (bone, truth) in [
        (Bone::new(Joint::LeftHip, Joint::LeftKnee), THIGH),
        (Bone::new(Joint::LeftKnee, Joint::LeftAnkle), SHIN),
        (Bone::new(Joint::LeftHip, Joint::RightHip), 2.0 * HALF_HIPS),
        (Bone::new(Joint::Hip, Joint::Neck), SPINE),
        (Bone::new(Joint::Neck, Joint::Head), NECK),
    ] {
        let measured = skeleton
            .length(bone)
            .unwrap_or_else(|| panic!("{} was never measured", bone.label()));
        assert!(
            (measured - truth).abs() < 0.01,
            "{} measured {measured:.3} m against a true {truth:.3} m",
            bone.label()
        );
    }
}

/// A limb that goes behind something is the normal case in a real room, not an
/// edge case. Two of the three cameras lose the left knee partway through,
/// leaving one — which is one short of what triangulating anything needs — and
/// the fit has to place it from the joints either side.
#[test]
fn a_knee_that_goes_behind_something_is_still_placed() {
    let occluded = walk(true, Some(Joint::LeftKnee));

    assert!(
        occluded.inferred > 100,
        "the knee was only inferred on {} ticks",
        occluded.inferred
    );
    assert!(
        occluded.hidden_error < 0.06,
        "the inferred knee was off by {:.1} cm",
        occluded.hidden_error * 100.0
    );

    // And losing a joint must not cost the rest of the body anything.
    let clear = walk(true, None);
    assert!(
        occluded.raw < clear.raw * 1.5,
        "the rest of the body degraded from {:.1} cm to {:.1} cm",
        clear.raw * 100.0,
        occluded.raw * 100.0
    );
}

/// Repeatable standard normals, so a threshold means the same thing on every
/// machine and every run.
struct Noise(u64);

impl Noise {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}
