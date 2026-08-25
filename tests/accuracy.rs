//! How far out is the skeleton, measured from the pixels?
//!
//! Every other test in this project starts after inference. `tests/fusion.rs`
//! projects a known body into each camera and asks what the chain downstream
//! does with perfect keypoints; that is the right question for the fusion
//! stage, and it cannot answer the one that decides whether the application
//! works at all — given a *picture* of a person, how far out is the knee that
//! comes back?
//!
//! So this renders a walk in a simulated room from four unlike ceiling cameras
//! ([`optra::sim`]), runs the real ONNX detector and pose model on those
//! pixels, and puts the result through the real triangulation, the real
//! skeleton fit and the real filter. The figure was built by forward kinematics
//! from stated bone lengths, so the answer is known exactly.
//!
//! The report separates four things that a single error figure runs together:
//!
//! - **pixels** — how far each camera's keypoint is from where that joint
//!   really projects. This is the pose model and nothing else, reported in
//!   pixels and in angle, since a pixel is worth several times the angle on a
//!   wide 480p camera that it is on a narrow 1080p one.
//! - **bias** — the part of the 3D error that is constant in the walking
//!   body's own frame. A pose model's joints are where its training set was
//!   annotated, which is not where the bone is; Halpe's "head" is not the
//!   centre of the skull. That shows up here as several centimetres in a fixed
//!   direction, and it is a labelling convention rather than a failure. The
//!   output stage's per-tracker offsets exist to absorb exactly this.
//! - **spread** — what is left once the bias is removed. Nothing downstream can
//!   absorb it, so this is the number the assertions are written against.
//! - **swaps** — ticks where a joint came back nearer to its mirror image than
//!   to itself. A foot tracker on the wrong foot is not a slightly worse foot
//!   tracker, and averaging that into a tail would hide it.
//!
//! Bone lengths are reported alongside, measured against the lengths that were
//! drawn. A body reconstructed to the wrong size is the one error a set of
//! cameras cannot detect about itself.
//!
//! The cameras are deliberately synchronous here. Temporal alignment has its
//! own test and its own failure mode, and mixing the two would leave a number
//! that could move for either reason.
//!
//! Running it:
//!
//! ```text
//! cargo test --release --test accuracy -- --ignored --nocapture
//! ```
//!
//! It is ignored by default because it downloads a few hundred megabytes of
//! model. The harness itself is exercised on every `cargo test` by the same
//! walk with the keypoints projected rather than inferred, which is what stops
//! it rotting between the runs that matter.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::time::{Duration, Instant};

use nalgebra::Point2;

use optra::fusion::align::align;
use optra::fusion::bones::{BONES, Bone, BoneMeter, MeasureOptions, Skeleton};
use optra::fusion::filter::{FilterOptions, PoseFilter};
use optra::fusion::fit::{FitOptions, Fitter};
use optra::fusion::fuse::{FuseOptions, Pose3d, fuse};
use optra::geometry::camera::Camera;
use optra::infer::session::ProviderChoice;
use optra::infer::traits::{Detector, ImageView, Keypoint, Keypoints2d, Pose2d};
use optra::models::keypoints::Joint;
use optra::models::manifest::Manifest;
use optra::models::{ModelSpec, store};
use optra::pipeline::PoseFrame;
use optra::sim::body::Posture;
use optra::sim::{Rng, Scene};

/// How many cameras hang in the room.
const CAMERAS: usize = 4;
/// Fusion ticks per second. The cameras are synchronous, so this is also the
/// frame rate of every one of them.
const RATE: f64 = 20.0;
/// How long the walk lasts, in seconds.
///
/// Long enough that *half* of it settles the skeleton, because the halves are
/// used for different things: the bones are measured from the first and the fit
/// is scored on the second. Seven seconds was enough when one pass did both,
/// and three and a half is not — the measurement came back with 82% of the
/// skeleton. Twelve is where it returns to all of it, and this leaves a margin,
/// since a real pose model drops joints and the meter gets fewer samples per
/// second than the projected walk hands it.
const LENGTH: f64 = 14.0;

// ---------------------------------------------------------------------------
// Where the keypoints come from
// ---------------------------------------------------------------------------

/// A source of 2D keypoints for one instant of the walk.
///
/// Two implementations, and the whole point of the harness is that everything
/// downstream cannot tell them apart: [`Projected`] hands over the truth with a
/// little noise on it, and [`Models`] renders the frame and asks a pose model.
trait Eyes {
    fn label(&self) -> String;

    /// What each camera reports at `t`. The result matches `cameras` in order,
    /// and a camera that saw nothing contributes an empty entry.
    fn look(&mut self, scene: &Scene, cameras: &[Camera], t: f64) -> Vec<Keypoints2d>;
}

/// The truth, projected, with a pixel of noise on it.
///
/// This is what `tests/fusion.rs` feeds its chain, and having it here as well
/// is what makes the harness's own arithmetic testable without a model: if the
/// numbers below are bad for `Projected`, the fault is in the harness or the
/// room, not in inference.
struct Projected {
    noise: f64,
    rng: Rng,
}

impl Eyes for Projected {
    fn label(&self) -> String {
        format!("projected truth, {:.1} px of noise", self.noise)
    }

    fn look(&mut self, scene: &Scene, cameras: &[Camera], t: f64) -> Vec<Keypoints2d> {
        let posture = scene.posture(t);
        cameras
            .iter()
            .map(|camera| {
                let mut keypoints = Keypoints2d::default();
                for (joint, point) in posture.iter() {
                    let Some(pixel) = camera.project(point) else {
                        continue;
                    };
                    let x = pixel.x + self.rng.normal() * self.noise;
                    let y = pixel.y + self.rng.normal() * self.noise;
                    if !inside(camera, x, y) {
                        continue;
                    }
                    keypoints.set(
                        joint,
                        Keypoint {
                            x: x as f32,
                            y: y as f32,
                            confidence: 0.9,
                        },
                    );
                }
                keypoints
            })
            .collect()
    }
}

/// The real thing: render each camera's view and run detection and pose
/// estimation over the batch, exactly as the inference stage does.
struct Models {
    detector_id: String,
    pose_id: String,
    detector: Box<dyn Detector>,
    pose: Box<dyn Pose2d>,
}

impl Eyes for Models {
    fn label(&self) -> String {
        format!("{} + {}", self.detector_id, self.pose_id)
    }

    fn look(&mut self, scene: &Scene, cameras: &[Camera], t: f64) -> Vec<Keypoints2d> {
        let images: Vec<_> = cameras.iter().map(|camera| scene.view(camera, t)).collect();
        let views: Vec<ImageView<'_>> = images.iter().map(|image| image.view()).collect();

        let found = self.detector.detect(&views).expect("detection should run");
        let mut people = Vec::new();
        let mut seats = Vec::new();
        for (seat, detections) in found.iter().enumerate() {
            if let Some(person) = detections
                .iter()
                .max_by(|a, b| a.score.total_cmp(&b.score))
                .copied()
            {
                people.push((views[seat], person));
                seats.push(seat);
            }
        }

        let estimated = if people.is_empty() {
            Vec::new()
        } else {
            self.pose.estimate(&people).expect("pose should run")
        };

        let mut keypoints = vec![Keypoints2d::default(); cameras.len()];
        for (seat, found) in seats.into_iter().zip(estimated) {
            keypoints[seat] = found;
        }
        keypoints
    }
}

fn inside(camera: &Camera, x: f64, y: f64) -> bool {
    x >= 0.0
        && y >= 0.0
        && x < camera.intrinsics.width as f64
        && y < camera.intrinsics.height as f64
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

/// The value `fraction` of the way through `values`, once sorted.
///
/// One definition, used by everything here. Two medians that disagree by one
/// sample on an even-sized column are two numbers that cannot be compared, and
/// this file prints them side by side.
fn quantile(values: &[f64], fraction: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[(((sorted.len() - 1) as f64) * fraction).round() as usize]
}

/// A column of errors, kept so that the median and the tail can be reported.
///
/// The mean on its own is the wrong summary for this. One frame where the pose
/// model puts an ankle on the other foot moves the mean by more than a hundred
/// good frames move it back, and a mean that says four centimetres over a
/// median of one is describing two different populations as though they were
/// one.
#[derive(Debug, Default, Clone)]
struct Tally {
    errors: Vec<f64>,
    /// Occasions the value could have been measured and was not.
    absent: usize,
}

impl Tally {
    fn add(&mut self, error: f64) {
        self.errors.push(error);
    }

    fn miss(&mut self) {
        self.absent += 1;
    }

    fn count(&self) -> usize {
        self.errors.len()
    }

    fn quantile(&self, fraction: f64) -> f64 {
        quantile(&self.errors, fraction)
    }

    fn median(&self) -> f64 {
        self.quantile(0.5)
    }

    fn p95(&self) -> f64 {
        self.quantile(0.95)
    }

    fn worst(&self) -> f64 {
        self.errors.iter().copied().fold(0.0, f64::max)
    }

    /// The fraction of chances that produced a value at all.
    fn presence(&self) -> f64 {
        let total = self.count() + self.absent;
        if total == 0 {
            0.0
        } else {
            self.count() as f64 / total as f64
        }
    }
}

/// A joint's 3D error, kept as a vector in the walking body's own frame.
///
/// The magnitude alone hides the distinction that matters most here. A pose
/// model's idea of where a joint is comes from how its training set was
/// annotated, and that is not the same as where the bone is: Halpe's "head" is
/// not the centre of the skull and its "neck" is not the top of the spine. The
/// result is an offset that is *constant in the body's frame* — nine
/// centimetres, every frame, in the same direction relative to the person —
/// which is a labelling difference and not an error at all, and which the
/// output stage's per-tracker offsets exist to absorb.
///
/// So the offsets are decomposed across, up and along the body, the median of
/// each is taken as the bias, and what is left over is the spread. The spread
/// is the part no convention and no offset can fix, and it is what the
/// assertions are written against.
#[derive(Debug, Default, Clone)]
struct Spatial {
    /// Error as `(across, up, along)`, in metres, relative to the body.
    offsets: Vec<[f64; 3]>,
    absent: usize,
}

impl Spatial {
    fn add(&mut self, offset: [f64; 3]) {
        self.offsets.push(offset);
    }

    fn miss(&mut self) {
        self.absent += 1;
    }

    /// The constant part, componentwise. Median rather than mean, so that a
    /// handful of frames with a limb on the wrong side does not move it.
    fn bias(&self) -> [f64; 3] {
        std::array::from_fn(|axis| {
            let column: Vec<f64> = self.offsets.iter().map(|offset| offset[axis]).collect();
            quantile(&column, 0.5)
        })
    }

    fn bias_distance(&self) -> f64 {
        let bias = self.bias();
        (bias[0] * bias[0] + bias[1] * bias[1] + bias[2] * bias[2]).sqrt()
    }

    /// Total distance from the truth, which is bias and spread together.
    fn distance(&self) -> Tally {
        Tally {
            errors: self
                .offsets
                .iter()
                .map(|o| (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt())
                .collect(),
            absent: self.absent,
        }
    }

    /// What is left once the constant part is taken out.
    fn spread(&self) -> Tally {
        let bias = self.bias();
        Tally {
            errors: self
                .offsets
                .iter()
                .map(|o| {
                    let d: [f64; 3] = std::array::from_fn(|axis| o[axis] - bias[axis]);
                    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
                })
                .collect(),
            absent: self.absent,
        }
    }
}

/// One bone, as drawn and as measured.
///
/// `measured` is `None` for a bone whose samples never settled — too few of
/// them, or too scattered to be worth holding a body to. That is a result, not
/// a gap: a short bone between two joints each located to half a centimetre
/// cannot be measured to better than its own tolerance, and the meter declining
/// to guess is the behaviour being checked.
#[derive(Debug, Clone, Copy)]
struct BoneScore {
    truth: f64,
    measured: Option<f64>,
    samples: usize,
    scatter: f64,
}

impl BoneScore {
    fn error(&self) -> Option<f64> {
        self.measured.map(|length| (length - self.truth).abs())
    }
}

/// Everything one walk produced.
#[derive(Debug, Default)]
struct Report {
    label: String,
    /// Distance from a reported keypoint to where the joint really projects, in
    /// pixels, over every camera.
    pixels: BTreeMap<Joint, Tally>,
    /// The same in angle, in millidegrees, which is the comparison that means
    /// the same thing on a 480p wide camera and a 1080p narrow one.
    rays: BTreeMap<Joint, Tally>,
    /// Where the triangulated joint is relative to the truth, in metres.
    fused: BTreeMap<Joint, Spatial>,
    /// The same after the skeleton fit.
    fitted: BTreeMap<Joint, Spatial>,
    /// The same after smoothing.
    smoothed: BTreeMap<Joint, Spatial>,
    /// Ticks where a joint was reconstructed nearer to its mirror image than to
    /// itself, which is the pose model putting a limb on the wrong side.
    swapped: BTreeMap<Joint, usize>,
    /// Measured bone length against the length that was drawn, in metres.
    bones: BTreeMap<Bone, BoneScore>,
    /// Fraction of the skeleton the measurement settled on.
    bone_coverage: f32,
    /// Ticks that produced a reconstruction.
    ticks: usize,
    /// Camera frames where nothing at all came back — the detector found
    /// nobody, or the person was outside the frame. A different failure from a
    /// badly placed keypoint, and it has to be reported as one.
    ///
    /// Counted in [`walk`] from what each camera handed over rather than inside
    /// an [`Eyes`], so that both implementations are counted the same way and
    /// the table cannot claim a run had no frames in it.
    blind: usize,
    frames: usize,
}

impl Report {
    fn over(&self, tallies: &BTreeMap<Joint, Tally>, only: fn(Joint) -> bool) -> Tally {
        let mut all = Tally::default();
        for (joint, tally) in tallies {
            if only(*joint) {
                all.errors.extend(&tally.errors);
                all.absent += tally.absent;
            }
        }
        all
    }

    /// The summary that matters: the joints the lower-body trackers are built
    /// from. An elbow being wrong costs this application nothing.
    fn lower_body(&self, tallies: &BTreeMap<Joint, Tally>) -> Tally {
        self.over(tallies, Joint::is_lower_body)
    }

    /// Total distance from the truth, per joint.
    fn distances(&self, stage: &BTreeMap<Joint, Spatial>) -> BTreeMap<Joint, Tally> {
        stage
            .iter()
            .map(|(joint, spatial)| (*joint, spatial.distance()))
            .collect()
    }

    /// What is left once each joint's constant offset is removed.
    fn spreads(&self, stage: &BTreeMap<Joint, Spatial>) -> BTreeMap<Joint, Tally> {
        stage
            .iter()
            .map(|(joint, spatial)| (*joint, spatial.spread()))
            .collect()
    }
}

/// Renders the report as a table, because these numbers are read far more often
/// than they are asserted on.
fn table(report: &Report) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "\n{}", report.label);
    let _ = writeln!(
        out,
        "{} ticks, {} camera frames, {} with nobody found",
        report.ticks, report.frames, report.blind
    );

    let _ = writeln!(
        out,
        "\n{:<16} {:>7} {:>7} {:>6} | {:>6} {:>7} {:>7} {:>7} | {:>7} {:>6} {:>6}",
        "joint",
        "px p50",
        "px p95",
        "mdeg",
        "bias",
        "sprd p50",
        "sprd p95",
        "worst",
        "fit p50",
        "swap",
        "seen"
    );
    let _ = writeln!(out, "{}", "-".repeat(100));

    for joint in Joint::ALL {
        let pixels = report.pixels.get(&joint).cloned().unwrap_or_default();
        let rays = report.rays.get(&joint).cloned().unwrap_or_default();
        let fused = report.fused.get(&joint).cloned().unwrap_or_default();
        let fitted = report.fitted.get(&joint).cloned().unwrap_or_default();
        if pixels.count() == 0 && fused.offsets.is_empty() {
            continue;
        }
        let spread = fused.spread();

        let _ = writeln!(
            out,
            "{:<16} {:>7.1} {:>7.1} {:>6.0} | {:>6.1} {:>8.1} {:>8.1} {:>7.1} | {:>7.1} {:>6} {:>5.0}%",
            joint.name(),
            pixels.median(),
            pixels.p95(),
            rays.median(),
            fused.bias_distance() * 100.0,
            spread.median() * 100.0,
            spread.p95() * 100.0,
            fused.distance().worst() * 100.0,
            fitted.spread().median() * 100.0,
            report.swapped.get(&joint).copied().unwrap_or(0),
            fused.distance().presence() * 100.0
        );
    }
    let _ = writeln!(
        out,
        "\nbias is the constant part of the error in the body's own frame — a \
         labelling convention,\nnot a failure. spread is what is left, and it \
         is the part nothing downstream can absorb.\ndistances are centimetres."
    );

    for (name, only) in [
        ("all joints", (|_| true) as fn(Joint) -> bool),
        ("lower body", Joint::is_lower_body),
    ] {
        let distance = report.over(&report.distances(&report.fused), only);
        let spread = report.over(&report.spreads(&report.fused), only);
        let _ = writeln!(
            out,
            "\n{name}: from the truth, median {:.1} cm, p95 {:.1} cm; \
             spread alone, median {:.1} cm, p95 {:.1} cm; present {:.0}%",
            distance.median() * 100.0,
            distance.p95() * 100.0,
            spread.median() * 100.0,
            spread.p95() * 100.0,
            distance.presence() * 100.0
        );
    }
    let fitted = report.lower_body(&report.spreads(&report.fitted));
    let smoothed = report.lower_body(&report.spreads(&report.smoothed));
    let _ = writeln!(
        out,
        "lower body spread after the fit: median {:.1} cm, p95 {:.1} cm; \
         after smoothing: median {:.1} cm",
        fitted.median() * 100.0,
        fitted.p95() * 100.0,
        smoothed.median() * 100.0
    );

    let _ = writeln!(
        out,
        "\n{:<32} {:>8} {:>8} {:>8} {:>9} {:>8}",
        "bone", "drawn cm", "meas cm", "error cm", "scatter %", "samples"
    );
    let _ = writeln!(out, "{}", "-".repeat(80));
    for (bone, score) in &report.bones {
        match score.measured {
            Some(measured) => {
                let _ = writeln!(
                    out,
                    "{:<32} {:>8.1} {:>8.1} {:>8.1} {:>9.1} {:>8}",
                    bone.label(),
                    score.truth * 100.0,
                    measured * 100.0,
                    score.error().unwrap_or(f64::NAN) * 100.0,
                    100.0 * score.scatter / measured.max(1e-9),
                    score.samples
                );
            }
            None => {
                let _ = writeln!(
                    out,
                    "{:<32} {:>8.1} {:>8} {:>8} {:>9.1} {:>8}",
                    bone.label(),
                    score.truth * 100.0,
                    "-",
                    "unsettled",
                    100.0 * score.scatter / score.truth.max(1e-9),
                    score.samples
                );
            }
        }
    }
    let _ = writeln!(
        out,
        "\n{:.0}% of the skeleton settled",
        report.bone_coverage * 100.0
    );

    out
}

// ---------------------------------------------------------------------------
// The walk
// ---------------------------------------------------------------------------

/// Runs one walk end to end and scores it.
///
/// The walk is split in two. The skeleton is measured over the first half and
/// the fit and the filter are scored over the second, so that no tick is ever
/// held to a skeleton its own reconstruction helped to measure.
///
/// The split is the point. A real session measures a body over minutes and then
/// goes on fitting to it, so measuring first is the realistic order — but
/// measuring over the *whole* walk and then fitting the same frames again, as
/// this used to, quietly hands each frame a skeleton that already knows about
/// it. The effect is small, because a bone length is a median over thousands of
/// samples and one pose barely moves it, but "small" is a claim and this is
/// cheaper than defending it.
fn walk(eyes: &mut dyn Eyes, scene: &Scene) -> Report {
    let cameras = scene.cameras(CAMERAS);
    let start = Instant::now() + Duration::from_secs(60);
    let step = Duration::from_secs_f64(1.0 / RATE);
    let ticks = (LENGTH * RATE) as usize;

    let mut report = Report {
        label: eyes.label(),
        ..Report::default()
    };
    let mut reconstructions: Vec<(f64, Pose3d)> = Vec::with_capacity(ticks);
    let fuse_options = FuseOptions::default();

    for tick in 0..ticks {
        let t = tick as f64 / RATE;
        let at = start + step.mul_f64(tick as f64);
        let posture = scene.posture(t);
        let seen = eyes.look(scene, &cameras, t);

        let mut views = Vec::new();
        for (seat, keypoints) in seen.iter().enumerate() {
            score_pixels(&mut report, &cameras[seat], &posture, keypoints);
            report.frames += 1;
            if keypoints.is_empty() {
                report.blind += 1;
                continue;
            }

            // The cameras are synchronous here, so the tick lands exactly on a
            // frame and the alignment is an identity. It still goes through
            // `align`, because that is the type `fuse` consumes and writing a
            // second way to build one would be a second thing to keep right.
            let frame = PoseFrame {
                seq: tick as u64,
                captured_at: at,
                width: cameras[seat].intrinsics.width,
                height: cameras[seat].intrinsics.height,
                detection: None,
                keypoints: keypoints.clone(),
            };
            let aligned = align(&frame, &frame, at);
            if !aligned.is_empty() {
                views.push((seat, aligned));
            }
        }

        let reconstruction = fuse(&cameras, &views, at, &fuse_options);
        if reconstruction.is_empty() {
            continue;
        }
        report.ticks += 1;
        score_positions(&mut report.fused, &posture, |joint| {
            reconstruction.get(joint).map(|fused| fused.point)
        });
        score_swaps(&mut report.swapped, &posture, &reconstruction);
        reconstructions.push((t, reconstruction));
    }

    let split = reconstructions.len() / 2;
    let measure = MeasureOptions::default();
    let mut meter = BoneMeter::new(measure.clone());
    for (_, reconstruction) in &reconstructions[..split] {
        meter.observe(reconstruction);
    }
    let skeleton = meter.finish();
    score_bones(&mut report, &skeleton, scene, &measure);

    let mut fitter = Fitter::new(FitOptions::default());
    let mut filter = PoseFilter::new(FilterOptions::default());
    for (t, reconstruction) in &reconstructions[split..] {
        let posture = scene.posture(*t);
        let fitted = fitter.fit(reconstruction, &skeleton);
        let smoothed = filter.push(&fitted);

        score_positions(&mut report.fitted, &posture, |joint| fitted.position(joint));
        score_positions(&mut report.smoothed, &posture, |joint| {
            smoothed.position(joint)
        });
    }

    report
}

/// How far each reported keypoint is from where its joint really projects.
fn score_pixels(report: &mut Report, camera: &Camera, posture: &Posture, keypoints: &Keypoints2d) {
    for (joint, point) in posture.iter() {
        let Some(expected) = camera.project(point).filter(|p| inside(camera, p.x, p.y)) else {
            continue;
        };
        match keypoints.get(joint) {
            Some(found) => {
                let observed = Point2::new(found.x as f64, found.y as f64);
                report
                    .pixels
                    .entry(joint)
                    .or_default()
                    .add((observed - expected).norm());
                if let Some(angle) = camera.angular_error(point, observed) {
                    report
                        .rays
                        .entry(joint)
                        .or_default()
                        .add(angle.to_degrees() * 1000.0);
                }
            }
            None => {
                report.pixels.entry(joint).or_default().miss();
                report.rays.entry(joint).or_default().miss();
            }
        }
    }
}

/// Records where each reconstructed joint is relative to the truth, decomposed
/// in the body's own frame so that a constant offset can be told from noise.
fn score_positions(
    stage: &mut BTreeMap<Joint, Spatial>,
    posture: &Posture,
    reconstructed: impl Fn(Joint) -> Option<nalgebra::Point3<f64>>,
) {
    for (joint, expected) in posture.iter() {
        let spatial = stage.entry(joint).or_default();
        match reconstructed(joint) {
            Some(point) => {
                let error = point - expected;
                spatial.add([
                    error.dot(&posture.right),
                    error.y,
                    error.dot(&posture.facing),
                ]);
            }
            None => spatial.miss(),
        }
    }
}

/// Counts the joints that came out nearer to their mirror image than to
/// themselves.
///
/// This is a failure with a name, and it deserves to be counted rather than
/// averaged into a tail: a pose model looking down at somebody from behind has
/// nothing much to tell their left leg from their right, and a foot tracker on
/// the wrong foot is not a slightly worse foot tracker.
fn score_swaps(swapped: &mut BTreeMap<Joint, usize>, posture: &Posture, pose: &Pose3d) {
    for joint in Joint::ALL {
        let mirror = joint.mirror();
        if mirror == joint {
            continue;
        }
        let (Some(here), Some(there)) = (posture.get(joint), posture.get(mirror)) else {
            continue;
        };
        let Some(found) = pose.get(joint) else {
            continue;
        };

        if (found.point - there).norm() < (found.point - here).norm() {
            *swapped.entry(joint).or_default() += 1;
        }
    }
}

/// The measured skeleton against the one that was drawn.
fn score_bones(report: &mut Report, skeleton: &Skeleton, scene: &Scene, options: &MeasureOptions) {
    // Both sides of the body are drawn to the same length and every bone is
    // rigid, so the truth for a bone is the distance between its two ends in
    // any posture at all.
    let posture = scene.posture(0.0);
    report.bone_coverage = skeleton.coverage(options);

    for bone in BONES {
        let (Some(from), Some(to)) = (posture.get(bone.from), posture.get(bone.to)) else {
            continue;
        };
        let found = skeleton.get(*bone);
        report.bones.insert(
            *bone,
            BoneScore {
                truth: (to - from).norm(),
                measured: found
                    .filter(|measured| measured.is_settled(options))
                    .map(|measured| measured.length),
                samples: found.map_or(0, |measured| measured.samples),
                scatter: found.map_or(f64::NAN, |measured| measured.scatter),
            },
        );
    }
}

fn spec(id: &str) -> ModelSpec {
    Manifest::load()
        .expect("the catalogue")
        .into_iter()
        .find(|spec| spec.id == id)
        .unwrap_or_else(|| panic!("{id} is in the catalogue"))
}

// ---------------------------------------------------------------------------
// The tests
// ---------------------------------------------------------------------------

/// The harness against itself. Nothing here is inferred: the keypoints are the
/// truth with a pixel of noise, so what this measures is the room, the camera
/// set, the triangulation and the fit — everything the ignored test below has
/// to be able to take for granted before its numbers mean anything.
///
/// A pixel of noise is roughly what a good pose model contributes on a clear
/// view, so this is also the floor: no amount of model improvement gets the
/// ignored test below this.
#[test]
fn a_simulated_walk_is_reconstructed_from_perfect_keypoints() {
    let scene = Scene::default();
    let mut eyes = Projected {
        noise: 1.0,
        rng: Rng::new(0x0F7A_ACC0),
    };
    let report = walk(&mut eyes, &scene);
    eprint!("{}", table(&report));

    assert!(
        report.ticks as f64 > LENGTH * RATE * 0.95,
        "only {} of {} ticks reconstructed anything",
        report.ticks,
        (LENGTH * RATE) as usize
    );

    let lower = report.lower_body(&report.distances(&report.fused));
    assert!(
        lower.presence() > 0.98,
        "only {:.0}% of the lower body came through",
        lower.presence() * 100.0
    );
    assert!(
        lower.median() < 0.01,
        "the lower body was off by {:.1} cm",
        lower.median() * 100.0
    );
    assert!(
        lower.p95() < 0.02,
        "the worst 5% of the lower body was off by {:.1} cm",
        lower.p95() * 100.0
    );
    assert!(
        report.swapped.is_empty(),
        "a limb came back on the wrong side of the body: {:?}",
        report.swapped
    );

    // Every bone, measured, and measured right. This is a stronger claim than
    // it looks: it is what caught the simulated foot being hung off the floor
    // rather than off the ankle, which made the ankle-to-heel distance rise
    // and fall through every stride. The meter reported forty per cent scatter
    // on a bone that has none, and it was the body that was wrong.
    for (bone, score) in &report.bones {
        let error = score.error().unwrap_or_else(|| {
            panic!(
                "{} never settled: {:.0}% scatter over {} samples",
                bone.label(),
                100.0 * score.scatter / score.truth.max(1e-9),
                score.samples
            )
        });
        assert!(
            error < 0.02,
            "{} was measured {:.1} cm from the {:.1} cm it was drawn",
            bone.label(),
            error * 100.0,
            score.truth * 100.0
        );
    }
    assert!(
        report.bone_coverage > 0.99,
        "only {:.0}% of the skeleton settled over a {LENGTH} second walk",
        report.bone_coverage * 100.0
    );
}

/// A single camera cannot place a joint at all.
///
/// This is a guard rather than a measurement, and worth being clear about:
/// `fuse` drops any joint fewer than two cameras offered, so the first half
/// cannot fail unless somebody removes that rule. It is here because removing
/// it is an easy thing to do by accident — one ray fixes a direction and a
/// depth pulled out of nothing, and a chain that accepts it would still produce
/// plausible-looking numbers everywhere above.
#[test]
fn one_camera_cannot_place_a_body_and_four_can() {
    let scene = Scene::default();
    let cameras = scene.cameras(CAMERAS);

    let at = Instant::now() + Duration::from_secs(60);

    let mut eyes = Projected {
        noise: 1.0,
        rng: Rng::new(1),
    };
    let seen = eyes.look(&scene, &cameras, 2.0);

    let views: Vec<_> = seen
        .iter()
        .enumerate()
        .map(|(seat, keypoints)| {
            let frame = PoseFrame {
                seq: 0,
                captured_at: at,
                width: cameras[seat].intrinsics.width,
                height: cameras[seat].intrinsics.height,
                detection: None,
                keypoints: keypoints.clone(),
            };
            (seat, align(&frame, &frame, at))
        })
        .collect();

    let alone = fuse(&cameras, &views[..1], at, &FuseOptions::default());
    let together = fuse(&cameras, &views, at, &FuseOptions::default());

    assert!(
        alone.is_empty(),
        "one camera reconstructed {} joints, which it cannot have measured",
        alone.count()
    );
    assert!(together.count() >= Joint::ALL.len() - 2);
}

/// The measurement this whole file exists for.
///
/// Prints the table and asserts only what would be a genuine regression rather
/// than a bad afternoon for the model: that a person is found at all, that the
/// lower body comes through, and that the joints the trackers are built from
/// land within a few centimetres. The tight numbers to compare against are the
/// ones the test above produces from perfect keypoints — the gap between the
/// two tables *is* the contribution of inference, which is the thing that could
/// not previously be looked at.
#[test]
#[ignore = "downloads models"]
fn the_pose_models_reconstruct_the_body_from_rendered_frames() {
    const DETECTOR: &str = "yolox-tiny-humanart-416";
    const POSE: &str = "rtmpose-m-halpe26-256x192";

    for id in [DETECTOR, POSE] {
        store::install(&spec(id), &mut |_| {}).expect("the model should install");
    }

    let provider = ProviderChoice::DirectMl;
    let mut eyes = Models {
        detector_id: DETECTOR.to_owned(),
        pose_id: POSE.to_owned(),
        detector: optra::infer::arch::build_detector(&spec(DETECTOR), provider)
            .expect("the detector should load"),
        pose: optra::infer::arch::build_pose2d(&spec(POSE), provider)
            .expect("the pose model should load"),
    };

    let scene = Scene::default();
    let started = Instant::now();
    let report = walk(&mut eyes, &scene);

    eprint!("{}", table(&report));
    eprintln!(
        "\n{:.1} s for {} ticks over {CAMERAS} cameras",
        started.elapsed().as_secs_f64(),
        report.ticks
    );

    assert!(
        report.blind * 20 < report.frames,
        "the detector found nobody in {} of {} frames",
        report.blind,
        report.frames
    );
    assert!(
        report.ticks as f64 > LENGTH * RATE * 0.9,
        "only {} ticks produced a reconstruction",
        report.ticks
    );

    let lower = report.lower_body(&report.distances(&report.fused));
    assert!(
        lower.presence() > 0.85,
        "only {:.0}% of the lower body came through",
        lower.presence() * 100.0
    );
    assert!(
        lower.median() < 0.06,
        "the lower body was off by {:.1} cm",
        lower.median() * 100.0
    );

    // The spread is the number a better annotation convention or a per-tracker
    // offset cannot rescue, so it is held to more than the distance is.
    let spread = report.lower_body(&report.spreads(&report.fused));
    assert!(
        spread.median() < 0.04,
        "the lower body scattered by {:.1} cm about its own offset",
        spread.median() * 100.0
    );

    let hips = report
        .fused
        .get(&Joint::Hip)
        .map(|spatial| spatial.distance().median())
        .unwrap_or(f64::NAN);
    assert!(
        hips < 0.08,
        "the hip tracker would be {:.1} cm out",
        hips * 100.0
    );
}
