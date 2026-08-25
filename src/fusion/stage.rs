//! The fusion thread.
//!
//! One loop on a fixed clock, running every stage of this module in order:
//! resample each camera onto the tick, triangulate, measure the body,
//! hold the body to the measurement, filter, publish.
//!
//! The clock deliberately runs behind real time. Interpolating a camera onto an
//! instant needs a frame *after* that instant, so the tick sits far enough back
//! that even the latest camera has already delivered the frame that follows it.
//! The lag is not a loss: the prediction stage has to compensate for a delay
//! several times larger anyway, and predicting forward from a properly aligned
//! reconstruction beats fusing rays taken at three different instants.
//!
//! How far back that is is measured rather than configured. It depends on the
//! model each camera is running, the resolution it is running at and what else
//! is on the GPU, none of which is known when the thread starts, and getting it
//! wrong is not a small error: a camera the clock does not wait for does not
//! quietly degrade, it drops in and out of ticks, and a joint reconstructed
//! from a different set of cameras every few ticks moves by the disagreement
//! between them each time the set changes. That disagreement is the calibration
//! error, it is centimetres, and it is a square wave between two different
//! right answers rather than noise any filter can take out.

use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::calib::RoomCalibration;
use crate::config::FusionConfig;
use crate::geometry::camera::Camera;
use crate::pipeline::PoseChannel;
use crate::worker::timing::{Rate, Ticker, ema};
use crate::worker::{Shutdown, Supervisor};

use super::align::align;
use super::bones::{BoneMeter, MeasureOptions, Skeleton};
use super::filter::{Filtered, PoseFilter};
use super::fit::{Fitted, Fitter};
use super::floor::FloorMeter;
use super::fuse::{Pose3d, fuse};
use super::shake::{Shake, ShakeMeter};

/// How often the bone measurement is recomputed, in ticks.
///
/// It is a median over thousands of samples per bone, which is not free, and it
/// moves by fractions of a millimetre between ticks. Once a second is more
/// often than the body changes shape.
const REMEASURE_EVERY: u32 = 60;

/// One tick's output, at every stage it passed through.
///
/// The raw reconstruction is kept alongside the finished one because it is
/// what the diagnostics are about: the residuals, the cameras that were
/// dropped, and how much the fit had to change. Showing only the filtered
/// result would leave a user watching something that looks fine while a camera
/// contributes nothing.
#[derive(Debug, Clone)]
pub struct FusionFrame {
    pub raw: Pose3d,
    pub fitted: Fitted,
    pub filtered: Filtered,
}

/// How much of the answer one camera is carrying.
#[derive(Debug, Clone, Default)]
pub struct CameraContribution {
    pub id: String,
    /// Fraction of recent ticks this camera had frames to interpolate between.
    /// Well under one means it is too slow, stalled, or seeing nobody.
    pub aligned: f32,
    /// Mean share of the joints it carried, over the joints it voted on.
    pub weight: f32,
    /// Fraction of its rays dropped for disagreeing with the others. Steadily
    /// high means this camera is mis-calibrated or badly placed.
    pub rejected: f32,
    /// The delay the calibration measured for it, in milliseconds.
    pub latency_ms: f32,
    /// Why this camera is not being used, when it is not.
    pub problem: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct FusionStats {
    pub running: bool,
    /// Measured rate of the fusion clock.
    pub rate: f32,
    /// How far behind real time the clock is running, in milliseconds.
    pub lag_ms: f32,
    /// Joints reconstructed on the most recent tick.
    pub joints: usize,
    /// Of those, how many the fit placed rather than the cameras seeing.
    pub inferred: usize,
    /// Lower-body joints present, which is what the trackers need.
    pub lower_body: usize,
    /// Largest distance the fit had to move a joint, in metres.
    pub worst_correction: f64,
    /// How much the body is wobbling at each stage of the chain, in metres.
    pub shake: Shake,
    pub cameras: Vec<CameraContribution>,
    /// The body measurement as it stands.
    pub body: Skeleton,
    /// Where the feet say the floor is, relative to where SteamVR says it is,
    /// in metres. None until enough of the user has been seen to judge.
    pub floor: Option<f64>,
    pub measuring: bool,
    pub warning: Option<String>,
}

/// The shared face of the fusion thread.
#[derive(Default)]
pub struct FusionChannel {
    stop: Shutdown,
    latest: Mutex<Option<Arc<FusionFrame>>>,
    stats: Mutex<FusionStats>,
}

impl FusionChannel {
    pub fn latest(&self) -> Option<Arc<FusionFrame>> {
        self.latest.lock().clone()
    }

    pub fn stats(&self) -> FusionStats {
        self.stats.lock().clone()
    }
}

/// Owns the fusion thread, if one is running.
#[derive(Default)]
pub struct Fusion {
    channel: Option<Arc<FusionChannel>>,
}

impl Fusion {
    pub fn channel(&self) -> Option<&Arc<FusionChannel>> {
        self.channel.as_ref()
    }

    pub fn is_running(&self) -> bool {
        self.channel.is_some()
    }

    /// The current measurement, for the UI to save.
    pub fn body(&self) -> Option<Skeleton> {
        Some(self.channel.as_ref()?.stats().body)
    }

    /// A stage with no thread behind it, holding one fixed frame.
    ///
    /// The tracking panel draws everything from a channel, so laying it out
    /// with a body in it — the state that carries all the interesting
    /// formatting — would otherwise need a calibrated room and three running
    /// cameras. See §12.2 of the design document for why that layout is worth
    /// testing at all.
    pub fn detached(stats: FusionStats, frame: Option<FusionFrame>) -> Self {
        Self {
            channel: Some(Arc::new(FusionChannel {
                stop: Shutdown::default(),
                latest: Mutex::new(frame.map(Arc::new)),
                stats: Mutex::new(stats),
            })),
        }
    }

    /// Starts fusing, if the room and the cameras allow it.
    ///
    /// Returns why it could not start rather than failing silently: "tracking
    /// is not running" with no reason is the least useful thing this could do.
    pub fn start(
        &mut self,
        config: &FusionConfig,
        cameras: Vec<(String, Arc<PoseChannel>)>,
        room: &RoomCalibration,
        body: Skeleton,
        supervisor: &mut Supervisor,
    ) -> Result<(), String> {
        self.stop();

        let mut tracked = Vec::new();
        for (id, poses) in cameras {
            let Some(calibrated) = room.camera(&id) else {
                continue;
            };
            tracked.push(Tracked {
                id,
                poses,
                camera: calibrated.camera.clone(),
                // A latency the walk was too slow to pin down was reported and
                // not applied, and it must not be applied here either: a delay
                // guessed off a flat curve is worse than no delay at all.
                latency: calibrated
                    .latency
                    .filter(|estimate| estimate.is_confident())
                    .map(|estimate| estimate.latency)
                    .unwrap_or_default(),
                behind: 0.0,
                admitted: true,
                flipping: None,
                lateness: None,
                aligned: 0.0,
                weight: 0.0,
                rejected: 0.0,
                problem: None,
            });
        }

        if tracked.len() < 2 {
            return Err(format!(
                "{} of the running cameras are in this room profile; fusion needs at least two",
                tracked.len()
            ));
        }

        let channel = Arc::new(FusionChannel {
            stop: Shutdown::default(),
            latest: Mutex::new(None),
            stats: Mutex::new(FusionStats {
                running: true,
                ..FusionStats::default()
            }),
        });
        self.channel = Some(channel.clone());

        let config = config.clone();
        supervisor.spawn("fusion", move |global| {
            run(channel, config, tracked, body, global)
        });

        Ok(())
    }

    pub fn stop(&mut self) {
        if let Some(channel) = self.channel.take() {
            channel.stop.cancel();
            channel.stats.lock().running = false;
        }
    }
}

/// One camera as the fusion loop sees it.
struct Tracked {
    id: String,
    poses: Arc<PoseChannel>,
    camera: Camera,
    latency: Duration,
    /// Worst recent staleness of this camera.s newest frame, in seconds. Rises
    /// at once and decays slowly, so the clock is set by what the camera does
    /// on a bad tick rather than on a lucky one.
    behind: f64,
    /// Whether the fusion clock waits for this camera at all.
    admitted: bool,
    /// When the admission decision last started wanting to change.
    flipping: Option<Instant>,
    /// Set when this camera is too far behind to be waited for.
    lateness: Option<String>,
    aligned: f32,
    weight: f32,
    rejected: f32,
    problem: Option<String>,
}

impl Tracked {
    /// How far back the clock has to sit for this camera to have a frame after
    /// the tick: what its timestamps are early by, plus how stale its newest
    /// frame gets, plus the margin.
    fn wait(&self, slack: Duration) -> Duration {
        self.latency + Duration::from_secs_f64(self.behind.max(0.0)) + slack
    }

    /// Adjusts the calibrated optics to the resolution the camera is actually
    /// running at.
    ///
    /// Changing a camera's resolution does not change its lens, and forcing a
    /// re-calibration for it would be absurd — but the keypoints arrive in the
    /// new resolution's pixels, and using them against the old intrinsics would
    /// silently reconstruct a body in the wrong place.
    fn match_resolution(&mut self, width: u32, height: u32) -> bool {
        if self.camera.intrinsics.width == width && self.camera.intrinsics.height == height {
            self.problem = None;
            return true;
        }

        match self.camera.intrinsics.scaled_to(width, height) {
            Some(scaled) => {
                tracing::info!(
                    camera = %self.id,
                    "rescaled the calibrated optics from {}x{} to {width}x{height}",
                    self.camera.intrinsics.width,
                    self.camera.intrinsics.height,
                );
                self.camera.intrinsics = scaled;
                self.problem = None;
                true
            }
            // A different aspect ratio is a different field of view, not a
            // different readout, and no scaling recovers it.
            None => {
                self.problem = Some(format!(
                    "running at {width}x{height}, calibrated at {}x{}",
                    self.camera.intrinsics.width, self.camera.intrinsics.height
                ));
                false
            }
        }
    }
}

/// How long the worst recent delivery gap takes to halve, in seconds.
///
/// The clock follows the latest camera, and following it back down quickly
/// would mean re-tightening on one lucky moment and dropping every camera out
/// again on the next ordinary one. Rising is instant; falling is deliberate.
const LAG_HALF_LIFE: f64 = 4.0;

/// How long an admission decision has to be wanted before it is taken.
///
/// Long, because the decision is which cameras are reconstructing the body,
/// and changing it moves the body. A hiccup must not cost a camera.
const ADMISSION_DWELL: Duration = Duration::from_secs(2);

/// How far back the fusion clock has to sit for the cameras to answer it.
///
/// Measured every tick rather than assumed once at startup. What matters is
/// not the delay the calibration attributed to a camera's timestamps but how
/// stale that camera's newest frame actually is, which depends on the model it
/// is running, the resolution it is running at and what else is on the GPU —
/// none of which is known when the thread starts.
fn follow_cameras(
    tracked: &mut [Tracked],
    now: Instant,
    dt: f64,
    config: &FusionConfig,
) -> Duration {
    let slack = Duration::from_millis(config.align_slack_ms as u64);
    let ceiling = Duration::from_millis(config.max_lag_ms.max(config.align_slack_ms) as u64);
    let decay = 0.5f64.powf(dt / LAG_HALF_LIFE);

    for camera in tracked.iter_mut() {
        // How stale this camera's newest frame is. The clock has to sit at
        // least this far back, or there is nothing after the tick to
        // interpolate towards. A camera that has delivered nothing at all is
        // treated as being past the ceiling rather than as being on time.
        let behind = camera
            .poses
            .span()
            .map(|(_, newest)| now.saturating_duration_since(newest).as_secs_f64())
            .unwrap_or_else(|| ceiling.as_secs_f64());
        camera.behind = behind.max(camera.behind * decay);
    }

    // Decided before the lag is: a camera nobody is waiting for must not be
    // what everybody waits for.
    admit(tracked, now, ceiling, slack);

    tracked
        .iter()
        .filter(|camera| camera.admitted)
        .map(|camera| camera.wait(slack))
        .max()
        .unwrap_or(slack)
        .clamp(slack, ceiling)
}

/// Decides which cameras the clock is prepared to wait for.
///
/// A camera either is waited for or is not. What it must never do is be waited
/// for on some ticks and not others, which is what the old fixed clock left it
/// doing: every joint it votes on moves by the disagreement between it and the
/// rest each time it joins or leaves, that disagreement is the calibration
/// error and is centimetres, and it happens at whatever rate the camera is
/// missing ticks. That is not noise a filter can take out — it is a square
/// wave between two different right answers, and it is what shaking looks like
/// from the inside.
fn admit(tracked: &mut [Tracked], now: Instant, ceiling: Duration, slack: Duration) {
    // Triangulation needs two cameras, so the two least late are waited for
    // whatever they cost. A late body beats no body, and the alternative to
    // waiting for them is not a faster body but an empty room.
    let mut order: Vec<usize> = (0..tracked.len()).collect();
    order.sort_by_key(|index| tracked[*index].wait(slack));
    let essential = &order[..order.len().min(2)];

    let wanted: Vec<bool> = tracked
        .iter()
        .enumerate()
        .map(|(index, camera)| {
            // Hysteresis: a camera has to come back comfortably inside the
            // ceiling to be waited for again, not merely back to the edge of it.
            let limit = if camera.admitted {
                ceiling
            } else {
                ceiling.mul_f64(0.85)
            };
            essential.contains(&index) || camera.wait(slack) <= limit
        })
        .collect();

    for (camera, wanted) in tracked.iter_mut().zip(wanted) {
        if wanted == camera.admitted {
            camera.flipping = None;
            continue;
        }

        let since = *camera.flipping.get_or_insert(now);
        if now.duration_since(since) < ADMISSION_DWELL {
            continue;
        }

        let need = camera.wait(slack);
        camera.admitted = wanted;
        camera.flipping = None;
        camera.lateness = (!wanted).then(|| {
            format!(
                "delivering {:.0} ms behind real time, and the fusion clock waits at most {:.0} ms",
                need.as_secs_f64() * 1000.0,
                ceiling.as_secs_f64() * 1000.0,
            )
        });
        tracing::info!(
            camera = %camera.id,
            behind_ms = need.as_secs_f64() * 1000.0,
            "camera {} the fusion clock",
            if wanted { "rejoined" } else { "left" },
        );
    }
}

fn run(
    channel: Arc<FusionChannel>,
    config: FusionConfig,
    mut tracked: Vec<Tracked>,
    body: Skeleton,
    global: Shutdown,
) {
    tracing::info!(cameras = tracked.len(), "fusion started");

    // How far behind real time the clock runs, followed rather than fixed. It
    // starts at the margin and grows to whatever the cameras turn out to need.
    let mut lag = Duration::from_millis(config.align_slack_ms as u64);
    let period = 1.0 / f64::from(config.rate_hz.max(1));

    let fuse_options = config.fuse_options();
    let measure_options = MeasureOptions::default();

    let mut meter = BoneMeter::new(measure_options.clone());
    let mut floor = FloorMeter::default();
    let mut skeleton = body;
    let mut fitter = Fitter::new(config.fit_options(measure_options.clone()));
    let mut filter = PoseFilter::new(config.filter_options());

    // One meter per stage, so that "it shakes" can be answered with which
    // stage it started shaking at.
    let mut shaking = ShakeMeters::default();

    let mut ticker = Ticker::at_hz(config.rate_hz as f32);
    let mut rate = Rate::default();
    let mut since_measure = 0u32;

    while !channel.stop.is_cancelled() && !global.is_cancelled() {
        let now = Instant::now();
        // The clock may slow down but it must never run backwards. A
        // reconstruction stamped before the last one gives every stage after it
        // a negative time step, and the honest response to a camera that has
        // just got slower is to let real time catch up rather than to rewind.
        // Capped at half a tick per tick, the clock falls behind at half speed
        // until it has covered the new delay, which takes a fraction of a
        // second and looks like nothing.
        let want = follow_cameras(&mut tracked, now, period, &config);
        let step = Duration::from_secs_f64(period * 0.5);
        lag = if want > lag {
            lag + step.min(want - lag)
        } else {
            want
        };
        let at = now - lag;

        let mut cameras = Vec::with_capacity(tracked.len());
        let mut views = Vec::with_capacity(tracked.len());

        for camera in tracked.iter_mut() {
            let index = cameras.len();

            // A camera the clock is not waiting for takes no part at all. It is
            // the flickering that hurts, not the absence.
            if !camera.admitted {
                camera.aligned = ema(camera.aligned, 0.0);
                cameras.push(camera.camera.clone());
                continue;
            }

            // The frame that shows the world at `at` is stamped a latency
            // later, because that is when it arrived rather than when the light
            // landed on the sensor.
            let sampled_at = at + camera.latency;

            let bracket = camera.poses.bracket(sampled_at);
            let usable = match &bracket {
                Some((before, _)) => camera.match_resolution(before.width, before.height),
                None => false,
            };

            camera.aligned = ema(camera.aligned, if usable { 1.0 } else { 0.0 });
            // After the resolution check, which may have rescaled the optics.
            cameras.push(camera.camera.clone());

            if let (Some((before, after)), true) = (bracket, usable) {
                let aligned = align(&before, &after, sampled_at);
                if !aligned.is_empty() {
                    views.push((index, aligned));
                }
            }
        }

        let raw = fuse(&cameras, &views, at, &fuse_options);

        // Measured from the reconstruction, never from the fitted result: the
        // fit already holds the body to this measurement, so measuring its
        // output would be a loop that confirms whatever it started with.
        // Measured whatever the body settings say, because this is not about
        // the user's anatomy: it is about whether the frame everything else is
        // expressed in has its floor in the right place.
        floor.observe(&raw);

        if config.measure_body && !raw.is_empty() {
            meter.observe(&raw);
            since_measure += 1;
            if since_measure >= REMEASURE_EVERY {
                since_measure = 0;
                skeleton = meter.finish();
            }
        }

        let fitted = fitter.fit(&raw, &skeleton);
        let filtered = filter.push(&fitted);
        shaking.observe(&raw, &fitted, &filtered);

        publish(
            &channel,
            &config,
            &mut tracked,
            &raw,
            &fitted,
            &filtered,
            &skeleton,
            &measure_options,
            floor.estimate(),
            shaking.shake(),
            rate.tick(now),
            lag,
        );

        *channel.latest.lock() = Some(Arc::new(FusionFrame {
            raw,
            fitted,
            filtered,
        }));

        if !ticker.wait(&channel.stop) || global.is_cancelled() {
            break;
        }
    }

    channel.stats.lock().running = false;
    tracing::info!("fusion stopped");
}

#[allow(clippy::too_many_arguments)]
fn publish(
    channel: &FusionChannel,
    config: &FusionConfig,
    tracked: &mut [Tracked],
    raw: &Pose3d,
    fitted: &Fitted,
    filtered: &Filtered,
    skeleton: &Skeleton,
    measure: &MeasureOptions,
    floor: Option<f64>,
    shake: Shake,
    rate: f32,
    lag: Duration,
) {
    // Per camera: the share of the joints it carried, and how often it was
    // outvoted. Both are averaged over the joints it had a say in, so a camera
    // that can only see the legs is not marked down for the arms.
    for (index, camera) in tracked.iter_mut().enumerate() {
        let mut weight = 0.0;
        let mut voted = 0usize;
        let mut rejected = 0usize;

        for (_, joint) in raw.iter() {
            if let Some((_, share)) = joint.weights.iter().find(|(id, _)| *id == index) {
                weight += share;
                voted += 1;
            } else if joint.rejected.contains(&index) {
                rejected += 1;
                voted += 1;
            }
        }

        if voted > 0 {
            camera.weight = ema(camera.weight, (weight / voted as f64) as f32);
            camera.rejected = ema(camera.rejected, rejected as f32 / voted as f32);
        }
    }

    let mut stats = channel.stats.lock();
    stats.running = true;
    stats.rate = rate;
    stats.lag_ms = lag.as_secs_f32() * 1000.0;
    stats.joints = filtered.count();
    stats.inferred = fitted.inferred();
    stats.lower_body = raw.lower_body();
    stats.worst_correction = fitted.worst_correction();
    stats.measuring = config.measure_body;
    stats.body = skeleton.clone();
    stats.floor = floor;
    stats.shake = shake;
    stats.cameras = tracked
        .iter()
        .map(|camera| CameraContribution {
            id: camera.id.clone(),
            aligned: camera.aligned,
            weight: camera.weight,
            rejected: camera.rejected,
            latency_ms: camera.latency.as_secs_f32() * 1000.0,
            problem: camera.problem.clone().or_else(|| camera.lateness.clone()),
        })
        .collect();
    stats.warning = warning(&stats, skeleton, measure);
}

/// The single thing most worth telling the user, if anything is wrong.
fn warning(stats: &FusionStats, skeleton: &Skeleton, measure: &MeasureOptions) -> Option<String> {
    if let Some(camera) = stats.cameras.iter().find(|camera| camera.problem.is_some()) {
        return Some(format!(
            "{} is not being used: {}",
            camera.id,
            camera.problem.as_deref().unwrap_or_default()
        ));
    }

    // Ahead of everything else, because it is the one problem that makes the
    // whole output wrong while every other number looks healthy.
    if let Some(floor) = stats.floor.filter(|floor| floor.abs() > 0.06) {
        return Some(format!(
            "your feet are reconstructing {:.0} cm {} the floor SteamVR reports, so its \
             room setup is off by that much",
            floor.abs() * 100.0,
            if floor < 0.0 { "below" } else { "above" }
        ));
    }

    let contributing = stats
        .cameras
        .iter()
        .filter(|camera| camera.aligned > 0.5)
        .count();
    if contributing < 2 {
        return Some(format!(
            "only {contributing} camera(s) are keeping up with the fusion clock"
        ));
    }

    if let Some(worst) = stats
        .cameras
        .iter()
        .filter(|camera| camera.aligned > 0.5)
        .max_by(|a, b| a.rejected.total_cmp(&b.rejected))
        .filter(|camera| camera.rejected > 0.4)
    {
        return Some(format!(
            "{} disagrees with the others on {:.0}% of the joints it sees",
            worst.id,
            worst.rejected * 100.0
        ));
    }

    if stats.lower_body == 0 {
        return Some("no lower-body joints are being reconstructed".to_owned());
    }

    if skeleton.coverage(measure) < 0.5 {
        return Some("still measuring the body; move around for a moment".to_owned());
    }

    if stats.worst_correction > 0.15 {
        return Some(format!(
            "the measured body and the cameras disagree by {:.0} cm",
            stats.worst_correction * 100.0
        ));
    }

    None
}

/// The four stage meters, kept together because they are always read together.
///
/// The prediction is measured where the joint is *told* to be rather than where
/// it was smoothed to, which is the whole reason for having a fourth meter: a
/// prediction is a position plus a velocity times a lead, and the velocity is
/// the one part of this chain that can add movement to a signal instead of
/// removing it.
#[derive(Debug, Default)]
struct ShakeMeters {
    raw: ShakeMeter,
    fitted: ShakeMeter,
    filtered: ShakeMeter,
    predicted: ShakeMeter,
}

impl ShakeMeters {
    fn observe(&mut self, raw: &Pose3d, fitted: &Fitted, filtered: &Filtered) {
        self.raw
            .observe(raw.iter().map(|(joint, fused)| (joint, fused.point)));
        self.fitted.observe(
            fitted
                .iter()
                .map(|(joint, joint_fit)| (joint, joint_fit.point)),
        );
        self.filtered
            .observe(filtered.iter().map(|(joint, one)| (joint, one.point)));
        self.predicted
            .observe(filtered.iter().map(|(joint, one)| (joint, one.predicted)));
    }

    fn shake(&self) -> Shake {
        Shake {
            raw: self.raw.metres(),
            fitted: self.fitted.metres(),
            filtered: self.filtered.metres(),
            predicted: self.predicted.metres(),
        }
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{Point3, Vector3};

    use super::*;
    use crate::geometry::camera::Intrinsics;
    use crate::geometry::lens::Lens;

    fn camera() -> Camera {
        Camera::look_at(
            Intrinsics::from_fov(1280, 720, 70f64.to_radians()),
            Lens::default(),
            Point3::new(-1.8, 2.4, -1.8),
            Point3::new(0.0, 1.0, 0.0),
            Vector3::y(),
        )
    }

    fn tracked(id: &str) -> Tracked {
        Tracked {
            id: id.to_owned(),
            poses: Arc::new(PoseChannel::default()),
            camera: camera(),
            latency: Duration::ZERO,
            behind: 0.0,
            admitted: true,
            flipping: None,
            lateness: None,
            aligned: 0.0,
            weight: 0.0,
            rejected: 0.0,
            problem: None,
        }
    }

    /// Changing a camera's resolution does not change its lens, and should not
    /// invalidate a calibration.
    #[test]
    fn the_same_lens_at_another_resolution_is_rescaled() {
        let mut camera = tracked("cam0");
        assert!(camera.match_resolution(640, 360));

        assert_eq!(camera.camera.intrinsics.width, 640);
        assert!(
            (camera.camera.intrinsics.horizontal_fov() - 70f64.to_radians()).abs() < 1e-9,
            "the field of view should be unchanged"
        );
        assert!(camera.problem.is_none());
    }

    /// A different aspect ratio is a different field of view, and quietly
    /// scaling it would put the body somewhere else.
    #[test]
    fn a_different_shape_of_frame_takes_the_camera_out_of_service() {
        let mut camera = tracked("cam0");
        assert!(!camera.match_resolution(640, 480));
        assert!(camera.problem.is_some());
    }

    /// Hands a camera one frame stamped at `captured_at`, which is all
    /// [`follow_cameras`] looks at.
    fn feed(camera: &Tracked, captured_at: Instant) {
        camera.poses.publish(crate::pipeline::PoseFrame {
            seq: 0,
            captured_at,
            width: 1280,
            height: 720,
            detection: None,
            keypoints: crate::infer::traits::Keypoints2d::default(),
        });
    }

    fn slack(config: &FusionConfig) -> Duration {
        Duration::from_millis(config.align_slack_ms as u64)
    }

    fn ceiling(config: &FusionConfig) -> Duration {
        Duration::from_millis(config.max_lag_ms as u64)
    }

    /// The clock has to sit behind the camera that hands its frames over last,
    /// because interpolating onto a tick needs a frame after it.
    #[test]
    fn the_clock_waits_for_the_camera_that_delivers_latest() {
        let config = FusionConfig::default();
        let base = Instant::now();
        let mut cameras = vec![tracked("cam0"), tracked("cam1"), tracked("cam2")];

        feed(&cameras[0], base);
        feed(&cameras[1], base);
        feed(&cameras[2], base - Duration::from_millis(100));

        let lag = follow_cameras(&mut cameras, base, 1.0 / 60.0, &config);
        assert_eq!(lag, Duration::from_millis(100) + slack(&config));
    }

    /// A camera nobody could wait for is left out, not waited for half the
    /// time. Being in the reconstruction on some ticks and not others moves
    /// every joint it votes on by the calibration disagreement each time the
    /// set changes, which is the shaking this is all about.
    #[test]
    fn a_camera_too_late_to_wait_for_is_left_out_rather_than_flickering() {
        let config = FusionConfig::default();
        let base = Instant::now();
        let mut cameras = vec![tracked("cam0"), tracked("cam1"), tracked("cam2")];

        feed(&cameras[0], base);
        feed(&cameras[1], base);
        feed(&cameras[2], base - Duration::from_millis(400));

        // One slow tick is not a verdict, so it is still being waited for and
        // the clock is pinned at its ceiling.
        let lag = follow_cameras(&mut cameras, base, 1.0 / 60.0, &config);
        assert!(cameras[2].admitted);
        assert_eq!(lag, ceiling(&config));

        // Once it has been slow for long enough, it goes.
        let later = base + ADMISSION_DWELL + Duration::from_millis(1);
        feed(&cameras[0], later);
        feed(&cameras[1], later);
        let lag = follow_cameras(&mut cameras, later, 1.0 / 60.0, &config);

        assert!(!cameras[2].admitted);
        assert!(
            cameras[2].lateness.is_some(),
            "and the panel should be able to say why"
        );
        assert_eq!(
            lag,
            slack(&config),
            "with the straggler out, the clock tightens back up to the two that are keeping up"
        );
    }

    /// Triangulation needs two cameras. When only two are left, waiting for
    /// them is not a choice: the alternative is not a faster body but no body.
    #[test]
    fn the_last_two_cameras_are_waited_for_however_late_they_are() {
        let config = FusionConfig::default();
        let base = Instant::now();
        let mut cameras = vec![tracked("cam0"), tracked("cam1")];

        feed(&cameras[0], base - Duration::from_millis(500));
        feed(&cameras[1], base - Duration::from_millis(500));

        let mut lag = Duration::ZERO;
        for tick in 0..300u64 {
            lag = follow_cameras(
                &mut cameras,
                base + Duration::from_millis(tick * 20),
                1.0 / 60.0,
                &config,
            );
        }

        assert!(cameras.iter().all(|camera| camera.admitted));
        assert_eq!(lag, ceiling(&config), "clamped, but still every camera");
    }

    /// The clock is set by what a camera manages on a bad tick, not on a lucky
    /// one, or it tightens onto one good moment and drops everybody out on the
    /// next ordinary one.
    #[test]
    fn one_early_frame_does_not_tighten_the_clock() {
        let config = FusionConfig::default();
        let base = Instant::now();
        let mut cameras = vec![tracked("cam0"), tracked("cam1")];

        feed(&cameras[0], base - Duration::from_millis(120));
        feed(&cameras[1], base - Duration::from_millis(120));
        let slow = follow_cameras(&mut cameras, base, 1.0 / 60.0, &config);

        // Both cameras suddenly deliver on time.
        let next = base + Duration::from_millis(16);
        feed(&cameras[0], next);
        feed(&cameras[1], next);
        let after = follow_cameras(&mut cameras, next, 1.0 / 60.0, &config);

        assert!(
            after > slow - Duration::from_millis(5),
            "one good tick moved the clock from {slow:?} to {after:?}"
        );
    }

    #[test]
    fn a_camera_nothing_can_reach_is_named_in_the_warning() {
        let mut stats = FusionStats {
            cameras: vec![
                CameraContribution {
                    id: "cam0".to_owned(),
                    aligned: 1.0,
                    ..CameraContribution::default()
                },
                CameraContribution {
                    id: "cam1".to_owned(),
                    aligned: 0.0,
                    ..CameraContribution::default()
                },
            ],
            lower_body: 6,
            ..FusionStats::default()
        };

        let message = warning(&stats, &Skeleton::default(), &MeasureOptions::default())
            .expect("one camera is not keeping up");
        assert!(message.contains("1 camera"), "{message}");

        stats.cameras[1].aligned = 1.0;
        stats.cameras[1].rejected = 0.8;
        let message = warning(&stats, &Skeleton::default(), &MeasureOptions::default()).unwrap();
        assert!(message.contains("cam1"), "{message}");
    }

    #[test]
    fn a_healthy_stage_says_nothing_once_the_body_is_measured() {
        let stats = FusionStats {
            cameras: (0..3)
                .map(|index| CameraContribution {
                    id: format!("cam{index}"),
                    aligned: 1.0,
                    weight: 0.33,
                    rejected: 0.02,
                    ..CameraContribution::default()
                })
                .collect(),
            lower_body: 8,
            worst_correction: 0.01,
            ..FusionStats::default()
        };

        let mut skeleton = Skeleton::default();
        for bone in super::super::bones::BONES {
            skeleton.bones.push(super::super::bones::BoneLength {
                bone: *bone,
                length: 0.4,
                samples: 1000,
                scatter: 0.002,
            });
        }

        assert_eq!(warning(&stats, &skeleton, &MeasureOptions::default()), None);
    }
}
