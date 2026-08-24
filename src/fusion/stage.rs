//! The fusion thread.
//!
//! One loop on a fixed clock, running every stage of this module in order:
//! resample each camera onto the tick, triangulate, measure the body,
//! hold the body to the measurement, filter, publish.
//!
//! The clock deliberately runs behind real time. Interpolating a camera onto an
//! instant needs a frame *after* that instant, and cameras hand their frames
//! over late by an amount the calibration measured — so the tick sits far
//! enough back that even the slowest camera has already delivered the frame
//! that follows it. The lag is not a loss: the prediction stage has to
//! compensate for a delay several times larger anyway, and predicting forward
//! from a properly aligned reconstruction beats fusing rays taken at three
//! different instants.

use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::calib::RoomCalibration;
use crate::config::FusionConfig;
use crate::geometry::camera::Camera;
use crate::pipeline::PoseChannel;
use crate::worker::timing::Ticker;
use crate::worker::{Shutdown, Supervisor};

use super::align::align;
use super::bones::{BoneMeter, MeasureOptions, Skeleton};
use super::filter::{Filtered, PoseFilter};
use super::fit::{Fitted, Fitter};
use super::floor::FloorMeter;
use super::fuse::{Pose3d, fuse};

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
    aligned: f32,
    weight: f32,
    rejected: f32,
    problem: Option<String>,
}

impl Tracked {
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

fn run(
    channel: Arc<FusionChannel>,
    config: FusionConfig,
    mut tracked: Vec<Tracked>,
    body: Skeleton,
    global: Shutdown,
) {
    // How far behind real time the clock runs: enough that the camera which
    // reports latest has still delivered the frame after the tick.
    let slowest = tracked
        .iter()
        .map(|camera| camera.latency)
        .max()
        .unwrap_or_default();
    let lag = slowest + Duration::from_millis(config.align_slack_ms as u64);

    tracing::info!(
        cameras = tracked.len(),
        lag_ms = lag.as_secs_f32() * 1000.0,
        "fusion started"
    );

    let fuse_options = config.fuse_options();
    let measure_options = MeasureOptions::default();

    let mut meter = BoneMeter::new(measure_options.clone());
    let mut floor = FloorMeter::default();
    let mut skeleton = body;
    let mut fitter = Fitter::new(config.fit_options(measure_options.clone()));
    let mut filter = PoseFilter::new(config.filter_options());

    let mut ticker = Ticker::at_hz(config.rate_hz as f32);
    let mut rate = Rate::default();
    let mut since_measure = 0u32;

    while !channel.stop.is_cancelled() && !global.is_cancelled() {
        let now = Instant::now();
        let at = now - lag;

        let mut cameras = Vec::with_capacity(tracked.len());
        let mut views = Vec::with_capacity(tracked.len());

        for camera in tracked.iter_mut() {
            let index = cameras.len();
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
    stats.cameras = tracked
        .iter()
        .map(|camera| CameraContribution {
            id: camera.id.clone(),
            aligned: camera.aligned,
            weight: camera.weight,
            rejected: camera.rejected,
            latency_ms: camera.latency.as_secs_f32() * 1000.0,
            problem: camera.problem.clone(),
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

/// A smoothed count of ticks per second.
#[derive(Default)]
struct Rate {
    last: Option<Instant>,
    rate: f32,
}

impl Rate {
    fn tick(&mut self, now: Instant) -> f32 {
        if let Some(previous) = self.last.replace(now) {
            let dt = now.duration_since(previous).as_secs_f32();
            if dt > 0.0 {
                self.rate = ema(self.rate, 1.0 / dt);
            }
        }
        self.rate
    }
}

fn ema(current: f32, sample: f32) -> f32 {
    const ALPHA: f32 = 0.05;
    if current == 0.0 {
        sample
    } else {
        current * (1.0 - ALPHA) + sample * ALPHA
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
