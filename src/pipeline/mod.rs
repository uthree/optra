//! The inference stage.
//!
//! One thread serves every camera: it takes the newest frame from each, runs
//! detection at a reduced rate, batches the pose crops of the cameras that
//! share a model, and publishes canonical keypoints per camera.
//!
//! Models load in the background. A camera keeps running the model it already
//! has until its replacement is ready, so changing a model never stalls
//! tracking, which matters because building a DirectML session takes about a
//! second.

mod models;

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender, unbounded};
use parking_lot::Mutex;

use crate::capture::{CameraChannel, Frame};
use crate::config::{CameraConfig, InferenceConfig};
use crate::infer::Backend;
use crate::infer::traits::{Detection, ImageView, Keypoints2d};
use crate::worker::{Shutdown, Supervisor};
use models::{ModelSet, Slot};

/// What the pipeline produced for one camera frame.
#[derive(Debug, Clone)]
pub struct PoseFrame {
    /// Sequence number of the camera frame this came from.
    pub seq: u64,
    /// When the frame was captured, not when it was processed.
    pub captured_at: Instant,
    /// Size of the frame the coordinates refer to.
    pub width: u32,
    pub height: u32,
    /// The box the keypoints were estimated from.
    pub detection: Option<Detection>,
    pub keypoints: Keypoints2d,
}

#[derive(Debug, Clone, Default)]
pub struct PoseStats {
    /// Frames that produced keypoints.
    pub processed: u64,
    /// Frames where no person was found.
    pub empty: u64,
    /// Smoothed rate at which this camera is being processed.
    pub fps: f32,
    /// Smoothed time from capture to published keypoints, in milliseconds.
    pub latency_ms: f32,
    pub detector_model: Option<String>,
    pub pose_model: Option<String>,
    pub backend: Option<Backend>,
    pub last_error: Option<String>,
}

/// How much of each camera's output is kept for the fusion stage to look back
/// through.
///
/// Long enough to cover the alignment lag plus the largest per-camera latency
/// correction, with room for a camera that stalls briefly. Nothing ever asks
/// for more.
const HISTORY: Duration = Duration::from_secs(2);

/// Per-camera output of the inference stage.
#[derive(Default)]
pub struct PoseChannel {
    slot: Mutex<Option<Arc<PoseFrame>>>,
    /// The last [`HISTORY`] of frames, oldest first.
    ///
    /// The newest frame is enough to draw an overlay but not to fuse. Fusion
    /// asks what every camera saw at one shared instant, and that instant falls
    /// between two frames on each of them; keeping the frames is what lets the
    /// answer be interpolated rather than guessed.
    history: Mutex<VecDeque<Arc<PoseFrame>>>,
    stats: Mutex<PoseStats>,
    last_at: Mutex<Option<Instant>>,
}

impl PoseChannel {
    pub fn peek(&self) -> Option<Arc<PoseFrame>> {
        self.slot.lock().clone()
    }

    /// The two frames either side of `at`, for the fusion stage to interpolate
    /// between.
    ///
    /// `None` when `at` falls outside the kept history, which is the honest
    /// answer for a camera that has not delivered a frame since. Extrapolating
    /// a keypoint forward invents a limb position, and one invented ray is
    /// enough to drag a triangulated joint across the room.
    pub fn bracket(&self, at: Instant) -> Option<(Arc<PoseFrame>, Arc<PoseFrame>)> {
        let history = self.history.lock();

        let after = history.iter().position(|frame| frame.captured_at >= at)?;
        // A tick landing exactly on the first frame brackets it with itself,
        // which interpolates to that frame. A tick before it does not.
        let before = match after.checked_sub(1) {
            Some(before) => before,
            None if history[after].captured_at == at => after,
            None => return None,
        };

        Some((history[before].clone(), history[after].clone()))
    }

    /// The span the kept history covers, which is what says whether a camera is
    /// keeping up with the fusion clock.
    pub fn span(&self) -> Option<(Instant, Instant)> {
        let history = self.history.lock();
        Some((history.front()?.captured_at, history.back()?.captured_at))
    }

    pub fn stats(&self) -> PoseStats {
        self.stats.lock().clone()
    }

    fn publish(&self, frame: PoseFrame) {
        let now = Instant::now();
        let latency = now.duration_since(frame.captured_at).as_secs_f32() * 1000.0;
        let empty = frame.keypoints.is_empty();

        let frame = Arc::new(frame);
        *self.slot.lock() = Some(frame.clone());
        self.remember(frame);

        let mut stats = self.stats.lock();
        if empty {
            stats.empty += 1;
        } else {
            stats.processed += 1;
        }
        stats.latency_ms = ema(stats.latency_ms, latency, 0.1);

        let mut last = self.last_at.lock();
        if let Some(previous) = last.replace(now) {
            let dt = now.duration_since(previous).as_secs_f32();
            if dt > 0.0 {
                stats.fps = ema(stats.fps, 1.0 / dt, 0.1);
            }
        }
    }

    /// Appends a frame to the history and drops what has aged out.
    fn remember(&self, frame: Arc<PoseFrame>) {
        let mut history = self.history.lock();

        // Every lookup relies on the history being sorted by capture time. A
        // camera that restarts can hand over a frame stamped before the last
        // one, and starting over costs that camera a single alignment window.
        if history
            .back()
            .is_some_and(|last| last.captured_at > frame.captured_at)
        {
            history.clear();
        }

        let cutoff = frame.captured_at.checked_sub(HISTORY);
        history.push_back(frame);

        while history
            .front()
            .zip(cutoff)
            .is_some_and(|(front, cutoff)| front.captured_at < cutoff)
        {
            history.pop_front();
        }
    }

    fn note_models(&self, detector: Option<&str>, pose: Option<&str>, backend: Option<Backend>) {
        let mut stats = self.stats.lock();
        stats.detector_model = detector.map(str::to_owned);
        stats.pose_model = pose.map(str::to_owned);
        stats.backend = backend;
    }

    fn fail(&self, message: String) {
        self.stats.lock().last_error = Some(message);
    }
}

fn ema(current: f32, sample: f32, alpha: f32) -> f32 {
    if current == 0.0 {
        sample
    } else {
        current * (1.0 - alpha) + sample * alpha
    }
}

/// A change the UI wants applied without restarting the stage.
#[derive(Debug, Clone)]
pub enum PipelineCommand {
    /// Replace the per-camera model assignment and inference settings.
    Configure {
        inference: InferenceConfig,
        cameras: Vec<CameraConfig>,
    },
}

/// Owns the running inference stage.
#[derive(Default)]
pub struct Pipeline {
    channels: HashMap<String, Arc<PoseChannel>>,
    commands: Option<Sender<PipelineCommand>>,
    stop: Shutdown,
}

impl Pipeline {
    pub fn is_running(&self) -> bool {
        self.commands.is_some()
    }

    pub fn channel(&self, camera: &str) -> Option<&Arc<PoseChannel>> {
        self.channels.get(camera)
    }

    /// Starts the stage over the given cameras.
    pub fn start(
        &mut self,
        inference: InferenceConfig,
        cameras: &[CameraConfig],
        capture: &[Arc<CameraChannel>],
        supervisor: &mut Supervisor,
    ) {
        self.stop();

        let mut channels = HashMap::new();
        for channel in capture {
            channels.insert(channel.config.id.clone(), Arc::new(PoseChannel::default()));
        }
        self.channels = channels.clone();

        let (commands_tx, commands_rx) = unbounded();
        self.commands = Some(commands_tx);
        self.stop = Shutdown::default();

        let stop = self.stop.clone();
        let sources: Vec<Arc<CameraChannel>> = capture.to_vec();
        let cameras = cameras.to_vec();

        supervisor.spawn("inference", move |global| {
            run(Worker {
                inference,
                cameras,
                sources,
                channels,
                commands: commands_rx,
                stop,
                global,
            });
        });
    }

    pub fn stop(&mut self) {
        self.stop.cancel();
        self.commands = None;
        self.channels.clear();
    }

    /// Applies new settings without interrupting tracking.
    pub fn configure(&self, inference: InferenceConfig, cameras: &[CameraConfig]) {
        if let Some(commands) = &self.commands {
            let _ = commands.send(PipelineCommand::Configure {
                inference,
                cameras: cameras.to_vec(),
            });
        }
    }
}

struct Worker {
    inference: InferenceConfig,
    cameras: Vec<CameraConfig>,
    sources: Vec<Arc<CameraChannel>>,
    channels: HashMap<String, Arc<PoseChannel>>,
    commands: Receiver<PipelineCommand>,
    stop: Shutdown,
    global: Shutdown,
}

/// Per-camera state that survives across ticks.
#[derive(Default)]
struct CameraState {
    /// Frames processed since the detector last ran.
    since_detection: u32,
    /// The box being tracked between detector runs.
    box_hint: Option<Detection>,
    /// Sequence number of the last frame processed.
    last_seq: u64,
}

fn run(mut worker: Worker) {
    let mut set = ModelSet::new(worker.inference.provider);
    let mut states: HashMap<String, CameraState> = HashMap::new();

    while !worker.stop.is_cancelled() && !worker.global.is_cancelled() {
        for command in worker.commands.try_iter() {
            match command {
                PipelineCommand::Configure { inference, cameras } => {
                    if inference.provider != worker.inference.provider {
                        // A different execution provider means every session
                        // has to be rebuilt, so the whole set is discarded.
                        set = ModelSet::new(inference.provider);
                    }
                    worker.inference = inference;
                    worker.cameras = cameras;
                }
            }
        }

        set.poll();

        let (wanted_detectors, wanted_poses) = assigned_models(&worker);
        set.ensure(&wanted_detectors, &wanted_poses);

        let mut worked = false;
        for source in &worker.sources {
            let camera = match worker
                .cameras
                .iter()
                .find(|config| config.id == source.config.id)
            {
                Some(config) => config,
                None => continue,
            };

            let Some(frame) = source.take() else { continue };
            let state = states.entry(camera.id.clone()).or_default();
            if frame.seq == state.last_seq {
                continue;
            }
            state.last_seq = frame.seq;
            worked = true;

            let detector_id = camera
                .detector_model
                .clone()
                .unwrap_or_else(|| worker.inference.detector_model.clone());
            let pose_id = camera
                .pose_model
                .clone()
                .unwrap_or_else(|| worker.inference.pose_model.clone());

            let channel = worker.channels.get(&camera.id).cloned();
            let Some(channel) = channel else { continue };

            if let Err(err) = process(
                &mut set,
                state,
                &worker.inference,
                &detector_id,
                &pose_id,
                &frame,
                &channel,
            ) {
                tracing::warn!(camera = %camera.id, "inference failed: {err:#}");
                channel.fail(format!("{err:#}"));
            }
        }

        // Models nobody is assigned to any more are dropped, so a swap frees
        // the memory of the model it replaced.
        set.retain(&wanted_detectors, &wanted_poses);

        if !worked {
            // Nothing new to do; yield rather than spin. A camera at 30 fps
            // leaves 33 ms between frames, and burning it is a waste of the
            // same CPU the models want.
            if !worker.stop.sleep(Duration::from_millis(2)) {
                break;
            }
        }
    }

    tracing::info!("inference stopped");
}

/// Every model id some camera currently wants.
fn assigned_models(worker: &Worker) -> (Vec<String>, Vec<String>) {
    let mut detectors = vec![worker.inference.detector_model.clone()];
    let mut poses = vec![worker.inference.pose_model.clone()];

    for camera in &worker.cameras {
        if let Some(id) = &camera.detector_model {
            detectors.push(id.clone());
        }
        if let Some(id) = &camera.pose_model {
            poses.push(id.clone());
        }
    }

    (detectors, poses)
}

/// Runs one camera's frame through detection and pose estimation.
fn process(
    set: &mut ModelSet,
    state: &mut CameraState,
    inference: &InferenceConfig,
    detector_id: &str,
    pose_id: &str,
    frame: &Frame,
    channel: &Arc<PoseChannel>,
) -> anyhow::Result<()> {
    let view = ImageView::new(frame.width, frame.height, &frame.rgb);

    // Detection is the expensive half and the subject is one slowly moving
    // person, so it runs on a stride and the previous keypoints carry the box
    // in between.
    let due = state.since_detection >= inference.detect_every || state.box_hint.is_none();

    let mut detection = state.box_hint;
    let mut backend = None;

    match set.detector(detector_id) {
        Slot::Ready(detector) if due => {
            state.since_detection = 0;
            backend = Some(detector.backend());
            let found = detector.detect(&[view])?;
            detection = found[0]
                .iter()
                .max_by(|a, b| a.score.total_cmp(&b.score))
                .copied();
        }
        Slot::Ready(detector) => {
            state.since_detection += 1;
            backend = Some(detector.backend());
        }
        Slot::Loading => state.since_detection += 1,
        Slot::Failed(err) => {
            let err = err.to_owned();
            channel.fail(format!("{detector_id}: {err}"));
            state.since_detection += 1;
        }
    }

    let Some(person) = detection else {
        channel.publish(PoseFrame {
            seq: frame.seq,
            captured_at: frame.captured_at,
            width: frame.width,
            height: frame.height,
            detection: None,
            keypoints: Keypoints2d::default(),
        });
        state.box_hint = None;
        return Ok(());
    };

    let keypoints = match set.pose(pose_id) {
        Slot::Ready(pose) => {
            backend = Some(pose.backend());
            let estimated = pose.estimate(&[(view, person)])?;
            estimated.into_iter().next().unwrap_or_default()
        }
        Slot::Loading => Keypoints2d::default(),
        Slot::Failed(err) => {
            let err = err.to_owned();
            channel.fail(format!("{pose_id}: {err}"));
            Keypoints2d::default()
        }
    };

    // Between detector runs the box comes from the keypoints themselves,
    // grown a little so a limb leaving the previous box is still covered.
    state.box_hint = keypoints
        .bounds()
        .map(|bounds| grow(&bounds, 0.25, frame.width, frame.height))
        .or(Some(person));

    channel.note_models(Some(detector_id), Some(pose_id), backend);
    channel.publish(PoseFrame {
        seq: frame.seq,
        captured_at: frame.captured_at,
        width: frame.width,
        height: frame.height,
        detection: Some(person),
        keypoints,
    });
    Ok(())
}

fn grow(detection: &Detection, factor: f32, width: u32, height: u32) -> Detection {
    let dx = detection.width() * factor * 0.5;
    let dy = detection.height() * factor * 0.5;
    Detection {
        x1: (detection.x1 - dx).max(0.0),
        y1: (detection.y1 - dy).max(0.0),
        x2: (detection.x2 + dx).min(width as f32),
        y2: (detection.y2 + dy).min(height as f32),
        score: detection.score,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(at: Instant, seq: u64) -> PoseFrame {
        PoseFrame {
            seq,
            captured_at: at,
            width: 640,
            height: 480,
            detection: None,
            keypoints: Keypoints2d::default(),
        }
    }

    #[test]
    fn a_tick_between_two_frames_finds_both() {
        let channel = PoseChannel::default();
        let start = Instant::now();
        for step in 0..5u64 {
            channel.publish(frame(start + Duration::from_millis(step * 20), step));
        }

        let (before, after) = channel
            .bracket(start + Duration::from_millis(31))
            .expect("the tick is inside the history");
        assert_eq!((before.seq, after.seq), (1, 2));
    }

    #[test]
    fn a_tick_landing_on_a_frame_brackets_it_with_itself() {
        let channel = PoseChannel::default();
        let start = Instant::now();
        channel.publish(frame(start, 0));
        channel.publish(frame(start + Duration::from_millis(20), 1));

        let (before, after) = channel.bracket(start).expect("an exact hit is in range");
        assert_eq!((before.seq, after.seq), (0, 0));
    }

    /// A camera that has stopped delivering must not be extrapolated forward.
    #[test]
    fn a_tick_outside_the_history_has_no_bracket() {
        let channel = PoseChannel::default();
        let start = Instant::now() + Duration::from_secs(1);
        channel.publish(frame(start, 0));
        channel.publish(frame(start + Duration::from_millis(20), 1));

        assert!(channel.bracket(start - Duration::from_millis(1)).is_none());
        assert!(channel.bracket(start + Duration::from_millis(21)).is_none());
    }

    #[test]
    fn the_history_forgets_what_has_aged_out() {
        let channel = PoseChannel::default();
        let start = Instant::now();
        for step in 0..40u64 {
            channel.publish(frame(start + Duration::from_millis(step * 100), step));
        }

        let (oldest, newest) = channel.span().expect("frames were published");
        assert!(newest.duration_since(oldest) <= HISTORY);
        assert!(channel.bracket(start + Duration::from_millis(50)).is_none());
    }

    /// A camera that restarts can stamp a frame before the one it just sent.
    /// The history has to stay sorted or every lookup through it is wrong.
    #[test]
    fn a_clock_that_jumps_backwards_restarts_the_history() {
        let channel = PoseChannel::default();
        let start = Instant::now() + Duration::from_secs(1);
        for step in 0..5u64 {
            channel.publish(frame(start + Duration::from_millis(step * 20), step));
        }
        channel.publish(frame(start - Duration::from_millis(500), 99));

        let (oldest, newest) = channel.span().unwrap();
        assert_eq!(oldest, newest, "only the newest frame should survive");
        assert!(channel.bracket(start + Duration::from_millis(30)).is_none());
    }

    #[test]
    fn growing_a_box_stays_inside_the_image() {
        let grown = grow(
            &Detection {
                x1: 5.0,
                y1: 5.0,
                x2: 15.0,
                y2: 25.0,
                score: 0.9,
            },
            1.0,
            20.0 as u32,
            30,
        );
        assert_eq!(grown.x1, 0.0);
        assert_eq!(grown.y1, 0.0);
        assert_eq!(grown.x2, 20.0);
        assert_eq!(grown.y2, 30.0);
    }

    #[test]
    fn growing_by_nothing_changes_nothing() {
        let original = Detection {
            x1: 5.0,
            y1: 6.0,
            x2: 15.0,
            y2: 26.0,
            score: 0.9,
        };
        assert_eq!(grow(&original, 0.0, 100, 100), original);
    }
}
