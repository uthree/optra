//! Frame capture.
//!
//! One thread per camera reads frames, timestamps them on arrival and publishes
//! the latest one into a single-slot mailbox. Nothing is queued: a frame that
//! the pipeline did not pick up before the next one arrived is worthless for
//! real-time tracking, so it is overwritten and counted.

pub mod source;

use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crossbeam_channel::{Receiver, Sender, unbounded};

use crate::config::{CameraConfig, ControlName};
use crate::worker::{Shutdown, Supervisor};
use source::{ControlInfo, ControlSession, FrameSource, NegotiatedFormat};

/// A decoded RGB8 frame.
pub struct FrameData {
    pub width: u32,
    pub height: u32,
    /// Tightly packed RGB8, `width * height * 3` bytes.
    pub rgb: Vec<u8>,
    /// When the frame was handed to us by the source.
    pub captured_at: Instant,
    /// Monotonic per-camera counter, used to detect a new frame.
    pub seq: u64,
}

pub type Frame = Arc<FrameData>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraState {
    Stopped,
    Opening,
    Running,
    Failed,
}

impl CameraState {
    pub fn label(self) -> &'static str {
        match self {
            CameraState::Stopped => "stopped",
            CameraState::Opening => "opening",
            CameraState::Running => "running",
            CameraState::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CameraStats {
    pub state: CameraState,
    pub captured: u64,
    /// Frames replaced in the mailbox before a consumer took them.
    pub overwritten: u64,
    pub errors: u64,
    /// Smoothed capture rate in frames per second.
    pub measured_fps: f32,
    /// Smoothed time spent decoding one frame, in milliseconds.
    pub decode_ms: f32,
    pub last_error: Option<String>,
    pub negotiated: Option<NegotiatedFormat>,
    /// Device properties, as last read from the camera.
    pub controls: Vec<ControlInfo>,
}

/// A request from the UI to the camera thread.
#[derive(Debug, Clone)]
pub enum CameraCommand {
    SetControl {
        name: ControlName,
        value: i64,
        auto: bool,
    },
}

impl Default for CameraStats {
    fn default() -> Self {
        Self {
            state: CameraState::Stopped,
            captured: 0,
            overwritten: 0,
            errors: 0,
            measured_fps: 0.0,
            decode_ms: 0.0,
            last_error: None,
            negotiated: None,
            controls: Vec::new(),
        }
    }
}

/// The runtime side of one configured camera.
pub struct CameraChannel {
    pub config: CameraConfig,
    stop: Shutdown,
    slot: Mutex<Slot>,
    stats: Mutex<CameraStats>,
    /// Smoothing state for the measured frame rate, kept out of `stats` so it
    /// does not travel with the snapshot handed to the UI.
    last_frame_at: Mutex<Option<Instant>>,
    commands: Sender<CameraCommand>,
}

#[derive(Default)]
struct Slot {
    frame: Option<Frame>,
    /// False once a consumer has taken the frame currently in the slot.
    unread: bool,
}

impl CameraChannel {
    fn new(config: CameraConfig) -> (Self, Receiver<CameraCommand>) {
        let (commands, receiver) = unbounded();
        let channel = Self {
            config,
            stop: Shutdown::default(),
            slot: Mutex::new(Slot::default()),
            stats: Mutex::new(CameraStats::default()),
            last_frame_at: Mutex::new(None),
            commands,
        };
        (channel, receiver)
    }

    /// Queues a request for the camera thread. Device properties can only be
    /// touched from the thread that owns the device, so the UI asks rather than
    /// acts.
    pub fn send(&self, command: CameraCommand) {
        if self.commands.send(command).is_err() {
            tracing::debug!(camera = %self.config.id, "the camera thread is gone");
        }
    }

    /// The most recent frame, without consuming it. Used by the preview.
    pub fn peek(&self) -> Option<Frame> {
        self.slot.lock().frame.clone()
    }

    /// The most recent frame if it has not been taken yet. Used by the
    /// pipeline, so that `overwritten` counts frames that were really missed.
    pub fn take(&self) -> Option<Frame> {
        let mut slot = self.slot.lock();
        if !slot.unread {
            return None;
        }
        slot.unread = false;
        slot.frame.clone()
    }

    pub fn stats(&self) -> CameraStats {
        self.stats.lock().clone()
    }

    fn publish(&self, frame: Frame, decode: Duration) {
        let now = frame.captured_at;
        {
            let mut slot = self.slot.lock();
            let missed = slot.unread;
            slot.frame = Some(frame);
            slot.unread = true;
            if missed {
                self.stats.lock().overwritten += 1;
            }
        }

        let mut stats = self.stats.lock();
        stats.captured += 1;
        stats.decode_ms = ema(stats.decode_ms, decode.as_secs_f32() * 1000.0, 0.1);
        drop(stats);

        self.update_rate(now);
    }

    fn update_rate(&self, now: Instant) {
        // Kept outside `stats` so the smoothing state is not part of the
        // snapshot handed to the UI.
        let mut last = self.last_frame_at.lock();
        if let Some(previous) = last.replace(now) {
            let dt = now.duration_since(previous).as_secs_f32();
            if dt > 0.0 {
                let mut stats = self.stats.lock();
                stats.measured_fps = ema(stats.measured_fps, 1.0 / dt, 0.1);
            }
        }
    }

    fn set_state(&self, state: CameraState) {
        self.stats.lock().state = state;
    }

    fn set_negotiated(&self, format: NegotiatedFormat) {
        self.stats.lock().negotiated = Some(format);
    }

    fn fail(&self, message: String) {
        let mut stats = self.stats.lock();
        stats.state = CameraState::Failed;
        stats.errors += 1;
        stats.last_error = Some(message);
    }
}

fn ema(current: f32, sample: f32, alpha: f32) -> f32 {
    if current == 0.0 {
        sample
    } else {
        current * (1.0 - alpha) + sample * alpha
    }
}

/// Owns the running cameras.
#[derive(Default)]
pub struct CaptureManager {
    channels: Vec<Arc<CameraChannel>>,
}

impl CaptureManager {
    pub fn channels(&self) -> &[Arc<CameraChannel>] {
        &self.channels
    }

    pub fn is_running(&self) -> bool {
        !self.channels.is_empty()
    }

    pub fn channel(&self, id: &str) -> Option<&Arc<CameraChannel>> {
        self.channels.iter().find(|c| c.config.id == id)
    }

    /// Starts every enabled camera. Already running cameras are stopped first,
    /// so this doubles as "apply the current configuration".
    pub fn start(&mut self, configs: &[CameraConfig], supervisor: &mut Supervisor) {
        self.stop();

        for config in configs.iter().filter(|c| c.enabled) {
            let (channel, commands) = CameraChannel::new(config.clone());
            let channel = Arc::new(channel);
            self.channels.push(channel.clone());

            let name = format!("capture:{}", config.id);
            supervisor.spawn(name, move |global| run_camera(channel, commands, global));
        }
    }

    /// Signals every camera thread to exit. The threads themselves are joined
    /// by the supervisor.
    pub fn stop(&mut self) {
        for channel in self.channels.drain(..) {
            channel.stop.cancel();
            channel.set_state(CameraState::Stopped);
        }
    }
}

/// Opens the source and streams from it, reopening after a failure until the
/// camera is stopped. A webcam that briefly drops off the bus should recover on
/// its own rather than requiring the user to restart tracking.
fn run_camera(channel: Arc<CameraChannel>, commands: Receiver<CameraCommand>, global: Shutdown) {
    let mut backoff = Duration::from_millis(500);
    let id = channel.config.id.clone();

    while !cancelled(&channel, &global) {
        channel.set_state(CameraState::Opening);

        // Properties are applied before streaming starts, so the very first
        // frames already have the exposure the user configured.
        let controls = source::open_controls(&channel.config);
        if let Some(session) = &controls {
            apply_configured_controls(&channel, session.as_ref());
            channel.stats.lock().controls = session.list();
        }

        match source::open(&channel.config) {
            Ok(mut source) => {
                let format = source.negotiated();
                tracing::info!(camera = %id, "opened: {format}");
                channel.set_negotiated(format);
                channel.set_state(CameraState::Running);
                backoff = Duration::from_millis(500);

                stream(
                    &channel,
                    source.as_mut(),
                    controls.as_deref(),
                    &commands,
                    &global,
                );
            }
            Err(err) => {
                tracing::warn!(camera = %id, "failed to open the camera: {err:#}");
                channel.fail(format!("{err:#}"));
                if !channel.stop.sleep(backoff) {
                    break;
                }
                backoff = (backoff * 2).min(Duration::from_secs(5));
            }
        }
    }

    channel.set_state(CameraState::Stopped);
    tracing::info!(camera = %id, "capture stopped");
}

/// Applies the properties stored in the camera's configuration.
fn apply_configured_controls(channel: &Arc<CameraChannel>, session: &dyn ControlSession) {
    for setting in &channel.config.controls {
        if let Err(err) = session.set(setting.name, setting.value, setting.auto) {
            tracing::warn!(camera = %channel.config.id, "{err:#}");
        }
    }
}

/// How often the property values are re-read while streaming. Properties the
/// device regulates itself drift, and the UI should show that, but each read is
/// a driver round trip so it is not done per frame.
const CONTROL_REFRESH: Duration = Duration::from_secs(2);

/// Reads frames until the camera is stopped or the source fails.
fn stream(
    channel: &Arc<CameraChannel>,
    source: &mut dyn FrameSource,
    controls: Option<&dyn ControlSession>,
    commands: &Receiver<CameraCommand>,
    global: &Shutdown,
) {
    let mut seq = 0u64;
    let mut refreshed = Instant::now();

    while !cancelled(channel, global) {
        for command in commands.try_iter() {
            let Some(session) = controls else {
                tracing::warn!(
                    camera = %channel.config.id,
                    "this camera has no controllable properties"
                );
                continue;
            };
            match command {
                CameraCommand::SetControl { name, value, auto } => {
                    match session.set(name, value, auto) {
                        // Read the property back rather than assuming the write
                        // landed: devices clamp, round to their step size, and
                        // sometimes refuse outright.
                        Ok(()) => update_control(channel, session, name),
                        Err(err) => tracing::warn!(camera = %channel.config.id, "{err:#}"),
                    }
                }
            }
        }

        if let Some(session) = controls
            && refreshed.elapsed() >= CONTROL_REFRESH
        {
            channel.stats.lock().controls = session.list();
            refreshed = Instant::now();
        }

        match source.next_frame() {
            Ok(raw) => {
                let decode = raw.decode;
                let raw = source::rotate(raw, channel.config.rotation);
                let captured_at = Instant::now();
                seq += 1;
                channel.publish(
                    Arc::new(FrameData {
                        width: raw.width,
                        height: raw.height,
                        rgb: raw.rgb,
                        captured_at,
                        seq,
                    }),
                    decode,
                );
            }
            Err(err) => {
                tracing::warn!(camera = %channel.config.id, "capture failed: {err:#}");
                channel.fail(format!("{err:#}"));
                return;
            }
        }
    }
}

fn cancelled(channel: &Arc<CameraChannel>, global: &Shutdown) -> bool {
    channel.stop.is_cancelled() || global.is_cancelled()
}

/// Re-reads one property into the published stats.
fn update_control(channel: &Arc<CameraChannel>, session: &dyn ControlSession, name: ControlName) {
    let Some(info) = session.get(name) else {
        return;
    };

    let mut stats = channel.stats.lock();
    match stats.controls.iter_mut().find(|c| c.name == name) {
        Some(existing) => *existing = info,
        None => stats.controls.push(info),
    }
}
