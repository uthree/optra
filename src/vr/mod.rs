//! The link to SteamVR.
//!
//! Calibration needs to know where the headset and controllers were at the
//! instant each camera frame was taken. A background thread samples the runtime
//! at a fixed rate and keeps a short history, and consumers ask for a pose at a
//! time rather than for the latest one — a webcam frame and a pose sample never
//! land on the same instant, and rounding them together is worth several
//! centimetres of error during a walk.

pub mod api;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use nalgebra::{Isometry3, Matrix3, Rotation3, Translation3, UnitQuaternion, Vector3};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::config::VrConfig;
use crate::worker::timing::Ticker;
use crate::worker::{Shutdown, Supervisor};

/// What a tracked device is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Head,
    LeftHand,
    RightHand,
    /// A standalone tracker, which Optra reads but does not calibrate against.
    Tracker,
    Other,
}

impl Role {
    pub fn label(self) -> &'static str {
        match self {
            Role::Head => "head",
            Role::LeftHand => "left hand",
            Role::RightHand => "right hand",
            Role::Tracker => "tracker",
            Role::Other => "other",
        }
    }

    /// Which calibration rig this device drives, if any.
    ///
    /// These indices are the `rig` field of a calibration sighting: each one
    /// carries its own offset from the device to the keypoint that stands in
    /// for it.
    pub fn rig(self) -> Option<usize> {
        match self {
            Role::Head => Some(0),
            Role::LeftHand => Some(1),
            Role::RightHand => Some(2),
            _ => None,
        }
    }

    /// Rigs the calibration solves for, in `rig` order.
    pub const RIGS: [Role; 3] = [Role::Head, Role::LeftHand, Role::RightHand];
}

/// One device at one instant.
#[derive(Debug, Clone)]
pub struct DevicePose {
    pub index: u32,
    pub role: Role,
    /// Device-to-world, in the standing universe: right-handed, +Y up, metres,
    /// which is Optra's world frame unchanged.
    pub pose: Isometry3<f64>,
    /// Whether the runtime considers this pose usable right now.
    pub tracking: bool,
    pub serial: String,
    pub model: String,
}

/// Every device at one instant.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub taken_at: Instant,
    pub devices: Vec<DevicePose>,
}

impl Snapshot {
    pub fn device(&self, role: Role) -> Option<&DevicePose> {
        self.devices.iter().find(|device| device.role == role)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkState {
    Stopped,
    /// No runtime found on this machine, or SteamVR is not running.
    Searching,
    Connected,
}

impl LinkState {
    pub fn label(self) -> &'static str {
        match self {
            LinkState::Stopped => "stopped",
            LinkState::Searching => "searching",
            LinkState::Connected => "connected",
        }
    }
}

#[derive(Debug, Clone)]
pub struct LinkStats {
    pub state: LinkState,
    /// Whether a SteamVR runtime exists on this machine at all. Distinguishing
    /// this from "not running" is the difference between two very different
    /// things for the user to do about it.
    pub installed: bool,
    pub runtime: Option<PathBuf>,
    pub last_error: Option<String>,
    pub samples: u64,
    /// Smoothed sampling rate, in hertz.
    pub measured_hz: f32,
    pub devices: usize,
}

impl Default for LinkStats {
    fn default() -> Self {
        Self {
            state: LinkState::Stopped,
            installed: false,
            runtime: None,
            last_error: None,
            samples: 0,
            measured_hz: 0.0,
            devices: 0,
        }
    }
}

/// The shared face of the VR thread.
pub struct VrChannel {
    stop: Shutdown,
    stats: Mutex<LinkStats>,
    history: Mutex<VecDeque<Snapshot>>,
    /// How much history to keep, in seconds.
    window: f32,
}

impl VrChannel {
    fn new(window: f32) -> Self {
        Self {
            stop: Shutdown::default(),
            stats: Mutex::new(LinkStats::default()),
            history: Mutex::new(VecDeque::new()),
            window,
        }
    }

    pub fn stats(&self) -> LinkStats {
        self.stats.lock().clone()
    }

    /// The most recent snapshot, for the UI.
    pub fn latest(&self) -> Option<Snapshot> {
        self.history.lock().back().cloned()
    }

    /// Where a device was at a given instant.
    ///
    /// Returns `None` when the instant falls outside the recorded history,
    /// rather than extrapolating: a pose invented beyond the ends of a walk is
    /// exactly the kind of thing that quietly poisons a calibration.
    pub fn pose_at(&self, role: Role, at: Instant) -> Option<Isometry3<f64>> {
        let history = self.history.lock();
        interpolate(&history, role, at)
    }

    /// Every snapshot taken after the given instant, oldest first.
    ///
    /// This is how a recording keeps a complete track without sampling the
    /// runtime a second time: it asks for whatever has arrived since it last
    /// looked. As long as it looks more often than the history window is long,
    /// nothing is missed.
    pub fn since(&self, after: Instant) -> Vec<Snapshot> {
        self.history
            .lock()
            .iter()
            .filter(|snapshot| snapshot.taken_at > after)
            .cloned()
            .collect()
    }

    /// Whether a device is being tracked right now.
    pub fn is_tracking(&self, role: Role) -> bool {
        self.history
            .lock()
            .back()
            .and_then(|snapshot| snapshot.device(role).map(|device| device.tracking))
            .unwrap_or(false)
    }

    fn push(&self, snapshot: Snapshot) {
        let mut history = self.history.lock();
        let horizon = Duration::from_secs_f32(self.window);

        while let Some(front) = history.front() {
            if snapshot.taken_at.duration_since(front.taken_at) > horizon {
                history.pop_front();
            } else {
                break;
            }
        }

        history.push_back(snapshot);
    }

    fn set_state(&self, state: LinkState) {
        self.stats.lock().state = state;
    }
}

/// Finds the two samples bracketing `at` and blends between them.
fn interpolate(history: &VecDeque<Snapshot>, role: Role, at: Instant) -> Option<Isometry3<f64>> {
    let mut before: Option<(Instant, Isometry3<f64>)> = None;
    let mut after: Option<(Instant, Isometry3<f64>)> = None;

    for snapshot in history {
        let Some(device) = snapshot.device(role) else {
            continue;
        };
        if !device.tracking {
            continue;
        }

        if snapshot.taken_at <= at {
            before = Some((snapshot.taken_at, device.pose));
        } else {
            after = Some((snapshot.taken_at, device.pose));
            break;
        }
    }

    match (before, after) {
        (Some(a), Some(b)) => Some(blend(a, b, at)),
        // Exactly at or after the last sample, with nothing following it: only
        // usable if it is the last sample itself.
        (Some((t0, a)), None) if t0 == at => Some(a),
        _ => None,
    }
}

/// Blends two samples to the instant between them.
///
/// Position moves in a straight line and orientation along the shorter arc,
/// which is what `lerp_slerp` does; over the few milliseconds between samples
/// nothing more elaborate is measurable.
fn blend(
    (t0, a): (Instant, Isometry3<f64>),
    (t1, b): (Instant, Isometry3<f64>),
    at: Instant,
) -> Isometry3<f64> {
    let span = t1.duration_since(t0).as_secs_f64();
    if span <= f64::EPSILON {
        return a;
    }
    a.lerp_slerp(&b, at.duration_since(t0).as_secs_f64() / span)
}

/// One device's path, kept for as long as a recording needs it.
///
/// The link's own history is a few seconds deep, which is right for pairing a
/// camera frame with a pose but useless once a calibration walk is over. A
/// recording copies what it needs into one of these, so the whole walk can be
/// re-read afterwards — which is what estimating each camera's latency needs,
/// since that means asking where the headset was at a range of shifted times.
#[derive(Debug, Clone, Default)]
pub struct Track {
    samples: Vec<(Instant, Isometry3<f64>)>,
}

impl Track {
    /// Appends a sample. Samples that are not newer than the last one are
    /// dropped, so the track stays sorted and free of repeats.
    pub fn push(&mut self, at: Instant, pose: Isometry3<f64>) -> bool {
        if let Some((last, _)) = self.samples.last()
            && at <= *last
        {
            return false;
        }
        self.samples.push((at, pose));
        true
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn first_at(&self) -> Option<Instant> {
        self.samples.first().map(|(at, _)| *at)
    }

    pub fn last_at(&self) -> Option<Instant> {
        self.samples.last().map(|(at, _)| *at)
    }

    pub fn span(&self) -> Duration {
        match (self.first_at(), self.last_at()) {
            (Some(first), Some(last)) => last.duration_since(first),
            _ => Duration::ZERO,
        }
    }

    /// Where the device was at an instant, or `None` outside the track.
    pub fn at(&self, at: Instant) -> Option<Isometry3<f64>> {
        if self.samples.len() < 2 {
            return self
                .samples
                .first()
                .filter(|(t, _)| *t == at)
                .map(|(_, pose)| *pose);
        }

        // The track is sorted, so the bracketing pair is one search away rather
        // than a scan; a walk holds tens of thousands of samples and this is
        // asked once per recorded keypoint.
        match self.samples.binary_search_by(|(t, _)| t.cmp(&at)) {
            Ok(index) => Some(self.samples[index].1),
            Err(0) => None,
            Err(index) if index == self.samples.len() => None,
            Err(index) => Some(blend(self.samples[index - 1], self.samples[index], at)),
        }
    }

    /// How well the device turned during the recording, from zero to one.
    ///
    /// The same question [`refine::offset_observability`] answers, asked of a
    /// track rather than of sightings, so the wizard can warn while the user is
    /// still walking rather than after the solve.
    ///
    /// [`refine::offset_observability`]: crate::geometry::refine::offset_observability
    pub fn rotation_spread(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }

        let mut sum = Matrix3::zeros();
        for (_, pose) in &self.samples {
            sum += pose.rotation.to_rotation_matrix().into_inner();
        }

        let mean = sum / self.samples.len() as f64;
        let smallest = mean
            .svd(false, false)
            .singular_values
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);

        (1.0 - smallest).clamp(0.0, 1.0)
    }
}

/// Owns the VR thread, if one is running.
#[derive(Default)]
pub struct VrLink {
    channel: Option<Arc<VrChannel>>,
}

impl VrLink {
    pub fn channel(&self) -> Option<&Arc<VrChannel>> {
        self.channel.as_ref()
    }

    pub fn is_running(&self) -> bool {
        self.channel.is_some()
    }

    pub fn start(&mut self, config: &VrConfig, supervisor: &mut Supervisor) {
        self.stop();
        if !config.enabled {
            return;
        }

        let channel = Arc::new(VrChannel::new(config.history_seconds));
        self.channel = Some(channel.clone());

        let config = config.clone();
        supervisor.spawn("vr", move |global| run(channel, config, global));
    }

    pub fn stop(&mut self) {
        if let Some(channel) = self.channel.take() {
            channel.stop.cancel();
            channel.set_state(LinkState::Stopped);
        }
    }
}

/// Connects, samples until something goes wrong, then connects again.
fn run(channel: Arc<VrChannel>, config: VrConfig, global: Shutdown) {
    let retry = Duration::from_secs_f32(config.retry_seconds.max(0.5));

    while !cancelled(&channel, &global) {
        channel.stats.lock().installed = api::is_installed();
        channel.set_state(LinkState::Searching);

        match api::Runtime::connect() {
            Ok(runtime) => {
                {
                    let mut stats = channel.stats.lock();
                    stats.state = LinkState::Connected;
                    stats.runtime = Some(runtime.path().to_path_buf());
                    stats.last_error = None;
                }
                tracing::info!(runtime = %runtime.path().display(), "connected to SteamVR");

                sample(&channel, &runtime, &config, &global);

                tracing::info!("the SteamVR connection ended");
            }
            Err(error) => {
                // Not an error worth shouting about: running Optra to set
                // cameras up with SteamVR closed is a normal thing to do.
                tracing::debug!(%error, "no SteamVR connection");
                channel.stats.lock().last_error = Some(format!("{error:#}"));
            }
        }

        channel.set_state(LinkState::Searching);
        if !wait(&channel, &global, retry) {
            break;
        }
    }

    channel.set_state(LinkState::Stopped);
}

/// Samples poses until the headset goes away or the thread is stopped.
fn sample(channel: &Arc<VrChannel>, runtime: &api::Runtime, config: &VrConfig, global: &Shutdown) {
    let mut ticker = Ticker::at_hz(config.poll_hz.max(1) as f32);
    let patience = Duration::from_secs_f32(config.patience_seconds.max(1.0));

    // Identity is asked of the runtime once per device rather than every tick:
    // the strings do not change, and the call is not free.
    let mut identity: Vec<Option<(String, String)>> = vec![None; api::MAX_TRACKED_DEVICES];
    let mut last_seen = Instant::now();
    let mut last_tick: Option<Instant> = None;

    while !cancelled(channel, global) {
        let now = Instant::now();
        let poses = runtime.poses(0.0);
        let mut devices = Vec::new();

        for (index, pose) in poses.iter().enumerate() {
            if !pose.device_is_connected {
                continue;
            }

            let index = index as u32;
            let class = runtime.device_class(index);
            let role = classify(runtime, index, class);
            if role == Role::Other {
                continue;
            }

            let slot = &mut identity[index as usize];
            let (serial, model) = slot
                .get_or_insert_with(|| (runtime.serial(index), runtime.model(index)))
                .clone();

            devices.push(DevicePose {
                index,
                role,
                pose: to_isometry(&pose.device_to_absolute_tracking),
                tracking: pose.pose_is_valid,
                serial,
                model,
            });
        }

        if devices.iter().any(|device| device.role == Role::Head) {
            last_seen = now;
        } else if now.duration_since(last_seen) > patience {
            // SteamVR does not report its own exit through this interface, so
            // the headset simply stopping being connected is the signal to drop
            // the runtime and start looking again.
            channel.stats.lock().last_error = Some("the headset stopped reporting".to_owned());
            return;
        }

        {
            let mut stats = channel.stats.lock();
            stats.samples += 1;
            stats.devices = devices.len();
            if let Some(previous) = last_tick {
                let elapsed = now.duration_since(previous).as_secs_f32();
                if elapsed > 0.0 {
                    let instant = 1.0 / elapsed;
                    stats.measured_hz = if stats.measured_hz == 0.0 {
                        instant
                    } else {
                        stats.measured_hz * 0.9 + instant * 0.1
                    };
                }
            }
        }
        last_tick = Some(now);

        channel.push(Snapshot {
            taken_at: now,
            devices,
        });

        // The ticker keeps the schedule rather than the interval, so the time
        // this pass spent talking to the runtime comes out of the next sleep
        // instead of being added to it.
        if !ticker.wait(&channel.stop) || global.is_cancelled() {
            return;
        }
    }
}

fn classify(runtime: &api::Runtime, index: u32, class: i32) -> Role {
    match class {
        api::CLASS_HMD if index == api::DEVICE_INDEX_HMD => Role::Head,
        api::CLASS_HMD => Role::Other,
        api::CLASS_CONTROLLER => match runtime.controller_role(index) {
            api::ROLE_LEFT_HAND => Role::LeftHand,
            api::ROLE_RIGHT_HAND => Role::RightHand,
            _ => Role::Other,
        },
        api::CLASS_TRACKER => Role::Tracker,
        _ => Role::Other,
    }
}

/// OpenVR hands out a row-major 3x4 with the position in the last column.
fn to_isometry(matrix: &api::HmdMatrix34) -> Isometry3<f64> {
    let m = &matrix.m;
    let rotation = Matrix3::new(
        m[0][0] as f64,
        m[0][1] as f64,
        m[0][2] as f64,
        m[1][0] as f64,
        m[1][1] as f64,
        m[1][2] as f64,
        m[2][0] as f64,
        m[2][1] as f64,
        m[2][2] as f64,
    );
    let translation = Vector3::new(m[0][3] as f64, m[1][3] as f64, m[2][3] as f64);

    Isometry3::from_parts(
        Translation3::from(translation),
        UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(rotation)),
    )
}

fn cancelled(channel: &Arc<VrChannel>, global: &Shutdown) -> bool {
    channel.stop.is_cancelled() || global.is_cancelled()
}

/// Sleeps unless either shutdown signal fires first.
fn wait(channel: &Arc<VrChannel>, global: &Shutdown, duration: Duration) -> bool {
    channel.stop.sleep(duration) && !global.is_cancelled()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;

    fn pose(x: f64, yaw: f64) -> Isometry3<f64> {
        Isometry3::from_parts(
            Translation3::new(x, 1.5, 0.0),
            UnitQuaternion::from_euler_angles(0.0, yaw, 0.0),
        )
    }

    fn snapshot(at: Instant, x: f64, yaw: f64, tracking: bool) -> Snapshot {
        Snapshot {
            taken_at: at,
            devices: vec![DevicePose {
                index: 0,
                role: Role::Head,
                pose: pose(x, yaw),
                tracking,
                serial: "test".into(),
                model: "test".into(),
            }],
        }
    }

    fn history(tracking: bool) -> (Instant, VecDeque<Snapshot>) {
        let start = Instant::now();
        let mut history = VecDeque::new();
        for step in 0..5 {
            history.push_back(snapshot(
                start + Duration::from_millis(step * 10),
                step as f64 * 0.1,
                step as f64 * 0.2,
                tracking,
            ));
        }
        (start, history)
    }

    #[test]
    fn a_pose_between_two_samples_is_blended() {
        let (start, history) = history(true);

        let middle = interpolate(&history, Role::Head, start + Duration::from_millis(15))
            .expect("the instant is inside the recording");

        assert!(
            (middle.translation.vector.x - 0.15).abs() < 1e-9,
            "got x = {}",
            middle.translation.vector.x
        );
        let yaw = middle.rotation.euler_angles().1;
        assert!((yaw - 0.3).abs() < 1e-9, "got yaw = {yaw}");
    }

    #[test]
    fn a_pose_on_a_sample_is_that_sample() {
        let (start, history) = history(true);
        let exact = interpolate(&history, Role::Head, start + Duration::from_millis(20)).unwrap();
        assert!((exact.translation.vector.x - 0.2).abs() < 1e-12);
    }

    /// A camera frame that arrived before the recording started, or after it
    /// stopped, has no pose to pair with. Inventing one is worse than skipping
    /// the frame.
    #[test]
    fn instants_outside_the_recording_have_no_pose() {
        let (start, history) = history(true);
        assert!(interpolate(&history, Role::Head, start - Duration::from_millis(5)).is_none());
        assert!(interpolate(&history, Role::Head, start + Duration::from_millis(80)).is_none());
    }

    #[test]
    fn untracked_samples_are_not_used() {
        let (start, history) = history(false);
        assert!(interpolate(&history, Role::Head, start + Duration::from_millis(15)).is_none());
    }

    #[test]
    fn a_role_that_was_never_seen_has_no_pose() {
        let (start, history) = history(true);
        assert!(interpolate(&history, Role::LeftHand, start + Duration::from_millis(15)).is_none());
    }

    #[test]
    fn history_is_trimmed_to_its_window() {
        let channel = VrChannel::new(0.05);
        let start = Instant::now();

        for step in 0..20 {
            channel.push(snapshot(
                start + Duration::from_millis(step * 10),
                0.0,
                0.0,
                true,
            ));
        }

        let held = channel.history.lock().len();
        assert!(
            (5..=7).contains(&held),
            "kept {held} samples for a 50 ms window at 100 Hz"
        );
    }

    /// OpenVR gives a row-major matrix with the position in the last column.
    /// Reading it as column-major would put the headset in the wrong place and
    /// look almost plausible while doing it.
    #[test]
    fn the_openvr_matrix_is_read_row_major() {
        // A quarter turn about Y, standing at (1, 1.6, -2).
        let matrix = api::HmdMatrix34 {
            m: [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 1.6],
                [-1.0, 0.0, 0.0, -2.0],
            ],
        };

        let isometry = to_isometry(&matrix);
        assert!((isometry.translation.vector - Vector3::new(1.0, 1.6, -2.0)).norm() < 1e-6);

        // The device's own -Z, which is where a headset faces, should point
        // along -X after a quarter turn.
        let facing = isometry.rotation * Vector3::new(0.0, 0.0, -1.0);
        assert!(
            (facing - Vector3::new(-1.0, 0.0, 0.0)).norm() < 1e-6,
            "the headset faces {facing:?}"
        );

        // And the point one metre in front of it lands beside the headset,
        // not in front of it, which is what a wrongly read matrix would give.
        let ahead = isometry * Point3::new(0.0, 0.0, -1.0);
        assert!((ahead - Point3::new(0.0, 1.6, -2.0)).norm() < 1e-6);
    }

    #[test]
    fn every_calibration_rig_has_a_distinct_index() {
        let indices: Vec<Option<usize>> = Role::RIGS.iter().map(|role| role.rig()).collect();
        assert_eq!(indices, vec![Some(0), Some(1), Some(2)]);
        assert_eq!(Role::Tracker.rig(), None);
    }
}
