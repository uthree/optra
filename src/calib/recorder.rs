//! Recording a calibration walk.
//!
//! While the user walks the room, this pairs each camera's keypoints with where
//! the matching tracked device was at the instant that frame was *captured* —
//! not when it was processed, which is tens of milliseconds later and enough to
//! bend the answer.
//!
//! What comes out is a [`Recording`]: one pixel trail per camera, plus the full
//! device tracks. The tracks are kept in their entirety rather than collapsed
//! into pose-per-sample, because estimating each camera's latency means asking
//! where the headset was at a range of shifted times, and that question cannot
//! be answered from a track that has already been sampled away.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use nalgebra::Point2;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::models::keypoints::Joint;
use crate::pipeline::PoseChannel;
use crate::vr::{Role, Track, VrChannel};
use crate::worker::timing::Ticker;
use crate::worker::{Shutdown, Supervisor};

/// A tracked device and the keypoint that stands in for it.
///
/// The pairing has to be this specific. A headset does not sit on the head
/// keypoint, and the constant gap between them is solved for — but "the head
/// keypoint" is not one point: a Halpe model reports a head centre while a COCO
/// model reports the nose, and those are several centimetres apart. Two cameras
/// running different models therefore need two offsets, so the rig is the pair
/// rather than the device alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rig {
    pub role: Role,
    pub joint: Joint,
}

impl Rig {
    /// Keypoints that may stand in for a device, best first.
    ///
    /// The head is tried before the nose because it is the more stable point
    /// under a headset, which hides most of the face from a ceiling camera.
    fn candidates(role: Role) -> &'static [Joint] {
        match role {
            Role::Head => &[Joint::Head, Joint::Nose],
            Role::LeftHand => &[Joint::LeftWrist],
            Role::RightHand => &[Joint::RightWrist],
            _ => &[],
        }
    }

    pub fn label(&self) -> String {
        format!("{} / {}", self.role.label(), self.joint.name())
    }
}

#[derive(Debug, Clone)]
pub struct RecorderConfig {
    /// Keypoint confidence below which a sighting is not recorded.
    pub min_confidence: f32,
    /// How far a keypoint must have moved, in pixels, before another sample of
    /// the same camera is kept. A user standing still otherwise contributes
    /// hundreds of identical rows that weight the solve toward one spot and
    /// tell it nothing.
    pub min_pixel_step: f32,
    /// Record the controllers as well as the headset. They reach heights the
    /// head never does, which is what stops the correspondences forming a
    /// plane, but the wrist keypoint sits less rigidly against a controller
    /// than the head does against a headset.
    pub use_controllers: bool,
    /// Weight given to controller sightings relative to the headset.
    pub controller_weight: f32,
    /// Stop accepting samples for a camera once it has this many. A longer walk
    /// past this point adds cost without adding information.
    pub max_per_camera: usize,
    /// How often the recorder looks for new frames. Faster than any camera
    /// runs, so nothing is missed; the pose channel holds one frame at a time.
    pub poll_hz: f32,
}

impl Default for RecorderConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            min_pixel_step: 6.0,
            use_controllers: true,
            controller_weight: 0.5,
            max_per_camera: 4000,
            poll_hz: 90.0,
        }
    }
}

/// One camera's sighting of a rig at one instant.
#[derive(Debug, Clone, Copy)]
pub struct Sample {
    /// When the frame was captured.
    pub at: Instant,
    pub rig: usize,
    pub pixel: Point2<f64>,
    pub confidence: f64,
}

/// How much of a camera's frame the walk covered.
///
/// Tracked per camera rather than globally: a narrow camera sees a smaller
/// slice of the room, so a walk that satisfies a wide one can leave a narrow
/// one under-constrained, and the wizard has to be able to say which.
#[derive(Debug, Clone)]
pub struct Coverage {
    pub columns: usize,
    pub rows: usize,
    cells: Vec<u32>,
}

impl Default for Coverage {
    fn default() -> Self {
        Self::new()
    }
}

impl Coverage {
    pub const COLUMNS: usize = 8;
    pub const ROWS: usize = 6;

    pub fn new() -> Self {
        Self {
            columns: Self::COLUMNS,
            rows: Self::ROWS,
            cells: vec![0; Self::COLUMNS * Self::ROWS],
        }
    }

    fn record(&mut self, pixel: Point2<f64>, width: u32, height: u32) {
        let Some((u, v)) = normalized(pixel, width, height) else {
            return;
        };

        // Bounded before the cast, not after: casting a small negative to an
        // integer truncates toward zero, so a keypoint just off the left edge
        // would otherwise be counted in the leftmost column.
        let column = ((u * self.columns as f64) as usize).min(self.columns - 1);
        let row = ((v * self.rows as f64) as usize).min(self.rows - 1);
        self.cells[row * self.columns + column] += 1;
    }

    pub fn count(&self, column: usize, row: usize) -> u32 {
        self.cells
            .get(row * self.columns + column)
            .copied()
            .unwrap_or(0)
    }

    /// Fraction of the frame that saw at least a handful of samples.
    pub fn filled(&self) -> f32 {
        const ENOUGH: u32 = 3;
        let filled = self.cells.iter().filter(|count| **count >= ENOUGH).count();
        filled as f32 / self.cells.len() as f32
    }
}

/// A pixel as a fraction of the frame, or `None` if it is outside it.
fn normalized(pixel: Point2<f64>, width: u32, height: u32) -> Option<(f64, f64)> {
    if width == 0 || height == 0 {
        return None;
    }

    let u = pixel.x / width as f64;
    let v = pixel.y / height as f64;
    ((0.0..1.0).contains(&u) && (0.0..1.0).contains(&v)).then_some((u, v))
}

/// What one camera contributed.
#[derive(Debug, Clone)]
pub struct CameraTrail {
    pub camera: String,
    /// Frame size the pixels refer to.
    pub width: u32,
    pub height: u32,
    pub samples: Vec<Sample>,
    pub coverage: Coverage,
    /// Frames seen but not kept, and why.
    pub rejected_confidence: u64,
    pub rejected_stationary: u64,
    pub rejected_no_pose: u64,
    pub rejected_offscreen: u64,
}

impl CameraTrail {
    pub fn new(camera: String) -> Self {
        Self {
            camera,
            width: 0,
            height: 0,
            samples: Vec::new(),
            coverage: Coverage::new(),
            rejected_confidence: 0,
            rejected_stationary: 0,
            rejected_no_pose: 0,
            rejected_offscreen: 0,
        }
    }

    /// Adds a sample and counts it toward the coverage map.
    pub fn record(&mut self, sample: Sample) {
        self.coverage.record(sample.pixel, self.width, self.height);
        self.samples.push(sample);
    }

    pub fn samples_for(&self, rig: usize) -> usize {
        self.samples.iter().filter(|s| s.rig == rig).count()
    }
}

/// Everything a walk produced.
#[derive(Debug, Clone, Default)]
pub struct Recording {
    /// The rigs that were actually seen, in the order sightings index them.
    pub rigs: Vec<Rig>,
    /// One track per rig, in the same order.
    pub tracks: Vec<Track>,
    pub cameras: Vec<CameraTrail>,
    pub duration: Duration,
}

impl Recording {
    pub fn samples(&self) -> usize {
        self.cameras.iter().map(|trail| trail.samples.len()).sum()
    }

    pub fn trail(&self, camera: &str) -> Option<&CameraTrail> {
        self.cameras.iter().find(|trail| trail.camera == camera)
    }

    /// How well each rig turned during the walk.
    ///
    /// A rig that never rotated cannot have its offset separated from a shift
    /// of every camera, and the walk needs doing again rather than the solve
    /// needing running.
    pub fn observability(&self) -> Vec<(Rig, f64)> {
        self.rigs
            .iter()
            .zip(&self.tracks)
            .map(|(rig, track)| (*rig, track.rotation_spread()))
            .collect()
    }
}

/// Live progress, for the wizard to show while the user is still walking.
#[derive(Debug, Clone, Default)]
pub struct RecorderStats {
    pub recording: bool,
    pub elapsed: Duration,
    pub samples: usize,
    /// Per camera: id, samples kept, fraction of the frame covered.
    pub cameras: Vec<(String, usize, f32)>,
    /// Per rig: what it is, and how well it has turned so far.
    pub rigs: Vec<(Rig, f64)>,
    /// Set when the walk cannot be used, and why.
    pub warning: Option<String>,
}

/// The shared face of the recorder thread.
pub struct RecordChannel {
    stop: Shutdown,
    stats: Mutex<RecorderStats>,
    recording: Mutex<Recording>,
}

impl RecordChannel {
    pub fn stats(&self) -> RecorderStats {
        self.stats.lock().clone()
    }

    /// The walk so far, copied out. Cheap enough to call when the user stops.
    pub fn recording(&self) -> Recording {
        self.recording.lock().clone()
    }
}

/// Owns the recorder thread, if one is running.
#[derive(Default)]
pub struct Recorder {
    channel: Option<Arc<RecordChannel>>,
}

impl Recorder {
    pub fn channel(&self) -> Option<&Arc<RecordChannel>> {
        self.channel.as_ref()
    }

    pub fn is_recording(&self) -> bool {
        self.channel.is_some()
    }

    /// Starts recording from the given cameras.
    pub fn start(
        &mut self,
        config: RecorderConfig,
        cameras: Vec<(String, Arc<PoseChannel>)>,
        vr: Arc<VrChannel>,
        supervisor: &mut Supervisor,
    ) {
        self.stop();

        let channel = Arc::new(RecordChannel {
            stop: Shutdown::default(),
            stats: Mutex::new(RecorderStats {
                recording: true,
                ..RecorderStats::default()
            }),
            recording: Mutex::new(Recording::default()),
        });
        self.channel = Some(channel.clone());

        supervisor.spawn("calib:record", move |global| {
            run(channel, config, cameras, vr, global)
        });
    }

    /// Stops the thread and hands back what it collected.
    pub fn finish(&mut self) -> Option<Recording> {
        let channel = self.channel.take()?;
        channel.stop.cancel();

        let recording = channel.recording();
        channel.stats.lock().recording = false;
        Some(recording)
    }

    pub fn stop(&mut self) {
        let _ = self.finish();
    }
}

fn run(
    channel: Arc<RecordChannel>,
    config: RecorderConfig,
    cameras: Vec<(String, Arc<PoseChannel>)>,
    vr: Arc<VrChannel>,
    global: Shutdown,
) {
    let started = Instant::now();
    let mut ticker = Ticker::at_hz(config.poll_hz);

    let mut state = State::new(&config, &cameras);
    // Poses that arrived before the walk began are of no interest.
    let mut track_cursor = started;

    while !channel.stop.is_cancelled() && !global.is_cancelled() {
        // Take the pose track first, so every frame processed below can find
        // the poses bracketing it.
        for snapshot in vr.since(track_cursor) {
            track_cursor = track_cursor.max(snapshot.taken_at);
            for device in &snapshot.devices {
                if device.tracking {
                    state.push_pose(device.role, snapshot.taken_at, device.pose);
                }
            }
        }

        for (index, (_, poses)) in cameras.iter().enumerate() {
            let Some(frame) = poses.peek() else { continue };
            state.consider(index, &frame, &config);
        }

        state.publish(&channel, started);

        if !ticker.wait(&channel.stop) || global.is_cancelled() {
            break;
        }
    }

    state.publish(&channel, started);
    tracing::info!(
        samples = state.recording.samples(),
        "calibration recording finished"
    );
}

/// The recorder's working state, kept out of `run` so it can be tested without
/// threads.
struct State {
    recording: Recording,
    rig_index: HashMap<Rig, usize>,
    /// Last frame taken from each camera, so a frame is never counted twice.
    last_seq: Vec<Option<u64>>,
    /// Last accepted pixel per camera and rig, for the stationary check.
    last_pixel: HashMap<(usize, usize), Point2<f64>>,
    /// Poses recorded before a rig existed. A rig only comes into being once
    /// its keypoint has been seen, and the keypoint may have been out of frame
    /// for the first part of the walk; without this those poses are lost and
    /// the earliest sightings of that rig have nothing to pair with.
    pending: HashMap<Role, Track>,
}

impl State {
    fn new(_config: &RecorderConfig, cameras: &[(String, Arc<PoseChannel>)]) -> Self {
        Self {
            recording: Recording {
                cameras: cameras
                    .iter()
                    .map(|(id, _)| CameraTrail::new(id.clone()))
                    .collect(),
                ..Recording::default()
            },
            rig_index: HashMap::new(),
            last_seq: vec![None; cameras.len()],
            last_pixel: HashMap::new(),
            pending: HashMap::new(),
        }
    }

    fn push_pose(&mut self, role: Role, at: Instant, pose: nalgebra::Isometry3<f64>) {
        // A track exists per rig, and a rig is only created once a keypoint has
        // been paired with it, so the pose goes to every rig of this role.
        for (index, rig) in self.recording.rigs.iter().enumerate() {
            if rig.role == role {
                self.recording.tracks[index].push(at, pose);
            }
        }
        // Held aside for rigs that appear later in the walk.
        self.pending.entry(role).or_default().push(at, pose);
    }

    fn rig(&mut self, rig: Rig) -> usize {
        if let Some(index) = self.rig_index.get(&rig) {
            return *index;
        }

        let index = self.recording.rigs.len();
        self.recording.rigs.push(rig);
        // A rig discovered mid-walk still needs the poses from before it was
        // discovered: its keypoint may simply have been out of frame.
        self.recording
            .tracks
            .push(self.pending.get(&rig.role).cloned().unwrap_or_default());
        self.rig_index.insert(rig, index);
        index
    }

    fn consider(
        &mut self,
        camera: usize,
        frame: &crate::pipeline::PoseFrame,
        config: &RecorderConfig,
    ) {
        if self.last_seq[camera] == Some(frame.seq) {
            return;
        }
        self.last_seq[camera] = Some(frame.seq);

        let trail = &mut self.recording.cameras[camera];
        trail.width = frame.width;
        trail.height = frame.height;
        if trail.samples.len() >= config.max_per_camera {
            return;
        }

        let roles: &[Role] = if config.use_controllers {
            &[Role::Head, Role::LeftHand, Role::RightHand]
        } else {
            &[Role::Head]
        };

        for role in roles {
            let Some((joint, keypoint)) = Rig::candidates(*role)
                .iter()
                .find_map(|joint| frame.keypoints.get(*joint).map(|kp| (*joint, kp)))
            else {
                continue;
            };

            if keypoint.confidence < config.min_confidence {
                self.recording.cameras[camera].rejected_confidence += 1;
                continue;
            }

            let rig = self.rig(Rig { role: *role, joint });
            if self.recording.tracks[rig].at(frame.captured_at).is_none() {
                self.recording.cameras[camera].rejected_no_pose += 1;
                continue;
            }

            let pixel = Point2::new(keypoint.x as f64, keypoint.y as f64);
            // A pose model will happily place a keypoint outside the frame
            // when the person is half out of shot. That is the model guessing,
            // not the camera seeing, and it has no business in a calibration.
            if normalized(pixel, frame.width, frame.height).is_none() {
                self.recording.cameras[camera].rejected_offscreen += 1;
                continue;
            }
            if let Some(previous) = self.last_pixel.get(&(camera, rig))
                && (pixel - previous).norm() < config.min_pixel_step as f64
            {
                self.recording.cameras[camera].rejected_stationary += 1;
                continue;
            }
            self.last_pixel.insert((camera, rig), pixel);

            self.recording.cameras[camera].record(Sample {
                at: frame.captured_at,
                rig,
                pixel,
                confidence: keypoint.confidence as f64,
            });
        }
    }

    fn publish(&mut self, channel: &Arc<RecordChannel>, started: Instant) {
        self.recording.duration = started.elapsed();

        let stats = RecorderStats {
            recording: true,
            elapsed: self.recording.duration,
            samples: self.recording.samples(),
            cameras: self
                .recording
                .cameras
                .iter()
                .map(|trail| {
                    (
                        trail.camera.clone(),
                        trail.samples.len(),
                        trail.coverage.filled(),
                    )
                })
                .collect(),
            rigs: self.recording.observability(),
            warning: self.warning(),
        };

        *channel.stats.lock() = stats;
        *channel.recording.lock() = self.recording.clone();
    }

    /// The one thing worth interrupting the user for while they are still
    /// walking.
    fn warning(&self) -> Option<String> {
        let head = self
            .recording
            .rigs
            .iter()
            .position(|rig| rig.role == Role::Head)?;

        // Only worth saying once there is enough of a walk to judge.
        if self.recording.tracks[head].span() < Duration::from_secs(10) {
            return None;
        }

        (self.recording.tracks[head].rotation_spread() < 0.15).then(|| {
            "turn your head more while walking: the offset between the headset \
             and the head keypoint cannot be separated from the camera \
             positions otherwise"
                .to_owned()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::traits::{Keypoint, Keypoints2d};
    use crate::pipeline::PoseFrame;
    use nalgebra::{Isometry3, Translation3, UnitQuaternion};

    const WIDTH: u32 = 1280;
    const HEIGHT: u32 = 720;

    fn state(cameras: usize) -> State {
        let channels: Vec<(String, Arc<PoseChannel>)> = (0..cameras)
            .map(|index| (format!("cam{index}"), Arc::new(PoseChannel::default())))
            .collect();
        State::new(&RecorderConfig::default(), &channels)
    }

    fn pose(x: f64) -> Isometry3<f64> {
        Isometry3::from_parts(
            Translation3::new(x, 1.5, 0.0),
            UnitQuaternion::from_euler_angles(0.0, x, 0.0),
        )
    }

    fn frame(seq: u64, at: Instant, joints: &[(Joint, f32, f32, f32)]) -> PoseFrame {
        let mut keypoints = Keypoints2d::default();
        for (joint, x, y, confidence) in joints {
            keypoints.set(
                *joint,
                Keypoint {
                    x: *x,
                    y: *y,
                    confidence: *confidence,
                },
            );
        }

        PoseFrame {
            seq,
            captured_at: at,
            width: WIDTH,
            height: HEIGHT,
            detection: None,
            keypoints,
        }
    }

    /// Poses either side of an instant, so a frame at that instant has
    /// something to interpolate between.
    fn surround(state: &mut State, role: Role, at: Instant) {
        state.push_pose(role, at - Duration::from_millis(10), pose(0.0));
        state.push_pose(role, at + Duration::from_millis(10), pose(0.4));
    }

    #[test]
    fn a_head_keypoint_becomes_a_head_rig() {
        let mut state = state(1);
        let at = Instant::now();
        surround(&mut state, Role::Head, at);

        state.consider(
            0,
            &frame(1, at, &[(Joint::Head, 640.0, 360.0, 0.9)]),
            &RecorderConfig::default(),
        );

        assert_eq!(
            state.recording.rigs,
            vec![Rig {
                role: Role::Head,
                joint: Joint::Head
            }]
        );
        assert_eq!(state.recording.cameras[0].samples.len(), 1);
    }

    /// A model without a head keypoint reports the nose instead, and the two
    /// are several centimetres apart. They must not share an offset.
    #[test]
    fn a_nose_and_a_head_are_different_rigs() {
        let config = RecorderConfig::default();
        let mut state = state(2);
        let at = Instant::now();
        surround(&mut state, Role::Head, at);

        state.consider(
            0,
            &frame(1, at, &[(Joint::Head, 100.0, 100.0, 0.9)]),
            &config,
        );
        state.consider(
            1,
            &frame(1, at, &[(Joint::Nose, 500.0, 300.0, 0.9)]),
            &config,
        );

        assert_eq!(state.recording.rigs.len(), 2);
        assert_ne!(
            state.recording.cameras[0].samples[0].rig,
            state.recording.cameras[1].samples[0].rig
        );
    }

    /// Both are present on a Halpe model. Recording both would give the head
    /// two offsets fitted to the same motion, so only the better one is taken.
    #[test]
    fn the_head_wins_when_both_are_present() {
        let mut state = state(1);
        let at = Instant::now();
        surround(&mut state, Role::Head, at);

        state.consider(
            0,
            &frame(
                1,
                at,
                &[
                    (Joint::Head, 100.0, 100.0, 0.9),
                    (Joint::Nose, 110.0, 120.0, 0.9),
                ],
            ),
            &RecorderConfig::default(),
        );

        assert_eq!(state.recording.rigs.len(), 1);
        assert_eq!(state.recording.rigs[0].joint, Joint::Head);
    }

    /// The pose channel holds one frame at a time and is polled faster than the
    /// cameras run, so the same frame is seen repeatedly.
    #[test]
    fn the_same_frame_is_not_recorded_twice() {
        let config = RecorderConfig::default();
        let mut state = state(1);
        let at = Instant::now();
        surround(&mut state, Role::Head, at);

        let frame = frame(7, at, &[(Joint::Head, 640.0, 360.0, 0.9)]);
        state.consider(0, &frame, &config);
        state.consider(0, &frame, &config);
        state.consider(0, &frame, &config);

        assert_eq!(state.recording.cameras[0].samples.len(), 1);
    }

    /// A user standing still contributes hundreds of identical rows, which
    /// weight the solve toward one spot and constrain nothing.
    #[test]
    fn standing_still_stops_adding_samples() {
        let config = RecorderConfig::default();
        let mut state = state(1);
        let at = Instant::now();
        surround(&mut state, Role::Head, at);

        for seq in 0..20 {
            let drift = seq as f32 * 0.2;
            state.consider(
                0,
                &frame(seq, at, &[(Joint::Head, 640.0 + drift, 360.0, 0.9)]),
                &config,
            );
        }

        assert_eq!(
            state.recording.cameras[0].samples.len(),
            1,
            "four pixels of drift over twenty frames is standing still"
        );
        assert_eq!(state.recording.cameras[0].rejected_stationary, 19);
    }

    #[test]
    fn walking_keeps_adding_samples() {
        let config = RecorderConfig::default();
        let mut state = state(1);
        let at = Instant::now();
        surround(&mut state, Role::Head, at);

        for seq in 0..20 {
            let moved = seq as f32 * 20.0;
            state.consider(
                0,
                &frame(seq, at, &[(Joint::Head, 100.0 + moved, 360.0, 0.9)]),
                &config,
            );
        }

        assert_eq!(state.recording.cameras[0].samples.len(), 20);
        assert_eq!(state.recording.cameras[0].rejected_stationary, 0);
    }

    #[test]
    fn a_weak_keypoint_is_not_recorded() {
        let config = RecorderConfig::default();
        let mut state = state(1);
        let at = Instant::now();
        surround(&mut state, Role::Head, at);

        state.consider(
            0,
            &frame(1, at, &[(Joint::Head, 640.0, 360.0, 0.2)]),
            &config,
        );

        assert!(state.recording.cameras[0].samples.is_empty());
        assert_eq!(state.recording.cameras[0].rejected_confidence, 1);
    }

    /// A frame taken while the headset was not tracking, or before the poses
    /// started arriving, has nothing to pair with. Recording it against the
    /// nearest pose available would be inventing data.
    #[test]
    fn a_frame_with_no_pose_to_pair_with_is_dropped() {
        let config = RecorderConfig::default();
        let mut state = state(1);
        let at = Instant::now();
        surround(&mut state, Role::Head, at);

        // An hour later, far outside the track.
        let orphan = at + Duration::from_secs(3600);
        state.consider(
            0,
            &frame(1, orphan, &[(Joint::Head, 640.0, 360.0, 0.9)]),
            &config,
        );

        assert!(state.recording.cameras[0].samples.is_empty());
        assert_eq!(state.recording.cameras[0].rejected_no_pose, 1);
    }

    /// A hand that only comes into frame halfway through the walk still needs
    /// the poses from before it appeared, because the frames either side of its
    /// first sighting have to interpolate across them.
    #[test]
    fn a_rig_discovered_late_still_gets_the_earlier_poses() {
        let config = RecorderConfig::default();
        let mut state = state(1);
        let start = Instant::now();

        for step in 0..20 {
            state.push_pose(
                Role::LeftHand,
                start + Duration::from_millis(step * 10),
                pose(step as f64 * 0.05),
            );
        }

        let late = start + Duration::from_millis(150);
        state.consider(
            0,
            &frame(1, late, &[(Joint::LeftWrist, 400.0, 200.0, 0.9)]),
            &config,
        );

        assert_eq!(state.recording.cameras[0].samples.len(), 1);
        let track = &state.recording.tracks[0];
        assert_eq!(
            track.len(),
            20,
            "the whole track should have been carried over"
        );
        assert!(track.at(start + Duration::from_millis(45)).is_some());
    }

    #[test]
    fn controllers_can_be_left_out() {
        let config = RecorderConfig {
            use_controllers: false,
            ..RecorderConfig::default()
        };
        let mut state = state(1);
        let at = Instant::now();
        surround(&mut state, Role::Head, at);
        surround(&mut state, Role::LeftHand, at);

        state.consider(
            0,
            &frame(
                1,
                at,
                &[
                    (Joint::Head, 640.0, 360.0, 0.9),
                    (Joint::LeftWrist, 400.0, 500.0, 0.9),
                ],
            ),
            &config,
        );

        assert_eq!(state.recording.rigs.len(), 1);
        assert_eq!(state.recording.rigs[0].role, Role::Head);
    }

    #[test]
    fn coverage_counts_where_in_the_frame_the_walk_went() {
        let mut coverage = Coverage::new();
        assert_eq!(coverage.filled(), 0.0);

        // Four samples in one cell, one in another.
        for _ in 0..4 {
            coverage.record(Point2::new(100.0, 100.0), WIDTH, HEIGHT);
        }
        coverage.record(Point2::new(1200.0, 700.0), WIDTH, HEIGHT);

        assert_eq!(coverage.count(0, 0), 4);
        assert_eq!(coverage.count(7, 5), 1);
        assert!(
            (coverage.filled() - 1.0 / 48.0).abs() < 1e-6,
            "one cell of forty-eight has enough samples, got {}",
            coverage.filled()
        );
    }

    /// Casting a small negative to an integer truncates toward zero, so a
    /// keypoint just off the left edge would land in the leftmost column if the
    /// bound were checked after the cast rather than before it.
    #[test]
    fn a_keypoint_outside_the_frame_does_not_land_in_a_cell() {
        let mut coverage = Coverage::new();
        coverage.record(Point2::new(-5.0, 100.0), WIDTH, HEIGHT);
        coverage.record(Point2::new(100.0, HEIGHT as f64 + 5.0), WIDTH, HEIGHT);

        assert_eq!(coverage.filled(), 0.0);
        assert_eq!(coverage.count(0, 0), 0);
    }

    /// A pose model asked about a person half out of shot will place the
    /// keypoint outside the frame. That is the model guessing rather than the
    /// camera seeing.
    #[test]
    fn an_offscreen_keypoint_is_not_recorded() {
        let config = RecorderConfig::default();
        let mut state = state(1);
        let at = Instant::now();
        surround(&mut state, Role::Head, at);

        state.consider(
            0,
            &frame(1, at, &[(Joint::Head, -12.0, 360.0, 0.9)]),
            &config,
        );

        assert!(state.recording.cameras[0].samples.is_empty());
        assert_eq!(state.recording.cameras[0].rejected_offscreen, 1);
    }

    /// The one thing worth interrupting a walk for: without head rotation the
    /// offset cannot be told apart from a shift of every camera.
    #[test]
    fn a_walk_without_head_rotation_is_flagged() {
        let mut state = state(1);
        let start = Instant::now();

        for step in 0..1200 {
            let at = start + Duration::from_millis(step * 10);
            state.push_pose(
                Role::Head,
                at,
                Isometry3::from_parts(
                    Translation3::new(step as f64 * 0.001, 1.5, 0.0),
                    UnitQuaternion::identity(),
                ),
            );
        }
        state.consider(
            0,
            &frame(
                1,
                start + Duration::from_millis(100),
                &[(Joint::Head, 640.0, 360.0, 0.9)],
            ),
            &RecorderConfig::default(),
        );

        assert!(
            state
                .warning()
                .is_some_and(|w| w.contains("turn your head")),
            "a walk with a fixed head orientation should be flagged"
        );
    }

    #[test]
    fn a_walk_that_turns_the_head_is_not_flagged() {
        let mut state = state(1);
        let start = Instant::now();

        for step in 0..1200 {
            let at = start + Duration::from_millis(step * 10);
            state.push_pose(Role::Head, at, pose(step as f64 * 0.005));
        }
        state.consider(
            0,
            &frame(
                1,
                start + Duration::from_millis(100),
                &[(Joint::Head, 640.0, 360.0, 0.9)],
            ),
            &RecorderConfig::default(),
        );

        assert!(state.warning().is_none());
    }
}
