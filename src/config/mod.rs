//! Application-wide settings, persisted to `config.toml`.
//!
//! Room-specific data (camera intrinsics, extrinsics, calibration quality) does
//! not live here; it belongs to a room profile so that a user can keep several
//! rooms without their UI preferences following along.
//!
//! Every field carries `#[serde(default)]` so that a config written by an older
//! build still loads.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

pub mod camera;

pub use camera::{CameraConfig, ControlName, ControlSetting, LensKind, Rotation, SourceConfig};

use crate::app::panels::Panel;
use crate::infer::ProviderChoice;
use crate::paths;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub window: WindowConfig,
    pub ui: UiConfig,
    pub capture: CaptureConfig,
    pub inference: InferenceConfig,
    pub vr: VrConfig,
    pub fusion: FusionConfig,
    pub output: OutputConfig,
    /// Room profile to load at startup, by name. The calibration a room needs
    /// belongs to the room rather than to the application, so only its name
    /// lives here.
    pub room: Option<String>,
    pub cameras: Vec<CameraConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CaptureConfig {
    /// Open the configured cameras as soon as the application starts. A user
    /// who has already set their room up expects it to just run.
    pub auto_start: bool,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self { auto_start: true }
    }
}

impl Config {
    /// Generates an id that no existing camera uses.
    pub fn fresh_camera_id(&self) -> String {
        (0..)
            .map(|n| format!("cam{n}"))
            .find(|id| !self.cameras.iter().any(|c| &c.id == id))
            .expect("an unused camera id always exists")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WindowConfig {
    pub size: [f32; 2],
    pub pos: Option<[f32; 2]>,
    pub maximized: bool,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            size: [1280.0, 800.0],
            pos: None,
            maximized: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct UiConfig {
    pub panel: Panel,
    /// Minimum level shown in the log panel.
    pub log_level: LogLevel,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            panel: Panel::Cameras,
            log_level: LogLevel::Info,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl LogLevel {
    pub const ALL: [LogLevel; 5] = [
        LogLevel::Error,
        LogLevel::Warn,
        LogLevel::Info,
        LogLevel::Debug,
        LogLevel::Trace,
    ];

    pub fn label(self) -> &'static str {
        match self {
            LogLevel::Error => "Error",
            LogLevel::Warn => "Warn",
            LogLevel::Info => "Info",
            LogLevel::Debug => "Debug",
            LogLevel::Trace => "Trace",
        }
    }

    fn rank(self) -> u8 {
        match self {
            LogLevel::Error => 0,
            LogLevel::Warn => 1,
            LogLevel::Info => 2,
            LogLevel::Debug => 3,
            LogLevel::Trace => 4,
        }
    }

    /// True if a record at `level` should be shown at this filter setting.
    pub fn includes(self, level: tracing::Level) -> bool {
        let record_rank = match level {
            tracing::Level::ERROR => 0,
            tracing::Level::WARN => 1,
            tracing::Level::INFO => 2,
            tracing::Level::DEBUG => 3,
            tracing::Level::TRACE => 4,
        };
        record_rank <= self.rank()
    }
}

impl Config {
    /// Loads the config, falling back to defaults if it is missing or broken.
    ///
    /// A corrupt config must not stop the app from starting: the user would
    /// have no way to fix it from the UI.
    pub fn load_or_default() -> Self {
        match Self::load() {
            Ok(Some(config)) => config,
            Ok(None) => Self::default(),
            Err(err) => {
                tracing::warn!("failed to load the config, using defaults: {err:#}");
                Self::default()
            }
        }
    }

    fn load() -> Result<Option<Self>> {
        let path = paths::config_file()?;
        if !path.exists() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let config =
            toml::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))?;
        Ok(Some(config))
    }

    /// Writes the config through a temporary file so an interrupted save
    /// cannot leave a truncated file behind.
    pub fn save(&self) -> Result<()> {
        let path = paths::config_file()?;
        let text = toml::to_string_pretty(self).context("failed to serialize the config")?;

        let tmp = path.with_extension("toml.tmp");
        std::fs::write(&tmp, text).with_context(|| format!("failed to write {}", tmp.display()))?;
        std::fs::rename(&tmp, &path)
            .with_context(|| format!("failed to replace {}", path.display()))?;
        Ok(())
    }
}

/// Settings for the inference stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct InferenceConfig {
    pub enabled: bool,
    /// Execution provider requested for every model.
    pub provider: ProviderChoice,
    /// Detector used by cameras that do not name their own.
    pub detector_model: String,
    /// Pose model used by cameras that do not name their own.
    pub pose_model: String,
    /// Frames to skip between detector runs. The subject is one slowly moving
    /// person, so detecting every frame buys little and costs a lot.
    pub detect_every: u32,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            provider: ProviderChoice::default(),
            detector_model: "yolox-tiny-humanart-416".to_owned(),
            pose_model: "rtmpose-m-halpe26-256x192".to_owned(),
            detect_every: 5,
        }
    }
}

/// Settings for the fusion stage.
///
/// Only the knobs a user has reason to touch. Everything else the stage needs —
/// outlier thresholds, constraint passes, process noise — is decided by the
/// code that has to answer for it, and putting it here would only make it
/// possible to break tracking from a text file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FusionConfig {
    pub enabled: bool,
    /// Rate of the fusion clock, in hertz.
    pub rate_hz: u32,
    /// Margin the fusion clock keeps on top of what the cameras are measured
    /// to be delivering, in milliseconds.
    ///
    /// Interpolating a camera onto an instant needs a frame after it, so the
    /// clock sits back far enough that one has arrived. How far that is is
    /// measured rather than configured — it is whatever the latest camera is
    /// actually managing — and this is the headroom on top, absorbing the
    /// ordinary tick-to-tick variation in when frames land.
    pub align_slack_ms: u32,
    /// How far behind real time the fusion clock will ever sit, in
    /// milliseconds.
    ///
    /// The clock follows whichever camera delivers latest, because a camera it
    /// does not wait for is a camera that drops in and out of ticks — and a
    /// joint reconstructed from a different set of cameras every few ticks
    /// moves by the disagreement between them each time the set changes, which
    /// is the calibration error and is nothing like noise that smoothing can
    /// remove.
    ///
    /// This is where waiting stops being worth it. A camera later than this is
    /// left out of the reconstruction entirely and said so in the Tracking
    /// panel, which is at least a decision the user can act on.
    pub max_lag_ms: u32,
    /// Keypoint confidence below which a ray is not used.
    pub min_confidence: f32,
    /// Positional uncertainty past which a joint is withheld, in metres.
    pub max_joint_sigma: f32,
    /// How far ahead to predict, in milliseconds.
    ///
    /// Only the part of the delay Optra cannot measure: the OSC hop and
    /// whatever the consumer does before it draws. The larger part — the time
    /// between the light landing on a sensor and a reconstruction existing —
    /// is measured by the fusion stage and added to this automatically, so
    /// there is nothing here to keep in step with the camera setup.
    pub prediction_ms: u32,
    /// Cutoff of the position smoothing at rest, in hertz. Lower is stiller and
    /// slower to respond.
    pub smoothing_hz: f32,
    /// Keep refining the body measurement while tracking runs.
    pub measure_body: bool,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rate_hz: 60,
            align_slack_ms: 40,
            max_lag_ms: 220,
            min_confidence: 0.3,
            max_joint_sigma: 0.10,
            prediction_ms: 20,
            smoothing_hz: 1.2,
            measure_body: true,
        }
    }
}

impl FusionConfig {
    pub fn fuse_options(&self) -> crate::fusion::fuse::FuseOptions {
        crate::fusion::fuse::FuseOptions {
            min_confidence: self.min_confidence as f64,
            max_sigma: self.max_joint_sigma as f64,
            ..crate::fusion::fuse::FuseOptions::default()
        }
    }

    pub fn fit_options(
        &self,
        measure: crate::fusion::bones::MeasureOptions,
    ) -> crate::fusion::fit::FitOptions {
        crate::fusion::fit::FitOptions {
            measure,
            ..crate::fusion::fit::FitOptions::default()
        }
    }

    pub fn filter_options(&self) -> crate::fusion::filter::FilterOptions {
        crate::fusion::filter::FilterOptions {
            min_cutoff: self.smoothing_hz as f64,
            horizon: std::time::Duration::from_millis(self.prediction_ms as u64),
            ..crate::fusion::filter::FilterOptions::default()
        }
    }
}

/// How Optra talks to SteamVR.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VrConfig {
    /// Look for SteamVR at all. Turning it off is for a machine being used
    /// only to set cameras up.
    pub enabled: bool,
    /// Pose sampling rate. Higher than any camera runs at, because the poses
    /// are interpolated to the instant each frame was taken and interpolation
    /// over a shorter gap is a better guess.
    pub poll_hz: u32,
    /// How much pose history to keep. Long enough that a frame still waiting
    /// on inference can find the poses it needs.
    pub history_seconds: f32,
    /// Wait between attempts to reach SteamVR.
    pub retry_seconds: f32,
    /// How long the headset may stop reporting before the connection is
    /// dropped and reopened.
    pub patience_seconds: f32,
}

impl Default for VrConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            poll_hz: 120,
            history_seconds: 4.0,
            retry_seconds: 5.0,
            patience_seconds: 5.0,
        }
    }
}

/// Where the trackers go.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SinkKind {
    /// VRChat's own OSC tracker input.
    VrchatOsc,
    /// VirtualMotionTracker's virtual SteamVR devices.
    Vmt,
}

impl SinkKind {
    pub const ALL: [SinkKind; 2] = [SinkKind::VrchatOsc, SinkKind::Vmt];

    pub fn label(self) -> &'static str {
        match self {
            SinkKind::VrchatOsc => "VRChat OSC",
            SinkKind::Vmt => "SteamVR via VMT",
        }
    }

    /// What choosing it means, for a user who has not met either.
    pub fn description(self) -> &'static str {
        match self {
            SinkKind::VrchatOsc => {
                "Straight into VRChat, no driver to install. Only VRChat sees the trackers."
            }
            SinkKind::Vmt => {
                "Real SteamVR devices, so anything that reads SteamVR sees them. \
                 Needs VirtualMotionTracker installed."
            }
        }
    }
}

/// One tracker's settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackerConfig {
    pub role: crate::output::TrackerRole,
    pub enabled: bool,
    /// Offset from the joint to where the tracker should sit, in the tracker's
    /// own frame and in metres.
    ///
    /// A real puck is strapped to the outside of a limb, not to the bone: the
    /// avatar's proportions are calibrated against wherever the tracker was,
    /// so being consistently a few centimetres off is harmless and being
    /// somewhere different every session is not.
    pub offset: [f32; 3],
}

/// Settings for the output stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputConfig {
    pub enabled: bool,
    pub sink: SinkKind,
    /// Sends per second.
    ///
    /// Higher than the fusion rate on purpose. Each send predicts to a later
    /// instant from the same reconstruction, so the poses really do advance
    /// between them, and the consumer's own render loop gets an answer closer
    /// to the moment it asked.
    pub rate_hz: u32,
    pub vrchat_target: String,
    pub vmt_target: String,
    /// Tell VMT what SteamVR's room setup is, for this run only.
    ///
    /// VMT places devices in the runtime's raw space and keeps its own idea of
    /// how that relates to the room. Optra can read the true one from OpenVR,
    /// which saves configuring the same thing twice — but a user who has set
    /// it themselves may want theirs left alone.
    pub vmt_send_room_matrix: bool,
    /// Furthest ahead of the reconstruction the output may extrapolate, in
    /// milliseconds.
    ///
    /// A cap on *time*, unlike the distance limit inside the filter. What it
    /// bounds is how much of what goes out is a guess: the fusion lag is
    /// measured rather than configured, and if the cameras are slow it can be
    /// most of the total on its own. Past a point a lagging body beats a
    /// guessed one.
    ///
    /// It is also how a user finds out whether prediction is what is wrong with
    /// their tracking. Set to zero, nothing is extrapolated at all and the
    /// trackers show where the cameras last saw the body — which is late, but
    /// if it is also *still*, the answer is here rather than upstream.
    pub max_lead_ms: u32,
    /// Positional uncertainty past which a tracker is not sent, in metres.
    pub max_sigma: f32,
    pub trackers: Vec<TrackerConfig>,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            sink: SinkKind::VrchatOsc,
            rate_hz: 90,
            vrchat_target: crate::output::vrchat::DEFAULT_TARGET.to_owned(),
            vmt_target: crate::output::vmt::DEFAULT_TARGET.to_owned(),
            vmt_send_room_matrix: true,
            max_lead_ms: 150,
            max_sigma: 0.08,
            trackers: crate::output::TrackerRole::ALL
                .iter()
                .map(|role| TrackerConfig {
                    role: *role,
                    // Hips and both feet: the three that make full-body
                    // tracking work, and the three a camera looking at a
                    // standing person can actually see. The rest are there to
                    // be turned on by someone who has checked that their room
                    // supports them.
                    enabled: role.is_essential(),
                    offset: [0.0; 3],
                })
                .collect(),
        }
    }
}

impl OutputConfig {
    /// Fills in any role missing from a config written by an older build, so a
    /// new tracker does not simply fail to appear in the panel.
    pub fn complete(&mut self) {
        for role in crate::output::TrackerRole::ALL {
            if !self.trackers.iter().any(|tracker| tracker.role == role) {
                self.trackers.push(TrackerConfig {
                    role,
                    enabled: false,
                    offset: [0.0; 3],
                });
            }
        }
    }

    pub fn enabled_roles(&self) -> Vec<crate::output::TrackerRole> {
        self.trackers
            .iter()
            .filter(|tracker| tracker.enabled)
            .map(|tracker| tracker.role)
            .collect()
    }

    /// Per-role offsets, in metres, for the roles that have one.
    pub fn offsets(&self) -> Vec<(crate::output::TrackerRole, nalgebra::Vector3<f64>)> {
        self.trackers
            .iter()
            .filter(|tracker| tracker.enabled && tracker.offset != [0.0; 3])
            .map(|tracker| {
                (
                    tracker.role,
                    nalgebra::Vector3::new(
                        tracker.offset[0] as f64,
                        tracker.offset[1] as f64,
                        tracker.offset[2] as f64,
                    ),
                )
            })
            .collect()
    }

    pub fn target(&self) -> &str {
        match self.sink {
            SinkKind::VrchatOsc => &self.vrchat_target,
            SinkKind::Vmt => &self.vmt_target,
        }
    }

    pub fn target_mut(&mut self) -> &mut String {
        match self.sink {
            SinkKind::VrchatOsc => &mut self.vrchat_target,
            SinkKind::Vmt => &mut self.vmt_target,
        }
    }
}
