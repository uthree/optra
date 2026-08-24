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
