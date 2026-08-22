//! Camera configuration.
//!
//! A camera is identified by its device path rather than its enumeration index,
//! so that replugging USB devices cannot silently swap two cameras and
//! invalidate a calibration.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CameraConfig {
    /// Stable identifier used in room profiles and logs.
    pub id: String,
    /// Name shown in the UI.
    pub label: String,
    pub enabled: bool,
    pub source: SourceConfig,
    /// Requested capture format. The device may negotiate something close.
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub lens: LensKind,
    /// Rotation applied to the captured image, in 90 degree steps.
    pub rotation: Rotation,
    /// Device properties to apply when the camera opens.
    pub controls: Vec<ControlSetting>,
    /// Detector for this camera, or the shared default when absent.
    pub detector_model: Option<String>,
    /// Pose model for this camera, or the shared default when absent.
    pub pose_model: Option<String>,
}

impl CameraConfig {
    pub fn control(&self, name: ControlName) -> Option<&ControlSetting> {
        self.controls.iter().find(|setting| setting.name == name)
    }

    pub fn set_control(&mut self, name: ControlName, auto: bool, value: i64) {
        match self.controls.iter_mut().find(|s| s.name == name) {
            Some(setting) => {
                setting.auto = auto;
                setting.value = value;
            }
            None => self.controls.push(ControlSetting { name, auto, value }),
        }
    }

    pub fn clear_control(&mut self, name: ControlName) {
        self.controls.retain(|setting| setting.name != name);
    }
}

/// A device property Optra applies when the camera opens.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlSetting {
    pub name: ControlName,
    /// Let the device regulate the value itself. `value` is then ignored.
    pub auto: bool,
    pub value: i64,
}

/// A device property Optra can read and write.
///
/// This vocabulary lives in the config rather than in the platform layer so
/// that a saved profile does not depend on how a given OS names its properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlName {
    Exposure,
    Gain,
    Brightness,
    Contrast,
    Saturation,
    Sharpness,
    WhiteBalance,
    BacklightCompensation,
    Focus,
    Zoom,
    Pan,
    Tilt,
}

impl ControlName {
    /// Ordered so that the properties which decide whether tracking works at
    /// all come first.
    pub const ALL: [ControlName; 12] = [
        ControlName::Exposure,
        ControlName::Gain,
        ControlName::Focus,
        ControlName::WhiteBalance,
        ControlName::Brightness,
        ControlName::Contrast,
        ControlName::Saturation,
        ControlName::Sharpness,
        ControlName::BacklightCompensation,
        ControlName::Zoom,
        ControlName::Pan,
        ControlName::Tilt,
    ];

    pub fn label(self) -> &'static str {
        match self {
            ControlName::Exposure => "Exposure",
            ControlName::Gain => "Gain",
            ControlName::Brightness => "Brightness",
            ControlName::Contrast => "Contrast",
            ControlName::Saturation => "Saturation",
            ControlName::Sharpness => "Sharpness",
            ControlName::WhiteBalance => "White balance",
            ControlName::BacklightCompensation => "Backlight compensation",
            ControlName::Focus => "Focus",
            ControlName::Zoom => "Zoom",
            ControlName::Pan => "Pan",
            ControlName::Tilt => "Tilt",
        }
    }

    /// UVC reports exposure in log2 seconds, which is unreadable on a slider,
    /// so the shutter time is spelled out next to it.
    pub fn describe_value(self, value: i64) -> Option<String> {
        if self != ControlName::Exposure {
            return None;
        }
        let seconds = 2f64.powi(value as i32);
        Some(if seconds >= 1.0 {
            format!("{seconds:.0} s")
        } else {
            format!("1/{:.0} s", 1.0 / seconds)
        })
    }

    /// The longest exposure that still allows `fps` frames per second, as a
    /// log2-seconds value.
    pub fn exposure_for_fps(fps: u32) -> i64 {
        let period = 1.0 / fps.max(1) as f64;
        period.log2().floor() as i64
    }
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            id: "cam".to_owned(),
            label: "Camera".to_owned(),
            enabled: true,
            source: SourceConfig::default(),
            width: 1280,
            height: 720,
            fps: 30,
            lens: LensKind::Standard,
            rotation: Rotation::None,
            controls: Vec::new(),
            detector_model: None,
            pose_model: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum SourceConfig {
    /// A real capture device.
    Webcam {
        /// Media Foundation symbolic link. This is the stable identity.
        device_path: String,
        /// Last known human-readable name, for reporting a missing device.
        device_name: String,
    },
    /// A generated scene, used to exercise multi-camera paths without hardware.
    Synthetic {
        /// Which corner of the virtual room this camera sits in.
        seat: u32,
    },
    /// A still image replayed at the configured frame rate, for testing the
    /// stages downstream against a known scene.
    Still { path: String },
}

impl Default for SourceConfig {
    fn default() -> Self {
        SourceConfig::Synthetic { seat: 0 }
    }
}

impl SourceConfig {
    pub fn is_synthetic(&self) -> bool {
        matches!(self, SourceConfig::Synthetic { .. })
    }
}

/// Lens model used when calibrating this camera.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LensKind {
    /// Radial-tangential distortion, for ordinary lenses.
    Standard,
    /// Radial-tangential with a wider field of view.
    Wide,
    /// Equidistant projection, for lenses beyond roughly 120 degrees.
    Fisheye,
}

impl LensKind {
    pub const ALL: [LensKind; 3] = [LensKind::Standard, LensKind::Wide, LensKind::Fisheye];

    pub fn label(self) -> &'static str {
        match self {
            LensKind::Standard => "Standard",
            LensKind::Wide => "Wide",
            LensKind::Fisheye => "Fisheye",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Rotation {
    None,
    Cw90,
    Cw180,
    Cw270,
}

impl Rotation {
    pub const ALL: [Rotation; 4] = [
        Rotation::None,
        Rotation::Cw90,
        Rotation::Cw180,
        Rotation::Cw270,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Rotation::None => "0\u{b0}",
            Rotation::Cw90 => "90\u{b0}",
            Rotation::Cw180 => "180\u{b0}",
            Rotation::Cw270 => "270\u{b0}",
        }
    }

    /// True when the rotation swaps width and height.
    pub fn transposes(self) -> bool {
        matches!(self, Rotation::Cw90 | Rotation::Cw270)
    }
}
