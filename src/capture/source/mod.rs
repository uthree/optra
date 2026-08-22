//! Frame sources.
//!
//! A source is whatever produces images for one camera. Real capture devices
//! are one implementation; a generated scene is another, which is what makes it
//! possible to develop and test the multi-camera paths without owning the
//! hardware yet.

#[cfg(windows)]
pub mod controls;
pub mod still;
pub mod synthetic;
#[cfg(windows)]
pub mod webcam;

use std::fmt;

use anyhow::Result;

use crate::config::{CameraConfig, ControlName, Rotation, SourceConfig};

/// One frame as handed over by a source, before rotation is applied.
pub struct RawFrame {
    pub width: u32,
    pub height: u32,
    /// Tightly packed RGB8.
    pub rgb: Vec<u8>,
    /// Time spent turning the device's output into RGB, excluding the wait for
    /// the frame itself.
    pub decode: std::time::Duration,
}

/// What the device actually gave us, which is not necessarily what was asked
/// for. Showing the difference is the point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NegotiatedFormat {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub pixel_format: String,
}

impl fmt::Display for NegotiatedFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}x{} @ {} fps ({})",
            self.width, self.height, self.fps, self.pixel_format
        )
    }
}

pub trait FrameSource {
    /// Blocks until the next frame is available.
    fn next_frame(&mut self) -> Result<RawFrame>;

    /// The format the source settled on.
    fn negotiated(&self) -> NegotiatedFormat;
}

/// What a device reports about one of its properties.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlInfo {
    pub name: ControlName,
    pub min: i64,
    pub max: i64,
    pub step: i64,
    pub default: i64,
    pub value: i64,
    /// True when the device is currently deciding this value by itself.
    pub auto: bool,
    pub auto_supported: bool,
    pub manual_supported: bool,
}

/// Read and write access to a device's properties.
///
/// This is separate from [`FrameSource`] because on Windows it is a separate
/// handle on the device: properties can be changed while another handle is
/// streaming, which is what makes live exposure adjustment possible.
pub trait ControlSession {
    fn list(&self) -> Vec<ControlInfo>;
    fn get(&self, name: ControlName) -> Option<ControlInfo>;
    fn set(&self, name: ControlName, value: i64, auto: bool) -> Result<()>;
}

/// Opens a property session for a camera, if the platform and the device
/// support one. A synthetic source has no properties to control.
pub fn open_controls(config: &CameraConfig) -> Option<Box<dyn ControlSession>> {
    match &config.source {
        SourceConfig::Synthetic { .. } | SourceConfig::Still { .. } => None,
        #[cfg(windows)]
        SourceConfig::Webcam { device_path, .. } => {
            match controls::DeviceControls::open(device_path) {
                Ok(session) => Some(Box::new(session) as Box<dyn ControlSession>),
                Err(err) => {
                    tracing::warn!(
                        camera = %config.id,
                        "device properties are unavailable: {err:#}"
                    );
                    None
                }
            }
        }
        #[cfg(not(windows))]
        SourceConfig::Webcam { .. } => None,
    }
}

/// Opens the source described by `config`.
pub fn open(config: &CameraConfig) -> Result<Box<dyn FrameSource>> {
    match &config.source {
        SourceConfig::Synthetic { seat } => {
            Ok(Box::new(synthetic::SyntheticSource::new(config, *seat)))
        }
        SourceConfig::Still { path } => Ok(Box::new(still::StillSource::open(
            config,
            std::path::Path::new(path),
        )?)),
        #[cfg(windows)]
        SourceConfig::Webcam {
            device_path,
            device_name,
        } => Ok(Box::new(webcam::WebcamSource::open(
            config,
            device_path,
            device_name,
        )?)),
        #[cfg(not(windows))]
        SourceConfig::Webcam { .. } => {
            anyhow::bail!("webcam capture is only implemented for Windows")
        }
    }
}

/// Rotates a frame by whole quarter turns.
///
/// A camera mounted sideways in a ceiling corner is common enough that fixing
/// it here is cheaper than teaching every later stage about it.
pub fn rotate(frame: RawFrame, rotation: Rotation) -> RawFrame {
    if rotation == Rotation::None {
        return frame;
    }

    let (w, h) = (frame.width as usize, frame.height as usize);
    let (out_w, out_h) = if rotation.transposes() {
        (h, w)
    } else {
        (w, h)
    };
    let mut out = vec![0u8; out_w * out_h * 3];

    for y in 0..h {
        for x in 0..w {
            let (nx, ny) = match rotation {
                Rotation::None => unreachable!(),
                Rotation::Cw90 => (h - 1 - y, x),
                Rotation::Cw180 => (w - 1 - x, h - 1 - y),
                Rotation::Cw270 => (y, w - 1 - x),
            };
            let src = (y * w + x) * 3;
            let dst = (ny * out_w + nx) * 3;
            out[dst..dst + 3].copy_from_slice(&frame.rgb[src..src + 3]);
        }
    }

    RawFrame {
        width: out_w as u32,
        height: out_h as u32,
        rgb: out,
        decode: frame.decode,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2x1 image: red then green.
    fn sample() -> RawFrame {
        RawFrame {
            width: 2,
            height: 1,
            rgb: vec![255, 0, 0, 0, 255, 0],
            decode: std::time::Duration::ZERO,
        }
    }

    #[test]
    fn quarter_turn_transposes_and_moves_the_first_pixel_to_the_top_right() {
        let rotated = rotate(sample(), Rotation::Cw90);
        assert_eq!((rotated.width, rotated.height), (1, 2));
        assert_eq!(rotated.rgb, vec![255, 0, 0, 0, 255, 0]);
    }

    #[test]
    fn half_turn_reverses_the_row() {
        let rotated = rotate(sample(), Rotation::Cw180);
        assert_eq!((rotated.width, rotated.height), (2, 1));
        assert_eq!(rotated.rgb, vec![0, 255, 0, 255, 0, 0]);
    }

    #[test]
    fn four_quarter_turns_are_the_identity() {
        let mut frame = sample();
        for _ in 0..4 {
            frame = rotate(frame, Rotation::Cw90);
        }
        assert_eq!((frame.width, frame.height), (2, 1));
        assert_eq!(frame.rgb, sample().rgb);
    }

    #[test]
    fn no_rotation_is_a_passthrough() {
        let frame = rotate(sample(), Rotation::None);
        assert_eq!(frame.rgb, sample().rgb);
    }
}
