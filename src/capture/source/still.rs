//! A still image replayed as a camera.
//!
//! Real cameras cannot be pointed at the same scene twice, which makes them a
//! poor basis for tests of everything downstream. A still source gives the
//! pipeline a known image at a known rate, so inference and, later, fusion can
//! be checked against an expected answer rather than against whatever happened
//! to be in front of the lens.

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use super::{FrameSource, NegotiatedFormat, RawFrame};
use crate::config::CameraConfig;

pub struct StillSource {
    width: u32,
    height: u32,
    rgb: Vec<u8>,
    fps: u32,
    next_frame_at: Instant,
}

impl StillSource {
    pub fn open(config: &CameraConfig, path: &Path) -> Result<Self> {
        let decoded = image::open(path)
            .with_context(|| format!("failed to read {}", path.display()))?
            .to_rgb8();
        let (width, height) = decoded.dimensions();

        Ok(Self {
            width,
            height,
            rgb: decoded.into_raw(),
            fps: config.fps.max(1),
            next_frame_at: Instant::now(),
        })
    }
}

impl FrameSource for StillSource {
    fn next_frame(&mut self) -> Result<RawFrame> {
        let now = Instant::now();
        if self.next_frame_at > now {
            std::thread::sleep(self.next_frame_at - now);
        }
        let period = Duration::from_secs_f32(1.0 / self.fps as f32);
        self.next_frame_at = (self.next_frame_at + period).max(Instant::now());

        Ok(RawFrame {
            width: self.width,
            height: self.height,
            rgb: self.rgb.clone(),
            decode: Duration::ZERO,
        })
    }

    fn negotiated(&self) -> NegotiatedFormat {
        NegotiatedFormat {
            width: self.width,
            height: self.height,
            fps: self.fps,
            pixel_format: "STILL".to_owned(),
        }
    }
}
