//! A generated scene, used to exercise the multi-camera pipeline without
//! multiple physical cameras.
//!
//! The scene comes from [`crate::sim`]: a figure walking a circle in a tiled
//! room, seen from one of the ceiling corners. Each seat gets a different field
//! of view and a different amount of barrel distortion, so a set of synthetic
//! cameras is unalike in the ways a set of real ones is.
//!
//! This used to draw a stick figure through a projection it worked out itself,
//! with its own vector maths and its own rasteriser. Two of those were copies
//! of [`crate::geometry::camera`] and the third could not produce anything a
//! pose model would recognise as a person. Going through the same camera model
//! as the rest of the application means a synthetic camera is now worth
//! pointing a model at, and means a bug in the projection cannot cancel itself
//! out by being present in both places.

use std::time::{Duration, Instant};

use anyhow::Result;

use super::{FrameSource, NegotiatedFormat, RawFrame};
use crate::config::CameraConfig;
use crate::geometry::camera::Camera;
use crate::sim::Scene;
use crate::sim::render::RenderOptions;

pub struct SyntheticSource {
    scene: Scene,
    camera: Camera,
    fps: u32,
    started: Instant,
    next_frame_at: Instant,
}

impl SyntheticSource {
    pub fn new(config: &CameraConfig, seat: u32) -> Self {
        let scene = Scene {
            render: RenderOptions {
                // No supersampling, unlike the accuracy harness. This one has a
                // frame rate to keep, and four times the fill is the difference
                // between a live preview and a slideshow.
                supersample: 1,
                ..RenderOptions::default()
            },
            ..Scene::default()
        };
        let camera = scene
            .room
            .camera_at(seat, config.width.max(64), config.height.max(64));
        let now = Instant::now();

        Self {
            scene,
            camera,
            fps: config.fps.max(1),
            started: now,
            next_frame_at: now,
        }
    }

    /// Where this camera is and how it sees, which is the answer a calibration
    /// run against synthetic cameras is trying to recover.
    pub fn camera(&self) -> &Camera {
        &self.camera
    }
}

impl FrameSource for SyntheticSource {
    fn next_frame(&mut self) -> Result<RawFrame> {
        // Pace the source at its nominal frame rate so that measured FPS and
        // the temporal behaviour of the pipeline mean something.
        let now = Instant::now();
        if self.next_frame_at > now {
            std::thread::sleep(self.next_frame_at - now);
        }
        let period = Duration::from_secs_f32(1.0 / self.fps as f32);
        self.next_frame_at = (self.next_frame_at + period).max(Instant::now());

        let render_started = Instant::now();
        let image = self
            .scene
            .view(&self.camera, self.started.elapsed().as_secs_f64());

        Ok(RawFrame {
            width: image.width,
            height: image.height,
            rgb: image.rgb,
            decode: render_started.elapsed(),
        })
    }

    fn negotiated(&self) -> NegotiatedFormat {
        NegotiatedFormat {
            width: self.camera.intrinsics.width,
            height: self.camera.intrinsics.height,
            fps: self.fps,
            pixel_format: "SYNTHETIC".to_owned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SourceConfig;

    fn config(width: u32, height: u32) -> CameraConfig {
        CameraConfig {
            source: SourceConfig::Synthetic { seat: 0 },
            width,
            height,
            fps: 30,
            ..CameraConfig::default()
        }
    }

    #[test]
    fn a_frame_comes_out_at_the_configured_size() {
        let mut source = SyntheticSource::new(&config(320, 240), 0);
        let frame = source.next_frame().expect("a synthetic frame");

        assert_eq!((frame.width, frame.height), (320, 240));
        assert_eq!(frame.rgb.len(), 320 * 240 * 3);
        assert_eq!(source.negotiated().width, 320);
    }

    #[test]
    fn each_seat_looks_from_a_different_corner() {
        let mut seen: Vec<[f64; 3]> = Vec::new();
        for seat in 0..4 {
            let source = SyntheticSource::new(&config(160, 120), seat);
            let position = source.camera().position();
            for other in &seen {
                let apart = (position.x - other[0]).abs() + (position.z - other[2]).abs();
                assert!(apart > 1.0, "two seats are in the same corner");
            }
            seen.push([position.x, position.y, position.z]);
        }
    }

    /// The point of the rewrite: the frame is a picture of a room with a person
    /// in it, not a stick figure on a dark background. Nothing here can check
    /// that it looks like a person, but a frame that is all one colour is
    /// certainly not one.
    #[test]
    fn the_frame_is_a_scene_rather_than_a_flat_colour() {
        let mut source = SyntheticSource::new(&config(320, 240), 1);
        let frame = source.next_frame().expect("a synthetic frame");

        let brightness: Vec<u8> = frame
            .rgb
            .as_chunks::<3>()
            .0
            .iter()
            .map(|pixel| (pixel[0] / 3) + (pixel[1] / 3) + (pixel[2] / 3))
            .collect();
        let lowest = brightness.iter().copied().min().unwrap_or(0);
        let highest = brightness.iter().copied().max().unwrap_or(0);

        assert!(
            highest - lowest > 60,
            "the frame runs from {lowest} to {highest}, which is not a lit room"
        );
    }

    #[test]
    fn the_scene_moves_between_frames() {
        let mut source = SyntheticSource::new(&config(160, 120), 0);
        let first = source.next_frame().expect("a frame").rgb;
        // Wound forward rather than waited for, so the test does not depend on
        // how long anything takes.
        source.started -= Duration::from_millis(700);
        let later = source.next_frame().expect("a frame").rgb;

        assert_ne!(first, later);
    }
}
