//! A generated scene, used to exercise the multi-camera pipeline without
//! multiple physical cameras.
//!
//! The scene is a virtual room with a figure walking in a circle, rendered
//! through a pinhole camera placed in one of the ceiling corners. Because the
//! camera pose and the figure's joint positions are known exactly, the same
//! source doubles as ground truth once triangulation exists.

use std::time::{Duration, Instant};

use anyhow::Result;

use super::{FrameSource, NegotiatedFormat, RawFrame};
use crate::config::CameraConfig;

/// Half the size of the virtual room, in metres.
const ROOM_HALF: f32 = 2.0;
/// Height of the virtual ceiling, in metres.
const CEILING: f32 = 2.4;
/// Horizontal field of view of the virtual camera, in radians.
const HFOV: f32 = 70.0_f32.to_radians();

type Vec3 = [f32; 3];

pub struct SyntheticSource {
    width: u32,
    height: u32,
    fps: u32,
    eye: Vec3,
    basis: Basis,
    started: Instant,
    next_frame_at: Instant,
}

/// Camera axes in world space.
struct Basis {
    right: Vec3,
    up: Vec3,
    forward: Vec3,
}

impl SyntheticSource {
    pub fn new(config: &CameraConfig, seat: u32) -> Self {
        let eye = seat_position(seat);
        let basis = look_at(eye, [0.0, 1.0, 0.0]);
        let now = Instant::now();

        Self {
            width: config.width.max(64),
            height: config.height.max(64),
            fps: config.fps.max(1),
            eye,
            basis,
            started: now,
            next_frame_at: now,
        }
    }

    /// Projects a world point into pixel coordinates, or `None` if it is behind
    /// the camera.
    fn project(&self, p: Vec3) -> Option<(f32, f32)> {
        let d = sub(p, self.eye);
        let z = dot(d, self.basis.forward);
        if z <= 0.05 {
            return None;
        }

        let fx = 0.5 * self.width as f32 / (HFOV * 0.5).tan();
        let fy = fx;
        let u = fx * dot(d, self.basis.right) / z + self.width as f32 * 0.5;
        // Image y grows downward, world y grows upward.
        let v = -fy * dot(d, self.basis.up) / z + self.height as f32 * 0.5;
        Some((u, v))
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
        let t = self.started.elapsed().as_secs_f32();
        let mut canvas = Canvas::new(self.width, self.height);
        canvas.clear([18, 20, 26]);

        self.draw_floor(&mut canvas);
        self.draw_figure(&mut canvas, t);

        Ok(RawFrame {
            width: self.width,
            height: self.height,
            rgb: canvas.rgb,
            decode: render_started.elapsed(),
        })
    }

    fn negotiated(&self) -> NegotiatedFormat {
        NegotiatedFormat {
            width: self.width,
            height: self.height,
            fps: self.fps,
            pixel_format: "SYNTHETIC".to_owned(),
        }
    }
}

impl SyntheticSource {
    fn draw_floor(&self, canvas: &mut Canvas) {
        let color = [52, 58, 72];
        let steps = 8;
        for i in 0..=steps {
            let offset = -ROOM_HALF + 2.0 * ROOM_HALF * i as f32 / steps as f32;
            self.draw_segment(
                canvas,
                [offset, 0.0, -ROOM_HALF],
                [offset, 0.0, ROOM_HALF],
                color,
            );
            self.draw_segment(
                canvas,
                [-ROOM_HALF, 0.0, offset],
                [ROOM_HALF, 0.0, offset],
                color,
            );
        }
    }

    fn draw_figure(&self, canvas: &mut Canvas, t: f32) {
        let joints = figure(t);
        let bones = [
            (Joint::Head, Joint::Chest),
            (Joint::Chest, Joint::Hips),
            (Joint::Hips, Joint::LeftKnee),
            (Joint::LeftKnee, Joint::LeftAnkle),
            (Joint::Hips, Joint::RightKnee),
            (Joint::RightKnee, Joint::RightAnkle),
        ];

        for (a, b) in bones {
            self.draw_segment(
                canvas,
                joints[a as usize],
                joints[b as usize],
                [220, 226, 240],
            );
        }
        for (index, joint) in joints.iter().enumerate() {
            let color = if index == Joint::Head as usize {
                [250, 190, 90]
            } else {
                [120, 200, 250]
            };
            if let Some((u, v)) = self.project(*joint) {
                canvas.disc(u, v, 4.0, color);
            }
        }
    }

    /// Draws a world-space line by sampling it; the segments are short enough
    /// that a proper clipped projection would not look different.
    fn draw_segment(&self, canvas: &mut Canvas, a: Vec3, b: Vec3, color: [u8; 3]) {
        const SAMPLES: usize = 24;
        let mut previous = None;
        for i in 0..=SAMPLES {
            let s = i as f32 / SAMPLES as f32;
            let p = [
                a[0] + (b[0] - a[0]) * s,
                a[1] + (b[1] - a[1]) * s,
                a[2] + (b[2] - a[2]) * s,
            ];
            let current = self.project(p);
            if let (Some((x0, y0)), Some((x1, y1))) = (previous, current) {
                canvas.line(x0, y0, x1, y1, color);
            }
            previous = current;
        }
    }
}

#[derive(Clone, Copy)]
enum Joint {
    Head,
    Chest,
    Hips,
    LeftKnee,
    LeftAnkle,
    RightKnee,
    RightAnkle,
}

/// The figure walks a circle while its legs swing, so that the lower body moves
/// independently of the body as a whole.
fn figure(t: f32) -> [Vec3; 7] {
    let angle = t * 0.35;
    let centre = [angle.cos() * 1.1, 0.0, angle.sin() * 1.1];
    let step = (t * 2.2).sin() * 0.28;

    // Facing along the walking direction, so the legs swing forward and back.
    let facing = [-angle.sin(), 0.0, angle.cos()];
    let side = [angle.cos(), 0.0, angle.sin()];

    let at = |lateral: f32, forward: f32, height: f32| -> Vec3 {
        [
            centre[0] + side[0] * lateral + facing[0] * forward,
            height,
            centre[2] + side[2] * lateral + facing[2] * forward,
        ]
    };

    [
        at(0.0, 0.0, 1.70),
        at(0.0, 0.0, 1.35),
        at(0.0, 0.0, 0.95),
        at(-0.12, step * 0.5, 0.50),
        at(-0.12, step, 0.08),
        at(0.12, -step * 0.5, 0.50),
        at(0.12, -step, 0.08),
    ]
}

/// Ceiling corner for a given seat, cycling through the four corners.
fn seat_position(seat: u32) -> Vec3 {
    let corners = [
        [-ROOM_HALF * 0.9, CEILING, -ROOM_HALF * 0.9],
        [ROOM_HALF * 0.9, CEILING, -ROOM_HALF * 0.9],
        [ROOM_HALF * 0.9, CEILING, ROOM_HALF * 0.9],
        [-ROOM_HALF * 0.9, CEILING, ROOM_HALF * 0.9],
    ];
    corners[seat as usize % corners.len()]
}

fn look_at(eye: Vec3, target: Vec3) -> Basis {
    let forward = normalize(sub(target, eye));
    let world_up = [0.0, 1.0, 0.0];
    let right = normalize(cross(forward, world_up));
    let up = cross(right, forward);
    Basis { right, up, forward }
}

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: Vec3) -> Vec3 {
    let len = dot(v, v).sqrt().max(1e-6);
    [v[0] / len, v[1] / len, v[2] / len]
}

/// A minimal RGB8 raster target.
struct Canvas {
    width: i32,
    height: i32,
    rgb: Vec<u8>,
}

impl Canvas {
    fn new(width: u32, height: u32) -> Self {
        Self {
            width: width as i32,
            height: height as i32,
            rgb: vec![0; width as usize * height as usize * 3],
        }
    }

    fn clear(&mut self, color: [u8; 3]) {
        for pixel in self.rgb.as_chunks_mut::<3>().0 {
            *pixel = color;
        }
    }

    fn put(&mut self, x: i32, y: i32, color: [u8; 3]) {
        if x < 0 || y < 0 || x >= self.width || y >= self.height {
            return;
        }
        let index = (y as usize * self.width as usize + x as usize) * 3;
        self.rgb[index..index + 3].copy_from_slice(&color);
    }

    fn line(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, color: [u8; 3]) {
        let steps = ((x1 - x0).abs().max((y1 - y0).abs()).ceil() as i32).max(1);
        for i in 0..=steps {
            let s = i as f32 / steps as f32;
            self.put(
                (x0 + (x1 - x0) * s).round() as i32,
                (y0 + (y1 - y0) * s).round() as i32,
                color,
            );
        }
    }

    fn disc(&mut self, cx: f32, cy: f32, radius: f32, color: [u8; 3]) {
        let r = radius.ceil() as i32;
        for dy in -r..=r {
            for dx in -r..=r {
                if (dx * dx + dy * dy) as f32 <= radius * radius {
                    self.put(cx.round() as i32 + dx, cy.round() as i32 + dy, color);
                }
            }
        }
    }
}
