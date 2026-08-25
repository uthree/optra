//! What the pipeline asks of a model.
//!
//! Two capabilities, and nothing about how they are implemented: turn an image
//! into person boxes, and turn a person's box into canonical keypoints. An
//! architecture adapter owns everything between those and the ONNX graph.

use anyhow::Result;

use crate::models::Joint;
use crate::models::keypoints::JointMap;

/// A borrowed RGB8 image.
#[derive(Clone, Copy)]
pub struct ImageView<'a> {
    pub width: u32,
    pub height: u32,
    /// Tightly packed RGB8, `width * height * 3` bytes.
    pub rgb: &'a [u8],
}

impl<'a> ImageView<'a> {
    pub fn new(width: u32, height: u32, rgb: &'a [u8]) -> Self {
        debug_assert_eq!(rgb.len(), width as usize * height as usize * 3);
        Self { width, height, rgb }
    }

    /// Nearest-neighbour sample, clamped at the edges.
    #[inline]
    pub fn sample(&self, x: i32, y: i32) -> [u8; 3] {
        let x = x.clamp(0, self.width as i32 - 1) as usize;
        let y = y.clamp(0, self.height as i32 - 1) as usize;
        let index = (y * self.width as usize + x) * 3;
        [self.rgb[index], self.rgb[index + 1], self.rgb[index + 2]]
    }
}

/// A detected person, in source image pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
}

impl Detection {
    pub fn width(&self) -> f32 {
        self.x2 - self.x1
    }

    pub fn height(&self) -> f32 {
        self.y2 - self.y1
    }

    pub fn center(&self) -> (f32, f32) {
        ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)
    }

    pub fn area(&self) -> f32 {
        (self.width() * self.height()).max(0.0)
    }
}

/// One keypoint in source image pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Keypoint {
    pub x: f32,
    pub y: f32,
    pub confidence: f32,
}

/// A person's keypoints, in the canonical joint set.
///
/// Joints the model does not provide, or provides too weakly to trust, are
/// absent rather than zeroed: the difference matters to triangulation.
#[derive(Debug, Clone, Default)]
pub struct Keypoints2d {
    joints: JointMap<Keypoint>,
}

impl Keypoints2d {
    pub fn get(&self, joint: Joint) -> Option<Keypoint> {
        self.joints.copied(joint)
    }

    pub fn set(&mut self, joint: Joint, keypoint: Keypoint) {
        self.joints.set(joint, keypoint);
    }

    pub fn iter(&self) -> impl Iterator<Item = (Joint, Keypoint)> + '_ {
        self.joints.iter().map(|(joint, kp)| (joint, *kp))
    }

    pub fn count(&self) -> usize {
        self.joints.count()
    }

    pub fn is_empty(&self) -> bool {
        self.joints.is_empty()
    }

    /// Mean confidence over the joints that are present.
    pub fn mean_confidence(&self) -> f32 {
        let (sum, count) = self
            .joints
            .values()
            .fold((0.0, 0usize), |(sum, count), kp| {
                (sum + kp.confidence, count + 1)
            });
        if count == 0 { 0.0 } else { sum / count as f32 }
    }

    /// Bounding box of the present joints, used to track a person between
    /// detector runs.
    pub fn bounds(&self) -> Option<Detection> {
        let mut bounds: Option<(f32, f32, f32, f32)> = None;
        for (_, kp) in self.iter() {
            bounds = Some(match bounds {
                None => (kp.x, kp.y, kp.x, kp.y),
                Some((x1, y1, x2, y2)) => (x1.min(kp.x), y1.min(kp.y), x2.max(kp.x), y2.max(kp.y)),
            });
        }
        bounds.map(|(x1, y1, x2, y2)| Detection {
            x1,
            y1,
            x2,
            y2,
            score: self.mean_confidence(),
        })
    }
}

/// Produces person boxes.
pub trait Detector: Send {
    /// Detects in each image. The outer vector matches `images`.
    fn detect(&mut self, images: &[ImageView<'_>]) -> Result<Vec<Vec<Detection>>>;

    /// Which execution provider this model actually ended up on. A silent
    /// demotion to CPU is the difference between tracking and not.
    fn backend(&self) -> crate::infer::Backend;
}

/// Produces keypoints for cropped people.
pub trait Pose2d: Send {
    /// Estimates keypoints for each `(image, box)` pair. The result matches the
    /// input order.
    fn estimate(&mut self, people: &[(ImageView<'_>, Detection)]) -> Result<Vec<Keypoints2d>>;

    fn backend(&self) -> crate::infer::Backend;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_joints_stay_absent() {
        let mut keypoints = Keypoints2d::default();
        assert!(keypoints.is_empty());
        assert!(keypoints.get(Joint::LeftAnkle).is_none());

        keypoints.set(
            Joint::LeftAnkle,
            Keypoint {
                x: 10.0,
                y: 20.0,
                confidence: 0.9,
            },
        );
        assert_eq!(keypoints.count(), 1);
        assert!(keypoints.get(Joint::RightAnkle).is_none());
    }

    #[test]
    fn bounds_cover_the_present_joints() {
        let mut keypoints = Keypoints2d::default();
        for (joint, x, y) in [
            (Joint::LeftAnkle, 10.0, 100.0),
            (Joint::RightAnkle, 30.0, 90.0),
            (Joint::Nose, 20.0, 10.0),
        ] {
            keypoints.set(
                joint,
                Keypoint {
                    x,
                    y,
                    confidence: 0.5,
                },
            );
        }

        let bounds = keypoints.bounds().expect("some joints are present");
        assert_eq!(
            (bounds.x1, bounds.y1, bounds.x2, bounds.y2),
            (10.0, 10.0, 30.0, 100.0)
        );
    }
}
