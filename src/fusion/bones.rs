//! Measuring the body the cameras are watching.
//!
//! Triangulated joints jitter, and the jitter is not anatomically possible: a
//! shin that measures 41 cm one frame and 46 cm the next is telling you about
//! keypoint noise, not about the leg. Knowing how long the bones really are is
//! what lets the fit downstream reject the half of the noise that would
//! lengthen them, and what lets a joint nobody can see be placed from the ones
//! who can.
//!
//! Lengths are measured rather than assumed. A table of average proportions
//! would be wrong for most people by more than the error being chased, and the
//! cameras are already producing the measurement for free.
//!
//! The lengths are metric because the calibration was: the room was solved from
//! headset positions in SteamVR's own standing frame, so distances come out in
//! metres without anything to scale them against. That makes the headset's
//! floor height a check rather than a source — if the measured leg disagrees
//! with the user's height, something is wrong with the calibration, and
//! rescaling would hide it.

use std::collections::HashMap;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::models::Joint;
use crate::paths;

use super::fuse::Pose3d;

/// A segment between two joints whose length does not change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Bone {
    pub from: Joint,
    pub to: Joint,
}

impl Bone {
    pub const fn new(from: Joint, to: Joint) -> Self {
        Self { from, to }
    }

    /// The same bone on the other side of the body.
    pub fn mirror(self) -> Self {
        Self::new(self.from.mirror(), self.to.mirror())
    }

    /// A key that a bone and its mirror share, so the two sides pool their
    /// measurements.
    fn family(self) -> Self {
        let mirrored = self.mirror();
        if (mirrored.from, mirrored.to) < (self.from, self.to) {
            mirrored
        } else {
            self
        }
    }

    pub fn label(self) -> String {
        format!("{} - {}", self.from.name(), self.to.name())
    }
}

/// The skeleton Optra holds the body to.
///
/// Arms are here even though the trackers Optra drives are lower-body: an elbow
/// is one of the eight points a user can enable, and the shoulder-to-elbow
/// length is what keeps a mis-detected wrist from claiming the elbow moved.
pub const BONES: &[Bone] = &[
    // Pelvis.
    Bone::new(Joint::LeftHip, Joint::RightHip),
    Bone::new(Joint::Hip, Joint::LeftHip),
    Bone::new(Joint::Hip, Joint::RightHip),
    // Legs.
    Bone::new(Joint::LeftHip, Joint::LeftKnee),
    Bone::new(Joint::RightHip, Joint::RightKnee),
    Bone::new(Joint::LeftKnee, Joint::LeftAnkle),
    Bone::new(Joint::RightKnee, Joint::RightAnkle),
    // Feet. The heel and toe are what give a foot an orientation rather than
    // just a position, so their spacing matters as much as the leg's.
    Bone::new(Joint::LeftAnkle, Joint::LeftHeel),
    Bone::new(Joint::RightAnkle, Joint::RightHeel),
    Bone::new(Joint::LeftAnkle, Joint::LeftBigToe),
    Bone::new(Joint::RightAnkle, Joint::RightBigToe),
    Bone::new(Joint::LeftHeel, Joint::LeftBigToe),
    Bone::new(Joint::RightHeel, Joint::RightBigToe),
    // Spine and shoulders.
    Bone::new(Joint::Hip, Joint::Neck),
    Bone::new(Joint::Neck, Joint::Head),
    Bone::new(Joint::Neck, Joint::LeftShoulder),
    Bone::new(Joint::Neck, Joint::RightShoulder),
    Bone::new(Joint::LeftShoulder, Joint::RightShoulder),
    // Arms.
    Bone::new(Joint::LeftShoulder, Joint::LeftElbow),
    Bone::new(Joint::RightShoulder, Joint::RightElbow),
    Bone::new(Joint::LeftElbow, Joint::LeftWrist),
    Bone::new(Joint::RightElbow, Joint::RightWrist),
];

#[derive(Debug, Clone)]
pub struct MeasureOptions {
    /// Positional uncertainty a joint must be under before its distance to a
    /// neighbour counts as a measurement, in metres.
    ///
    /// A bone between two joints located to five centimetres each says nothing
    /// about the bone.
    pub max_sigma: f64,
    /// Samples a bone needs before its length is trusted.
    pub min_samples: usize,
    /// How much a bone's samples may scatter, relative to its length, before
    /// the measurement is treated as never having settled.
    pub max_scatter: f64,
    /// Samples kept per bone family. Enough to measure through a walk; past it
    /// the oldest are dropped.
    pub capacity: usize,
}

impl Default for MeasureOptions {
    fn default() -> Self {
        Self {
            max_sigma: 0.03,
            min_samples: 120,
            max_scatter: 0.08,
            capacity: 3000,
        }
    }
}

/// One bone as it was measured.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BoneLength {
    pub bone: Bone,
    /// Length in metres.
    pub length: f64,
    /// How many observations went into it, over both sides of the body.
    pub samples: usize,
    /// Robust spread of those observations, in metres. Large means the
    /// measurement never settled and the number should not be relied on.
    pub scatter: f64,
}

impl BoneLength {
    /// Whether this length is worth holding the body to.
    pub fn is_settled(&self, options: &MeasureOptions) -> bool {
        self.samples >= options.min_samples
            && self.length > 0.0
            && self.scatter / self.length <= options.max_scatter
    }
}

/// A measured body.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Skeleton {
    pub bones: Vec<BoneLength>,
    pub measured_at: Option<String>,
}

impl Skeleton {
    pub fn length(&self, bone: Bone) -> Option<f64> {
        self.bones
            .iter()
            .find(|measured| measured.bone == bone)
            .map(|measured| measured.length)
    }

    pub fn get(&self, bone: Bone) -> Option<&BoneLength> {
        self.bones.iter().find(|measured| measured.bone == bone)
    }

    /// Hip to ankle with the leg straight, in metres.
    ///
    /// The number to compare against the user's height: it and the headset's
    /// floor height are two independent measurements of the same person, and
    /// they should agree.
    pub fn leg_length(&self) -> Option<f64> {
        let thigh = self.length(Bone::new(Joint::LeftHip, Joint::LeftKnee))?;
        let shin = self.length(Bone::new(Joint::LeftKnee, Joint::LeftAnkle))?;
        Some(thigh + shin)
    }

    /// Fraction of the skeleton that has a settled length.
    pub fn coverage(&self, options: &MeasureOptions) -> f32 {
        if BONES.is_empty() {
            return 0.0;
        }
        let settled = self
            .bones
            .iter()
            .filter(|measured| measured.is_settled(options))
            .count();
        settled as f32 / BONES.len() as f32
    }

    /// Writes the measurement beside the config.
    ///
    /// Not into the room profile, where the design first put it. A body belongs
    /// to a person and a room profile belongs to a set of cameras: storing them
    /// together would make a user re-measure themselves every time a camera is
    /// nudged, and would give two people sharing a machine one set of legs.
    pub fn save(&self) -> Result<()> {
        let path = paths::body_file()?;
        let text = toml::to_string_pretty(self)?;

        let temporary = path.with_extension("toml.tmp");
        std::fs::write(&temporary, text)?;
        std::fs::rename(&temporary, &path)?;

        tracing::info!(path = %path.display(), "saved the body measurement");
        Ok(())
    }

    pub fn load() -> Result<Self> {
        let path = paths::body_file()?;
        Ok(toml::from_str(&std::fs::read_to_string(path)?)?)
    }

    /// Loads the measurement if there is one, without treating its absence as
    /// a problem.
    pub fn load_or_default() -> Self {
        Self::load().unwrap_or_default()
    }
}

/// Accumulates bone lengths from fused poses.
#[derive(Debug, Clone)]
pub struct BoneMeter {
    options: MeasureOptions,
    /// Samples per bone family, so a limb the cameras rarely see borrows the
    /// measurement of the one they do.
    samples: HashMap<Bone, Vec<f64>>,
    poses: usize,
}

impl BoneMeter {
    pub fn new(options: MeasureOptions) -> Self {
        Self {
            options,
            samples: HashMap::new(),
            poses: 0,
        }
    }

    pub fn poses(&self) -> usize {
        self.poses
    }

    pub fn reset(&mut self) {
        self.samples.clear();
        self.poses = 0;
    }

    /// Records every bone this pose was confident enough about.
    pub fn observe(&mut self, pose: &Pose3d) {
        self.poses += 1;

        for bone in BONES {
            let (Some(from), Some(to)) = (pose.get(bone.from), pose.get(bone.to)) else {
                continue;
            };
            if from.sigma > self.options.max_sigma || to.sigma > self.options.max_sigma {
                continue;
            }

            let samples = self.samples.entry(bone.family()).or_default();
            samples.push((to.point - from.point).norm());
            if samples.len() > self.options.capacity {
                samples.remove(0);
            }
        }
    }

    /// The measurement so far.
    ///
    /// The estimate is a median rather than a mean, because a limb the model
    /// occasionally puts on the other leg produces samples that are wrong by
    /// tens of centimetres, and a mean would carry all of them.
    pub fn finish(&self) -> Skeleton {
        let mut bones = Vec::new();

        for bone in BONES {
            let Some(samples) = self.samples.get(&bone.family()) else {
                continue;
            };
            if samples.is_empty() {
                continue;
            }

            let length = median(samples);
            bones.push(BoneLength {
                bone: *bone,
                length,
                samples: samples.len(),
                scatter: scatter(samples, length),
            });
        }

        Skeleton {
            bones,
            measured_at: Some(chrono::Local::now().to_rfc3339()),
        }
    }

    /// Bones still short of a settled measurement, for the UI to point at.
    pub fn outstanding(&self) -> Vec<Bone> {
        let measured = self.finish();
        BONES
            .iter()
            .copied()
            .filter(|bone| {
                measured
                    .get(*bone)
                    .is_none_or(|length| !length.is_settled(&self.options))
            })
            .collect()
    }
}

impl Default for BoneMeter {
    fn default() -> Self {
        Self::new(MeasureOptions::default())
    }
}

fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    match sorted.len() {
        0 => 0.0,
        length if length % 2 == 1 => sorted[length / 2],
        length => 0.5 * (sorted[length / 2 - 1] + sorted[length / 2]),
    }
}

/// Robust spread, as the median absolute deviation scaled to compare with a
/// standard deviation.
fn scatter(samples: &[f64], centre: f64) -> f64 {
    const NORMAL: f64 = 1.4826;
    let deviations: Vec<f64> = samples
        .iter()
        .map(|sample| (sample - centre).abs())
        .collect();
    median(&deviations) * NORMAL
}

#[cfg(test)]
mod tests {
    use nalgebra::Point3;

    use super::*;
    use crate::fusion::fuse::FusedJoint;

    fn joint(point: Point3<f64>, sigma: f64) -> FusedJoint {
        FusedJoint {
            point,
            sigma,
            residual: 0.0,
            weights: vec![(0, 0.5), (1, 0.5)],
            rejected: Vec::new(),
        }
    }

    /// A body with legs of a known length, jittered by `noise` metres in a
    /// repeatable pattern.
    fn pose(step: usize, thigh: f64, noise: f64, sigma: f64) -> Pose3d {
        let mut pose = Pose3d::empty(std::time::Instant::now());
        let wobble = |phase: f64| noise * ((step as f64 * 1.7 + phase).sin());

        for (side, sign) in [(Joint::LeftHip, -1.0), (Joint::RightHip, 1.0)] {
            pose.set(side, joint(Point3::new(0.12 * sign, 0.95, 0.0), sigma));
        }
        pose.set(
            Joint::LeftKnee,
            joint(Point3::new(-0.12, 0.95 - thigh + wobble(0.0), 0.0), sigma),
        );
        pose.set(
            Joint::RightKnee,
            joint(Point3::new(0.12, 0.95 - thigh + wobble(2.0), 0.0), sigma),
        );
        pose
    }

    #[test]
    fn a_bone_is_measured_from_the_poses_it_appears_in() {
        let mut meter = BoneMeter::default();
        for step in 0..400 {
            meter.observe(&pose(step, 0.44, 0.005, 0.004));
        }

        let skeleton = meter.finish();
        let thigh = skeleton
            .get(Bone::new(Joint::LeftHip, Joint::LeftKnee))
            .expect("the thigh was visible throughout");

        assert!(
            (thigh.length - 0.44).abs() < 0.005,
            "measured {} instead of 0.44",
            thigh.length
        );
        assert!(thigh.is_settled(&MeasureOptions::default()));
    }

    /// Both sides pool their samples, so a leg the cameras half-see still gets
    /// a length from the one they see well.
    #[test]
    fn the_two_sides_of_the_body_share_one_measurement() {
        let mut meter = BoneMeter::default();
        for step in 0..400 {
            let mut pose = pose(step, 0.44, 0.004, 0.004);
            // The right knee is only visible now and then.
            if step % 20 != 0 {
                pose = {
                    let mut sparse = Pose3d::empty(pose.at);
                    for (name, fused) in pose.iter() {
                        if name != Joint::RightKnee {
                            sparse.set(name, fused.clone());
                        }
                    }
                    sparse
                };
            }
            meter.observe(&pose);
        }

        let skeleton = meter.finish();
        let left = skeleton
            .get(Bone::new(Joint::LeftHip, Joint::LeftKnee))
            .unwrap();
        let right = skeleton
            .get(Bone::new(Joint::RightHip, Joint::RightKnee))
            .unwrap();

        assert_eq!(left.length, right.length);
        assert!(right.is_settled(&MeasureOptions::default()));
    }

    /// A limb the model occasionally puts somewhere else produces samples wrong
    /// by tens of centimetres. The median should not notice.
    #[test]
    fn occasional_nonsense_does_not_move_the_answer() {
        let mut meter = BoneMeter::default();
        for step in 0..400 {
            let thigh = if step % 9 == 0 { 0.90 } else { 0.44 };
            meter.observe(&pose(step, thigh, 0.004, 0.004));
        }

        let thigh = meter
            .finish()
            .length(Bone::new(Joint::LeftHip, Joint::LeftKnee))
            .unwrap();
        assert!((thigh - 0.44).abs() < 0.01, "measured {thigh}");
    }

    /// Joints nobody could locate say nothing about the bone between them.
    #[test]
    fn uncertain_joints_are_not_measured() {
        let mut meter = BoneMeter::default();
        for step in 0..400 {
            meter.observe(&pose(step, 0.44, 0.05, 0.20));
        }

        assert!(meter.finish().bones.is_empty());
        assert_eq!(meter.poses(), 400);
    }

    /// A measurement that never settles has to say so, or the fit will hold the
    /// body to a number that came out of noise.
    #[test]
    fn a_length_that_never_settles_is_not_reported_as_settled() {
        let mut meter = BoneMeter::default();
        for step in 0..400 {
            meter.observe(&pose(step, 0.44, 0.12, 0.01));
        }

        let thigh = meter
            .finish()
            .get(Bone::new(Joint::LeftHip, Joint::LeftKnee))
            .copied()
            .unwrap();
        assert!(!thigh.is_settled(&MeasureOptions::default()));
        assert!(thigh.scatter > 0.04, "scatter was {}", thigh.scatter);
    }

    #[test]
    fn every_bone_has_a_mirror_that_is_also_a_bone() {
        for bone in BONES {
            let mirrored = bone.mirror();
            // A bone that spans the midline mirrors onto itself with its ends
            // swapped, which is the same segment and needs no separate entry.
            let spans_the_midline = mirrored == Bone::new(bone.to, bone.from);
            assert!(
                BONES.contains(&mirrored) || spans_the_midline,
                "{} has no mirror in the table",
                bone.label()
            );
            assert_eq!(
                bone.family(),
                mirrored.family(),
                "{} and its mirror must pool their samples",
                bone.label()
            );
        }
    }

    #[test]
    fn coverage_grows_as_bones_settle() {
        let options = MeasureOptions::default();
        let mut meter = BoneMeter::default();
        assert_eq!(meter.finish().coverage(&options), 0.0);

        for step in 0..400 {
            meter.observe(&pose(step, 0.44, 0.004, 0.004));
        }

        let coverage = meter.finish().coverage(&options);
        assert!(coverage > 0.0 && coverage < 1.0, "coverage {coverage}");
    }
}
