//! Keypoint layouts.
//!
//! Models disagree about how many keypoints they produce and in what order.
//! Everything downstream of inference works on the canonical [`Joint`] set, and
//! the mapping from a model's own ordering into it is data, not code, so a
//! model with an unfamiliar layout is supported by adding a table rather than
//! by editing this file.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

/// Layout tables shipped with Optra.
const BUILTIN: &str = include_str!("keypoints.toml");

/// A joint Optra knows how to use.
///
/// Not every model provides every joint; the ones a model does not provide are
/// simply absent, and the fusion stage already deals with absent joints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Joint {
    Nose,
    LeftEye,
    RightEye,
    LeftEar,
    RightEar,
    Head,
    Neck,
    LeftShoulder,
    RightShoulder,
    LeftElbow,
    RightElbow,
    LeftWrist,
    RightWrist,
    Hip,
    LeftHip,
    RightHip,
    LeftKnee,
    RightKnee,
    LeftAnkle,
    RightAnkle,
    LeftHeel,
    RightHeel,
    LeftBigToe,
    RightBigToe,
    LeftSmallToe,
    RightSmallToe,
}

impl Joint {
    pub const ALL: [Joint; 26] = [
        Joint::Nose,
        Joint::LeftEye,
        Joint::RightEye,
        Joint::LeftEar,
        Joint::RightEar,
        Joint::Head,
        Joint::Neck,
        Joint::LeftShoulder,
        Joint::RightShoulder,
        Joint::LeftElbow,
        Joint::RightElbow,
        Joint::LeftWrist,
        Joint::RightWrist,
        Joint::Hip,
        Joint::LeftHip,
        Joint::RightHip,
        Joint::LeftKnee,
        Joint::RightKnee,
        Joint::LeftAnkle,
        Joint::RightAnkle,
        Joint::LeftHeel,
        Joint::RightHeel,
        Joint::LeftBigToe,
        Joint::RightBigToe,
        Joint::LeftSmallToe,
        Joint::RightSmallToe,
    ];

    /// The joint's name, matching the one used in `keypoints.toml`.
    pub fn name(self) -> &'static str {
        match self {
            Joint::Nose => "nose",
            Joint::LeftEye => "left_eye",
            Joint::RightEye => "right_eye",
            Joint::LeftEar => "left_ear",
            Joint::RightEar => "right_ear",
            Joint::Head => "head",
            Joint::Neck => "neck",
            Joint::LeftShoulder => "left_shoulder",
            Joint::RightShoulder => "right_shoulder",
            Joint::LeftElbow => "left_elbow",
            Joint::RightElbow => "right_elbow",
            Joint::LeftWrist => "left_wrist",
            Joint::RightWrist => "right_wrist",
            Joint::Hip => "hip",
            Joint::LeftHip => "left_hip",
            Joint::RightHip => "right_hip",
            Joint::LeftKnee => "left_knee",
            Joint::RightKnee => "right_knee",
            Joint::LeftAnkle => "left_ankle",
            Joint::RightAnkle => "right_ankle",
            Joint::LeftHeel => "left_heel",
            Joint::RightHeel => "right_heel",
            Joint::LeftBigToe => "left_big_toe",
            Joint::RightBigToe => "right_big_toe",
            Joint::LeftSmallToe => "left_small_toe",
            Joint::RightSmallToe => "right_small_toe",
        }
    }

    /// Index into a dense per-joint array.
    pub fn index(self) -> usize {
        self as usize
    }

    /// The joint on the other side of the body, or the joint itself for the
    /// ones on the midline.
    ///
    /// People are symmetric to within a couple of percent, so a limb the
    /// cameras rarely see can borrow the measurement of the one they do.
    pub fn mirror(self) -> Joint {
        match self {
            Joint::LeftEye => Joint::RightEye,
            Joint::RightEye => Joint::LeftEye,
            Joint::LeftEar => Joint::RightEar,
            Joint::RightEar => Joint::LeftEar,
            Joint::LeftShoulder => Joint::RightShoulder,
            Joint::RightShoulder => Joint::LeftShoulder,
            Joint::LeftElbow => Joint::RightElbow,
            Joint::RightElbow => Joint::LeftElbow,
            Joint::LeftWrist => Joint::RightWrist,
            Joint::RightWrist => Joint::LeftWrist,
            Joint::LeftHip => Joint::RightHip,
            Joint::RightHip => Joint::LeftHip,
            Joint::LeftKnee => Joint::RightKnee,
            Joint::RightKnee => Joint::LeftKnee,
            Joint::LeftAnkle => Joint::RightAnkle,
            Joint::RightAnkle => Joint::LeftAnkle,
            Joint::LeftHeel => Joint::RightHeel,
            Joint::RightHeel => Joint::LeftHeel,
            Joint::LeftBigToe => Joint::RightBigToe,
            Joint::RightBigToe => Joint::LeftBigToe,
            Joint::LeftSmallToe => Joint::RightSmallToe,
            Joint::RightSmallToe => Joint::LeftSmallToe,
            midline => midline,
        }
    }

    /// True for the joints the lower-body trackers are built from. These are
    /// the ones whose absence actually matters.
    pub fn is_lower_body(self) -> bool {
        matches!(
            self,
            Joint::Hip
                | Joint::LeftHip
                | Joint::RightHip
                | Joint::LeftKnee
                | Joint::RightKnee
                | Joint::LeftAnkle
                | Joint::RightAnkle
                | Joint::LeftHeel
                | Joint::RightHeel
                | Joint::LeftBigToe
                | Joint::RightBigToe
                | Joint::LeftSmallToe
                | Joint::RightSmallToe
        )
    }
}

/// A value per joint, for the joints that have one.
///
/// Six things in this application are a body's worth of something with gaps in
/// it: the keypoints a model reports, the same resampled onto a fusion tick,
/// the triangulated pose, the fitted one, the smoothed one, and the
/// simulator's ground truth. Every one of them had its own array, its own
/// `get`, `set`, `iter`, `count` and `is_empty`, and the six copies had already
/// drifted into four different signatures.
///
/// The distinction it exists to keep is between a joint that is absent and a
/// joint that is zero. Those are not the same thing anywhere downstream — an
/// absent joint contributes no ray to a triangulation and a zeroed one drags
/// the answer to the origin — so the gaps are `None` rather than a sentinel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JointMap<T> {
    values: [Option<T>; Joint::ALL.len()],
}

impl<T> Default for JointMap<T> {
    fn default() -> Self {
        // `from_fn` rather than `[None; N]`, which would need `T: Copy` and
        // rule out the half of these that carry a `Vec`.
        Self {
            values: std::array::from_fn(|_| None),
        }
    }
}

impl<T> JointMap<T> {
    pub fn get(&self, joint: Joint) -> Option<&T> {
        self.values[joint.index()].as_ref()
    }

    pub fn set(&mut self, joint: Joint, value: T) {
        self.values[joint.index()] = Some(value);
    }

    /// Forgets a joint, which is not the same as setting it to anything.
    pub fn clear(&mut self, joint: Joint) {
        self.values[joint.index()] = None;
    }

    /// The joints that have a value, in the canonical order.
    ///
    /// The order is [`Joint::ALL`] and not a hash map's whim, which is what
    /// makes one run's diagnostics comparable with another's.
    pub fn iter(&self) -> impl Iterator<Item = (Joint, &T)> + '_ {
        Joint::ALL
            .iter()
            .filter_map(|joint| self.get(*joint).map(|value| (*joint, value)))
    }

    /// The values alone, for a summary that does not care which joint is which.
    pub fn values(&self) -> impl Iterator<Item = &T> + '_ {
        self.values.iter().flatten()
    }

    pub fn count(&self) -> usize {
        self.values.iter().filter(|value| value.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }
}

impl<T: Copy> JointMap<T> {
    /// The value itself rather than a reference, for the small payloads where
    /// a borrow is more awkward than the copy is expensive.
    pub fn copied(&self, joint: Joint) -> Option<T> {
        self.values[joint.index()]
    }
}

/// How one model's keypoint ordering maps into the canonical set.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Layout {
    /// How many keypoints the model emits.
    pub count: usize,
    /// Canonical joint to the index the model emits it at.
    pub joints: BTreeMap<Joint, usize>,
}

impl Layout {
    fn validate(&self, name: &str) -> Result<()> {
        for (joint, index) in &self.joints {
            if *index >= self.count {
                bail!(
                    "{name} maps {joint:?} to index {index}, past its {} keypoints",
                    self.count
                );
            }
        }
        Ok(())
    }

    /// Whether this layout provides the heel and toe points that make real foot
    /// orientation possible.
    pub fn has_feet(&self) -> bool {
        [
            Joint::LeftHeel,
            Joint::RightHeel,
            Joint::LeftBigToe,
            Joint::RightBigToe,
        ]
        .iter()
        .all(|joint| self.joints.contains_key(joint))
    }
}

fn table() -> &'static BTreeMap<String, Layout> {
    static TABLE: OnceLock<BTreeMap<String, Layout>> = OnceLock::new();

    TABLE.get_or_init(|| match load() {
        Ok(layouts) => layouts,
        Err(err) => {
            // The builtin table is compiled in, so a failure here is a bug
            // rather than a user problem; carrying on with no layouts would
            // turn it into a confusing "model has no keypoints" instead.
            panic!("the builtin keypoint layouts are broken: {err:#}");
        }
    })
}

fn load() -> Result<BTreeMap<String, Layout>> {
    let layouts: BTreeMap<String, Layout> =
        toml::from_str(BUILTIN).context("failed to parse the keypoint layouts")?;
    for (name, layout) in &layouts {
        layout.validate(name)?;
    }
    Ok(layouts)
}

/// Looks up a layout by the name a model spec refers to.
pub fn layout(name: &str) -> Option<&'static Layout> {
    table().get(name)
}

/// Every known layout name.
pub fn names() -> Vec<&'static str> {
    table().keys().map(String::as_str).collect()
}

/// Bones, for drawing a skeleton.
///
/// A bone is only drawn when both of its ends are present, so the same list
/// works for a 17-point model and a 26-point one: the extra bones simply do not
/// appear. The torso is spanned both through the spine and around the shoulders
/// and hips, so a layout without a neck or pelvis point still looks like a body.
pub const BONES: [(Joint, Joint); 24] = [
    (Joint::Head, Joint::Neck),
    (Joint::Nose, Joint::Head),
    (Joint::Neck, Joint::LeftShoulder),
    (Joint::Neck, Joint::RightShoulder),
    (Joint::LeftShoulder, Joint::RightShoulder),
    (Joint::LeftShoulder, Joint::LeftElbow),
    (Joint::LeftElbow, Joint::LeftWrist),
    (Joint::RightShoulder, Joint::RightElbow),
    (Joint::RightElbow, Joint::RightWrist),
    (Joint::Neck, Joint::Hip),
    (Joint::LeftShoulder, Joint::LeftHip),
    (Joint::RightShoulder, Joint::RightHip),
    (Joint::Hip, Joint::LeftHip),
    (Joint::Hip, Joint::RightHip),
    (Joint::LeftHip, Joint::RightHip),
    (Joint::LeftHip, Joint::LeftKnee),
    (Joint::LeftKnee, Joint::LeftAnkle),
    (Joint::RightHip, Joint::RightKnee),
    (Joint::RightKnee, Joint::RightAnkle),
    (Joint::LeftAnkle, Joint::LeftHeel),
    (Joint::LeftHeel, Joint::LeftBigToe),
    (Joint::LeftBigToe, Joint::LeftSmallToe),
    (Joint::RightAnkle, Joint::RightHeel),
    (Joint::RightHeel, Joint::RightBigToe),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_joint_map_starts_empty_and_remembers_what_it_is_given() {
        let mut map: JointMap<f64> = JointMap::default();
        assert!(map.is_empty());
        assert_eq!(map.count(), 0);
        assert!(map.get(Joint::LeftKnee).is_none());

        map.set(Joint::LeftKnee, 1.5);
        map.set(Joint::RightAnkle, 2.5);

        assert_eq!(map.copied(Joint::LeftKnee), Some(1.5));
        assert!(map.get(Joint::RightKnee).is_none());
        assert_eq!(map.count(), 2);
        assert!(!map.is_empty());
    }

    /// Forgetting a joint and setting it to zero are different answers, and
    /// everything downstream treats them differently.
    #[test]
    fn clearing_a_joint_leaves_no_value_rather_than_a_zero() {
        let mut map: JointMap<f64> = JointMap::default();
        map.set(Joint::Hip, 0.0);
        assert_eq!(map.count(), 1);

        map.clear(Joint::Hip);
        assert_eq!(map.count(), 0);
        assert!(map.get(Joint::Hip).is_none());
    }

    #[test]
    fn a_joint_map_iterates_in_the_canonical_order() {
        let mut map: JointMap<u32> = JointMap::default();
        for joint in [Joint::RightAnkle, Joint::Nose, Joint::Hip] {
            map.set(joint, joint.index() as u32);
        }

        let order: Vec<Joint> = map.iter().map(|(joint, _)| joint).collect();
        assert_eq!(order, vec![Joint::Nose, Joint::Hip, Joint::RightAnkle]);

        // The values alone come out in the same order, which is what makes a
        // summary over them comparable between one run and the next.
        let values: Vec<u32> = map.values().copied().collect();
        assert_eq!(
            values,
            order
                .iter()
                .map(|joint| joint.index() as u32)
                .collect::<Vec<u32>>()
        );
    }

    #[test]
    fn the_builtin_layouts_are_valid() {
        let layouts = load().expect("builtin layouts");
        assert!(layouts.contains_key("coco17"));
        assert!(layouts.contains_key("halpe26"));
        assert!(layouts.contains_key("coco_wholebody133"));
    }

    #[test]
    fn coco17_has_no_feet_but_halpe26_does() {
        assert!(!layout("coco17").unwrap().has_feet());
        assert!(layout("halpe26").unwrap().has_feet());
        assert!(layout("coco_wholebody133").unwrap().has_feet());
    }

    #[test]
    fn every_layout_provides_the_joints_the_lower_body_needs() {
        for name in names() {
            let layout = layout(name).unwrap();
            for joint in [
                Joint::LeftKnee,
                Joint::RightKnee,
                Joint::LeftAnkle,
                Joint::RightAnkle,
            ] {
                assert!(
                    layout.joints.contains_key(&joint),
                    "{name} does not provide {joint:?}"
                );
            }
        }
    }
}
