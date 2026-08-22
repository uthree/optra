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

    /// Index into a dense per-joint array.
    pub fn index(self) -> usize {
        self as usize
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

#[cfg(test)]
mod tests {
    use super::*;

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
