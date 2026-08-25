//! The model catalogue: what models exist, where they come from, and what
//! shape their inputs and outputs have.

pub mod keypoints;
pub mod manifest;
pub mod store;

pub use keypoints::{Joint, JointMap, Layout};
pub use manifest::{Manifest, ModelKind, ModelSource, ModelSpec};
