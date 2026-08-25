//! Sending the reconstructed body to whatever is going to wear it.

pub mod pose;
pub mod sink;
pub mod stage;
pub mod vmt;
pub mod vrchat;

pub use pose::{Posture, PostureJoint, TrackerPose, TrackerRole};
pub use sink::{TrackerFrame, TrackerSink, assign};
