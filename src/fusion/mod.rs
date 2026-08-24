//! The fusion stage.
//!
//! Turns per-camera 2D keypoints into one 3D skeleton. The work divides into
//! putting the cameras on a common clock, triangulating each joint from the
//! rays that agree, holding the result to something a body could actually do,
//! and filtering what comes out.

pub mod align;
pub mod bones;
pub mod fit;
pub mod fuse;
