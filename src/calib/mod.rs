//! Calibration: turning a recorded walk into a solved room.
//!
//! The geometry lives in [`crate::geometry`]; this is what feeds it. The walk
//! is recorded by [`recorder`], and [`mod@solve`] runs it through resection and
//! joint refinement to produce the cameras.

pub mod latency;
pub mod recorder;
pub mod solve;

pub use latency::{Estimate, LatencyOptions};
pub use recorder::{Recorder, RecorderConfig, Recording, Rig};
pub use solve::{CameraCalibration, RoomCalibration, SolveOptions, solve};
