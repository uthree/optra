//! Camera geometry: lens models, projection, calibration and triangulation.
//!
//! Everything here works in the world frame described in the design document:
//! right-handed, +Y up, metres, matching OpenVR's standing space. Camera axes
//! follow the OpenCV convention, +x right, +y down, +z into the scene.
//!
//! Errors and weights are angular rather than pixel-based throughout, because
//! the cameras in a room are not assumed to match: a pixel on one is not a
//! pixel on another.

pub mod camera;
pub mod lens;
pub mod refine;
pub mod resection;
pub mod triangulate;

pub use camera::{Camera, Intrinsics};
pub use lens::Lens;
pub use refine::{RefineOptions, Refinement, Sighting, refine};
pub use resection::{Correspondence, Resection, ResectionOptions, resect};
pub use triangulate::{Observation, Triangulation, triangulate};
