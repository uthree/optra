//! Model inference.
//!
//! The pipeline only knows the two capabilities in [`traits`]: turn an image
//! into person boxes, and turn a person's box into canonical keypoints. Which
//! model architecture provides them is the business of an adapter in [`arch`],
//! and which checkpoint an adapter runs is the business of the model manifest.
//! Swapping a model is therefore a configuration change, not a code change.

pub mod arch;
pub mod preprocess;
pub mod session;
pub mod traits;

pub use session::{Backend, ProviderChoice, SessionHandle};
pub use traits::{Detection, Detector, ImageView, Keypoint, Keypoints2d, Pose2d};
