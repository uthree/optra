//! Model inference.
//!
//! The pipeline only knows the two capabilities in [`traits`]: turn an image
//! into person boxes, and turn a crop into canonical keypoints. Which model
//! architecture provides them is the business of an adapter, and which
//! checkpoint an adapter runs is the business of the model manifest. Swapping a
//! model is therefore a configuration change, not a code change.

pub mod session;

pub use session::{Backend, ProviderChoice, SessionHandle, describe_io};
