//! Optra: multi-webcam lower-body tracking for VRChat and SteamVR.
//!
//! The binary is a thin shell around this library so that the pipeline can be
//! exercised by tests without a window.

pub mod app;
pub mod capture;
pub mod config;
pub mod logging;
pub mod paths;
pub mod worker;
