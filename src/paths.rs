//! Locations Optra reads and writes on disk.
//!
//! Everything lives under a single directory (`%APPDATA%/optra` on Windows) so
//! that a user can back up or wipe their setup by moving one folder.

use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use directories::BaseDirs;

/// Root directory for all Optra state.
pub fn root_dir() -> Result<PathBuf> {
    let base = BaseDirs::new().ok_or_else(|| anyhow!("no home directory for the current user"))?;
    Ok(base.config_dir().join("optra"))
}

/// Application-wide settings file.
pub fn config_file() -> Result<PathBuf> {
    Ok(root_dir()?.join("config.toml"))
}

/// The measured body, which belongs to the person rather than to the room.
pub fn body_file() -> Result<PathBuf> {
    Ok(root_dir()?.join("body.toml"))
}

/// Per-room calibration profiles.
pub fn rooms_dir() -> Result<PathBuf> {
    Ok(root_dir()?.join("rooms"))
}

/// Downloaded ONNX files and the model manifest.
pub fn models_dir() -> Result<PathBuf> {
    Ok(root_dir()?.join("models"))
}

/// Creates every directory Optra expects to exist.
pub fn ensure_dirs() -> Result<()> {
    for dir in [root_dir()?, rooms_dir()?, models_dir()?] {
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("failed to create {}", dir.display()))?;
    }
    Ok(())
}
