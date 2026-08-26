//! Locations Optra reads and writes on disk.
//!
//! Everything lives under a single directory (`%APPDATA%/optra` on Windows) so
//! that a user can back up or wipe their setup by moving one folder.

use std::path::{Path, PathBuf};

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

/// The log file and the rolled copies behind it.
pub fn logs_dir() -> Result<PathBuf> {
    Ok(root_dir()?.join("logs"))
}

/// Creates every directory Optra expects to exist.
pub fn ensure_dirs() -> Result<()> {
    for dir in [root_dir()?, rooms_dir()?, models_dir()?, logs_dir()?] {
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("failed to create {}", dir.display()))?;
    }
    Ok(())
}

/// Shows `dir` in the system file manager.
///
/// Asking a user to find `%APPDATA%` for themselves is asking most of them to
/// give up, and the log file is only worth writing if it can be handed over.
pub fn reveal(dir: &Path) -> Result<()> {
    #[cfg(windows)]
    let mut command = {
        // Explorer reports what it did through its window rather than through
        // an exit code, and returns non-zero on success often enough that
        // checking it would produce a false error more often than a true one.
        // Spawning it and not waiting is the whole interaction.
        let mut command = std::process::Command::new("explorer");
        command.arg(dir);
        command
    };

    #[cfg(not(windows))]
    let mut command = {
        let mut command = std::process::Command::new("xdg-open");
        command.arg(dir);
        command
    };

    command
        .spawn()
        .with_context(|| format!("failed to open {}", dir.display()))?;
    Ok(())
}
