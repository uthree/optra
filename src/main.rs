//! Optra: multi-webcam lower-body tracking for VRChat and SteamVR.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use anyhow::{Context, Result};
use optra::{app, config, logging, paths};

fn main() -> Result<()> {
    let log = logging::init();

    paths::ensure_dirs().context("failed to create the Optra data directories")?;
    tracing::info!("data directory: {}", paths::root_dir()?.display());

    let config = config::Config::load_or_default();

    let mut viewport = egui::ViewportBuilder::default()
        .with_title("Optra")
        .with_inner_size(config.window.size)
        .with_min_inner_size([960.0, 600.0])
        .with_maximized(config.window.maximized);
    if let Some(pos) = config.window.pos {
        viewport = viewport.with_position(pos);
    }

    let options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };

    eframe::run_native(
        "Optra",
        options,
        Box::new(move |cc| Ok(Box::new(app::OptraApp::new(cc, config, log)))),
    )
    .map_err(|err| anyhow::anyhow!("failed to start the UI: {err}"))
}
