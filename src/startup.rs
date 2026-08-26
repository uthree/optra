//! What Optra checks about itself before anyone asks it to track.
//!
//! Everything tracking needs is arranged over four panels, and each of them
//! reports its own part well enough on its own. What none of them can say is
//! whether the whole thing is ready — so the failure a user actually meets is
//! not an error message but a Tracking panel that sits there doing nothing,
//! with the reason one panel away and no indication which one.
//!
//! Nothing here talks to hardware beyond enumerating it. The question is
//! whether what was configured is still there, which is exactly the question a
//! user cannot answer for themselves: a camera identified by a Media Foundation
//! symbolic link is not something anybody recognises as the one they unplugged.

use crate::calib::RoomCalibration;
use crate::config::{Config, SourceConfig};
use crate::models::manifest::{Manifest, ModelSpec};
use crate::models::store;

/// Views a joint has to be seen from before it can be placed at all. One camera
/// gives a ray and no distance along it.
const MIN_CAMERAS: usize = 2;

/// How a single check came out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Verdict {
    /// Nothing to do.
    Ready,
    /// Tracking will run, and something about it is worse than it should be.
    Warning,
    /// Tracking cannot run until this is dealt with.
    Blocked,
}

impl Verdict {
    pub fn label(self) -> &'static str {
        match self {
            Verdict::Ready => "Ready",
            Verdict::Warning => "Check",
            Verdict::Blocked => "Not ready",
        }
    }
}

/// One question, its answer, and what to do about the answer.
#[derive(Debug, Clone)]
pub struct Check {
    pub title: &'static str,
    pub verdict: Verdict,
    pub detail: String,
    /// Where the user goes to fix it. Absent when there is nothing to fix.
    pub fix: Option<&'static str>,
}

impl Check {
    fn ready(title: &'static str, detail: impl Into<String>) -> Self {
        Self {
            title,
            verdict: Verdict::Ready,
            detail: detail.into(),
            fix: None,
        }
    }

    fn warning(title: &'static str, detail: impl Into<String>, fix: &'static str) -> Self {
        Self {
            title,
            verdict: Verdict::Warning,
            detail: detail.into(),
            fix: Some(fix),
        }
    }

    fn blocked(title: &'static str, detail: impl Into<String>, fix: &'static str) -> Self {
        Self {
            title,
            verdict: Verdict::Blocked,
            detail: detail.into(),
            fix: Some(fix),
        }
    }
}

/// Every check, in the order they have to be dealt with.
#[derive(Debug, Clone, Default)]
pub struct Report {
    pub checks: Vec<Check>,
}

impl Report {
    /// Runs every check against the machine as it is now.
    pub fn gather(config: &Config, room: Option<&RoomCalibration>) -> Self {
        let devices = present_devices();
        let catalogue = Manifest::load().ok();

        Self {
            checks: vec![
                cameras(config, devices.as_deref()),
                models(config, catalogue.as_deref(), &store::is_installed),
                room_profile(config, room),
            ],
        }
    }

    /// The worst verdict in the report, which is the one the user has to act on.
    pub fn verdict(&self) -> Verdict {
        self.checks
            .iter()
            .map(|check| check.verdict)
            .max()
            .unwrap_or(Verdict::Ready)
    }

    pub fn is_clear(&self) -> bool {
        self.verdict() == Verdict::Ready
    }

    /// Checks worth showing, which is everything that is not `Ready`.
    pub fn problems(&self) -> impl Iterator<Item = &Check> {
        self.checks
            .iter()
            .filter(|check| check.verdict != Verdict::Ready)
    }

    /// Writes the report to the log, which is where it can be read back after
    /// the user has closed whatever showed it to them.
    pub fn log(&self) {
        for check in &self.checks {
            match check.verdict {
                Verdict::Ready => tracing::info!(check = check.title, "{}", check.detail),
                Verdict::Warning => tracing::warn!(check = check.title, "{}", check.detail),
                Verdict::Blocked => {
                    tracing::warn!(check = check.title, "not ready: {}", check.detail)
                }
            }
        }
    }
}

/// Device paths of the capture devices attached right now, or `None` when the
/// platform cannot be asked.
fn present_devices() -> Option<Vec<String>> {
    #[cfg(windows)]
    {
        match crate::capture::source::webcam::list_devices() {
            Ok(devices) => Some(devices.into_iter().map(|info| info.misc()).collect()),
            Err(error) => {
                tracing::warn!("could not enumerate the capture devices: {error:#}");
                None
            }
        }
    }
    #[cfg(not(windows))]
    {
        None
    }
}

/// Are the cameras that were configured still attached?
///
/// `present` is the set of device paths currently enumerated, or `None` when
/// the platform could not be asked — which is not the same as an empty list and
/// must not be reported as every camera being gone.
pub fn cameras(config: &Config, present: Option<&[String]>) -> Check {
    const TITLE: &str = "Cameras";

    let enabled: Vec<_> = config.cameras.iter().filter(|c| c.enabled).collect();
    if enabled.is_empty() {
        return Check::blocked(
            TITLE,
            "no cameras are enabled",
            "Add them in the Cameras panel.",
        );
    }

    let missing: Vec<&str> = match present {
        Some(present) => enabled
            .iter()
            .filter_map(|camera| match &camera.source {
                SourceConfig::Webcam {
                    device_path,
                    device_name,
                } => (!present.iter().any(|path| path == device_path)).then_some(
                    // The name, because the path is a symbolic link nobody
                    // recognises as the camera they unplugged.
                    if device_name.is_empty() {
                        camera.label.as_str()
                    } else {
                        device_name.as_str()
                    },
                ),
                // Nothing that can be unplugged.
                SourceConfig::Synthetic { .. } | SourceConfig::Still { .. } => None,
            })
            .collect(),
        None => Vec::new(),
    };

    let usable = enabled.len() - missing.len();

    if usable < MIN_CAMERAS {
        let detail = if missing.is_empty() {
            format!("{usable} camera(s) enabled, and placing a joint needs {MIN_CAMERAS}")
        } else {
            format!(
                "{} of {} cameras are not attached: {}",
                missing.len(),
                enabled.len(),
                missing.join(", ")
            )
        };
        return Check::blocked(
            TITLE,
            detail,
            "Reconnect them, or enable more in the Cameras panel.",
        );
    }

    if !missing.is_empty() {
        // The room was solved with these cameras in it. Tracking continues
        // without them, from fewer views and so less well, and the calibration
        // does not have to be redone as long as nothing else moved.
        return Check::warning(
            TITLE,
            format!(
                "not attached: {}. The remaining {usable} will be used.",
                missing.join(", ")
            ),
            "Reconnect them if the tracking is worse than it was.",
        );
    }

    if present.is_none() {
        return Check::warning(
            TITLE,
            format!("{usable} camera(s) enabled; whether they are attached could not be checked"),
            "Open the Cameras panel to see what was found.",
        );
    }

    Check::ready(TITLE, format!("{usable} camera(s) enabled and attached"))
}

/// Is every model these cameras are set to use actually on disk?
pub fn models(
    config: &Config,
    catalogue: Option<&[ModelSpec]>,
    installed: &dyn Fn(&ModelSpec) -> bool,
) -> Check {
    const TITLE: &str = "Models";

    if !config.inference.enabled {
        return Check::warning(
            TITLE,
            "inference is switched off, so no keypoints are produced",
            "Turn it back on in the Models panel.",
        );
    }

    let Some(catalogue) = catalogue else {
        return Check::blocked(
            TITLE,
            "the model catalogue could not be read",
            "The Models panel reports why.",
        );
    };

    // Each camera may name its own and otherwise falls back to the shared
    // default. A model no camera uses is not this check's business, however
    // absent it is.
    let mut wanted: Vec<&str> = Vec::new();
    for camera in config.cameras.iter().filter(|c| c.enabled) {
        for id in [
            camera
                .detector_model
                .as_deref()
                .unwrap_or(&config.inference.detector_model),
            camera
                .pose_model
                .as_deref()
                .unwrap_or(&config.inference.pose_model),
        ] {
            if !wanted.contains(&id) {
                wanted.push(id);
            }
        }
    }

    if wanted.is_empty() {
        return Check::ready(TITLE, "no camera is running a model");
    }

    let mut unknown = Vec::new();
    let mut absent = Vec::new();
    for id in &wanted {
        match catalogue.iter().find(|spec| spec.id == *id) {
            Some(spec) if installed(spec) => {}
            Some(_) => absent.push(*id),
            None => unknown.push(*id),
        }
    }

    if !unknown.is_empty() {
        return Check::blocked(
            TITLE,
            format!("the catalogue has no model called {}", unknown.join(" or ")),
            "Pick a model for those cameras in the Models panel.",
        );
    }

    if !absent.is_empty() {
        return Check::blocked(
            TITLE,
            format!("not downloaded yet: {}", absent.join(", ")),
            "Install them in the Models panel.",
        );
    }

    Check::ready(TITLE, format!("{} model(s) ready", wanted.len()))
}

/// Is there a calibration, and does it describe the cameras that are running?
pub fn room_profile(config: &Config, room: Option<&RoomCalibration>) -> Check {
    const TITLE: &str = "Room profile";

    let Some(name) = config.room.as_deref() else {
        return Check::blocked(
            TITLE,
            "no room profile is selected, so nothing knows where the cameras are",
            "Run the calibration wizard in the Calibration panel.",
        );
    };

    let Some(room) = room else {
        // The name is kept rather than cleared: a profile can be missing
        // because a directory was moved, and forgetting which room this was
        // would turn a restore into a recalibration.
        return Check::blocked(
            TITLE,
            format!("the room profile '{name}' could not be loaded"),
            "Pick another profile, or calibrate again, in the Calibration panel.",
        );
    };

    let solved: Vec<&str> = room.cameras.iter().map(|c| c.id.as_str()).collect();
    let running = config
        .cameras
        .iter()
        .filter(|camera| camera.enabled && solved.contains(&camera.id.as_str()))
        .count();

    if running < MIN_CAMERAS {
        return Check::blocked(
            TITLE,
            format!(
                "'{name}' was solved for {} camera(s), and only {running} of them are enabled",
                solved.len()
            ),
            "Enable them in the Cameras panel, or calibrate this set of cameras.",
        );
    }

    // A camera the profile has never heard of contributes nothing: there is no
    // way to know where it is looking from. Worth saying out loud, because from
    // the Cameras panel it looks like it is working — it streams, it finds a
    // person, and none of that reaches the reconstruction.
    let uncalibrated: Vec<&str> = config
        .cameras
        .iter()
        .filter(|camera| camera.enabled && !solved.contains(&camera.id.as_str()))
        .map(|camera| camera.label.as_str())
        .collect();

    if !uncalibrated.is_empty() {
        return Check::warning(
            TITLE,
            format!(
                "'{name}' does not include {}, so nothing they see is used",
                uncalibrated.join(", ")
            ),
            "Calibrate again with every camera in the room.",
        );
    }

    Check::ready(
        TITLE,
        format!("'{name}', solved for {} camera(s)", solved.len()),
    )
}
