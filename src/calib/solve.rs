//! Turning a recorded walk into solved cameras.
//!
//! Two steps, in the order the design document sets out. Each camera is first
//! resected on its own from the device positions, which ignores the offset
//! between a device and its keypoint and so lands a few centimetres out. Then
//! every camera is refined together with those offsets free, which is what
//! removes the bias.

use std::collections::HashMap;

use anyhow::{Result, bail};
use nalgebra::{Point3, Vector3};
use serde::{Deserialize, Serialize};

use crate::config::{CameraConfig, LensKind};
use crate::geometry::camera::{Camera, Intrinsics};
use crate::geometry::lens::Lens;
use crate::geometry::refine::{RefineOptions, Sighting, refine};
use crate::geometry::resection::{Correspondence, ResectionOptions, resect};
use crate::paths;
use crate::vr::Role;

use super::recorder::{Recording, Rig};

#[derive(Debug, Clone)]
pub struct SolveOptions {
    pub resection: ResectionOptions,
    pub refine: RefineOptions,
    /// Sightings a camera needs before it is worth solving at all.
    pub min_samples: usize,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            resection: ResectionOptions {
                // Loose on purpose. At resection time the offset between each
                // device and its keypoint is still unknown, so every
                // correspondence is wrong by the same handful of centimetres —
                // which at three metres is a couple of degrees. A threshold
                // tight enough to reject that would reject the whole walk.
                inlier_threshold: 0.08,
                ..ResectionOptions::default()
            },
            refine: RefineOptions::default(),
            min_samples: 40,
        }
    }
}

/// What a camera came out as.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraCalibration {
    pub id: String,
    pub camera: Camera,
    /// RMS angular reprojection error after refinement, in radians.
    pub rms: f64,
    /// Sightings that survived to the final fit.
    pub sightings: usize,
    /// Fraction of the frame the walk covered.
    pub coverage: f32,
    /// How far the correspondences were from lying in a plane when this camera
    /// was resected. Near zero and the answer rests on nothing.
    pub spread: f64,
}

impl CameraCalibration {
    /// The reprojection error in degrees, which is the number to put in front
    /// of a user.
    pub fn rms_degrees(&self) -> f64 {
        self.rms.to_degrees()
    }
}

/// A solved room.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomCalibration {
    pub cameras: Vec<CameraCalibration>,
    /// The device-to-keypoint offsets that were solved for, one per rig.
    pub rigs: Vec<(Rig, Vector3<f64>)>,
    /// RMS angular reprojection error over every camera, in radians.
    pub rms: f64,
    /// Sightings discarded as outliers before the final fit.
    pub rejected: usize,
    /// Sightings the final fit used.
    pub used: usize,
    pub solved_at: String,
}

impl RoomCalibration {
    pub fn rms_degrees(&self) -> f64 {
        self.rms.to_degrees()
    }

    pub fn camera(&self, id: &str) -> Option<&CameraCalibration> {
        self.cameras.iter().find(|camera| camera.id == id)
    }

    /// Writes the profile under the rooms directory.
    pub fn save(&self, name: &str) -> Result<()> {
        let path = paths::rooms_dir()?.join(format!("{name}.toml"));
        let text = toml::to_string_pretty(self)?;

        // Written beside the target and renamed, so an interrupted write
        // cannot leave a half-finished profile where a good one was.
        let temporary = path.with_extension("toml.tmp");
        std::fs::write(&temporary, text)?;
        std::fs::rename(&temporary, &path)?;

        tracing::info!(path = %path.display(), "saved a room profile");
        Ok(())
    }

    pub fn load(name: &str) -> Result<Self> {
        let path = paths::rooms_dir()?.join(format!("{name}.toml"));
        Ok(toml::from_str(&std::fs::read_to_string(path)?)?)
    }

    /// Profiles saved on this machine, for the UI to offer.
    pub fn list() -> Vec<String> {
        let Ok(directory) = paths::rooms_dir() else {
            return Vec::new();
        };
        let Ok(entries) = std::fs::read_dir(directory) else {
            return Vec::new();
        };

        let mut names: Vec<String> = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .is_some_and(|extension| extension == "toml")
            })
            .filter_map(|path| Some(path.file_stem()?.to_string_lossy().into_owned()))
            .collect();
        names.sort();
        names
    }
}

/// Solves every camera in the recording.
pub fn solve(
    recording: &Recording,
    cameras: &[CameraConfig],
    options: &SolveOptions,
) -> Result<RoomCalibration> {
    if recording.rigs.is_empty() {
        bail!("the recording has no tracked devices in it");
    }

    let configs: HashMap<&str, &CameraConfig> =
        cameras.iter().map(|c| (c.id.as_str(), c)).collect();

    let mut ids = Vec::new();
    let mut seeds = Vec::new();
    let mut spreads = Vec::new();
    let mut coverages = Vec::new();

    for trail in &recording.cameras {
        if trail.samples.len() < options.min_samples {
            tracing::warn!(
                camera = %trail.camera,
                samples = trail.samples.len(),
                "not enough of the walk was visible; skipping this camera"
            );
            continue;
        }

        let Some(config) = configs.get(trail.camera.as_str()) else {
            bail!(
                "the recording names a camera that is not configured: {}",
                trail.camera
            );
        };

        // Every rig contributes, not just the head. The head alone traces a
        // narrow band of heights, and a set of points that close to a plane is
        // degenerate for the linear solve; a hand raised and lowered is what
        // gives the resection something to work with.
        let correspondences: Vec<Correspondence> = trail
            .samples
            .iter()
            .filter_map(|sample| {
                let pose = recording.tracks[sample.rig].at(sample.at)?;
                Some(Correspondence {
                    world: Point3::from(pose.translation.vector),
                    pixel: sample.pixel,
                })
            })
            .collect();

        let lens = Lens::for_kind(config.lens);
        let guess = Intrinsics::from_fov(
            trail.width,
            trail.height,
            seed_fov(config.lens).to_radians(),
        );

        let Some(resection) = resect(&guess, lens, &correspondences, &options.resection) else {
            bail!(
                "camera {} could not be solved from {} correspondences",
                trail.camera,
                correspondences.len()
            );
        };

        ids.push(trail.camera.clone());
        spreads.push(resection.spread);
        coverages.push(trail.coverage.filled());
        seeds.push(resection.camera);
    }

    if seeds.len() < 2 {
        bail!(
            "only {} camera(s) saw enough of the walk; tracking needs at least two",
            seeds.len()
        );
    }

    let index: HashMap<&str, usize> = ids
        .iter()
        .enumerate()
        .map(|(index, id)| (id.as_str(), index))
        .collect();

    let mut sightings = Vec::new();
    for trail in &recording.cameras {
        let Some(camera) = index.get(trail.camera.as_str()).copied() else {
            continue;
        };

        for sample in &trail.samples {
            let Some(anchor) = recording.tracks[sample.rig].at(sample.at) else {
                continue;
            };
            sightings.push(Sighting {
                camera,
                rig: sample.rig,
                anchor,
                pixel: sample.pixel,
                weight: weight_of(recording.rigs[sample.rig], sample.confidence),
            });
        }
    }

    let offsets = vec![Vector3::zeros(); recording.rigs.len()];
    let refined = refine(&seeds, &offsets, &sightings, &options.refine);

    let cameras = ids
        .into_iter()
        .enumerate()
        .map(|(slot, id)| CameraCalibration {
            id,
            camera: refined.cameras[slot].clone(),
            rms: refined.per_camera[slot].rms,
            sightings: refined.per_camera[slot].sightings,
            coverage: coverages[slot],
            spread: spreads[slot],
        })
        .collect();

    Ok(RoomCalibration {
        cameras,
        rigs: recording
            .rigs
            .iter()
            .copied()
            .zip(refined.offsets.iter().copied())
            .collect(),
        rms: refined.rms,
        rejected: refined.rejected,
        used: sightings.len() - refined.rejected,
        solved_at: chrono::Local::now().to_rfc3339(),
    })
}

/// A starting field of view for a lens kind, in degrees.
///
/// Only the order of magnitude matters: the solve recovers the real focal
/// length, and needs no better than a factor of two to get there.
fn seed_fov(kind: LensKind) -> f64 {
    match kind {
        LensKind::Standard => 70.0,
        LensKind::Wide => 100.0,
        LensKind::Fisheye => 150.0,
    }
}

/// How much a sighting is trusted, before the solver's own outlier handling.
///
/// A controller is held rather than worn: the wrist keypoint sits against it
/// less rigidly than the head sits against a headset, and a change of grip
/// moves the offset by a centimetre or two. They earn their place by reaching
/// heights the head never does, but they should not outvote it.
fn weight_of(rig: Rig, confidence: f64) -> f64 {
    let rigidity = if rig.role == Role::Head { 1.0 } else { 0.5 };
    confidence.clamp(0.0, 1.0) * rigidity
}
