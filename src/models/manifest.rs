//! Model specifications.
//!
//! Everything that varies between checkpoints of the same architecture lives
//! here rather than in code: where to fetch it, how to preprocess for it, which
//! tensors it produces and how to read them. Adding a new checkpoint, a new
//! input resolution or a new quantization of a supported architecture is a
//! manifest entry and nothing else.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::paths;

/// Manifest format version, bumped when the schema changes incompatibly.
pub const MANIFEST_VERSION: u32 = 1;

/// The catalogue Optra ships with.
const BUILTIN: &str = include_str!("builtin.toml");

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Manifest {
    pub manifest_version: u32,
    #[serde(default, rename = "model")]
    pub models: Vec<ModelSpec>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelSpec {
    /// Stable identifier, also the local file name.
    pub id: String,
    /// Name shown in the UI.
    pub name: String,
    pub kind: ModelKind,
    /// Which architecture adapter runs this model.
    pub arch: String,
    /// SPDX identifier of the upstream license.
    pub license: String,
    pub license_url: String,
    /// Where the equivalent model sits in PINTO's model zoo, when it does.
    #[serde(default)]
    pub zoo: Option<String>,
    /// One line on what this model is good for.
    #[serde(default)]
    pub notes: Option<String>,
    pub source: ModelSource,
    pub input: InputSpec,
    pub output: OutputSpec,
    /// Adapter-specific settings, interpreted by the architecture adapter.
    #[serde(default)]
    pub decoder: BTreeMap<String, toml::Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    /// Produces person bounding boxes.
    Detector,
    /// Produces keypoints for one cropped person.
    Pose2d,
}

impl ModelKind {
    pub fn label(self) -> &'static str {
        match self {
            ModelKind::Detector => "Detector",
            ModelKind::Pose2d => "2D pose",
        }
    }
}

/// Where a model comes from and how to unpack it.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelSource {
    /// A bare `.onnx` file.
    File {
        url: String,
        sha256: String,
        /// Expected download size, so the UI can show progress and warn before
        /// a large fetch.
        size: u64,
    },
    /// A zip archive containing exactly one `.onnx`, or the one named by
    /// `entry`.
    Zip {
        url: String,
        sha256: String,
        size: u64,
        #[serde(default)]
        entry: Option<String>,
    },
    /// A gzipped tar containing exactly one `.onnx`, or the one named by
    /// `entry`. This is the shape PINTO's model zoo publishes.
    TarGz {
        url: String,
        sha256: String,
        size: u64,
        #[serde(default)]
        entry: Option<String>,
    },
    /// A file the user already has. Nothing is downloaded or verified.
    Local { path: String },
}

impl ModelSource {
    pub fn url(&self) -> Option<&str> {
        match self {
            ModelSource::File { url, .. }
            | ModelSource::Zip { url, .. }
            | ModelSource::TarGz { url, .. } => Some(url),
            ModelSource::Local { .. } => None,
        }
    }

    /// Download size in bytes, where it is known ahead of time.
    pub fn size(&self) -> Option<u64> {
        match self {
            ModelSource::File { size, .. }
            | ModelSource::Zip { size, .. }
            | ModelSource::TarGz { size, .. } => Some(*size),
            ModelSource::Local { .. } => None,
        }
    }

    pub fn sha256(&self) -> Option<&str> {
        match self {
            ModelSource::File { sha256, .. }
            | ModelSource::Zip { sha256, .. }
            | ModelSource::TarGz { sha256, .. } => Some(sha256),
            ModelSource::Local { .. } => None,
        }
    }
}

/// How to turn an image into the tensor a model expects.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InputSpec {
    /// Name of the input tensor.
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// Channel order the model was trained on.
    #[serde(default)]
    pub color: ColorOrder,
    /// Per-channel mean subtracted before scaling, in the model's channel order.
    #[serde(default)]
    pub mean: [f32; 3],
    /// Per-channel divisor applied after the mean.
    #[serde(default = "unit_std")]
    pub std: [f32; 3],
    pub resize: ResizeMode,
}

fn unit_std() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ColorOrder {
    #[default]
    Rgb,
    Bgr,
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResizeMode {
    /// Preserve aspect ratio, pad the remainder. Used by detectors.
    Letterbox {
        /// Value the padding is filled with, in 0-255.
        pad: u8,
    },
    /// Warp the person's box to the model's aspect ratio with a margin. Used by
    /// top-down pose models.
    AffineCrop {
        /// How much larger than the box the crop is, e.g. 1.25.
        padding: f32,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OutputSpec {
    /// Output tensor names, in the order the adapter expects them.
    pub tensors: Vec<String>,
    /// Which keypoint layout the model produces, for `pose2d` models.
    #[serde(default)]
    pub keypoints: Option<String>,
}

impl Manifest {
    /// The builtin catalogue plus any user manifest in the models directory.
    ///
    /// A user entry with the same id as a builtin one replaces it, which is how
    /// a locally converted model can stand in for a downloaded one.
    pub fn load() -> Result<Vec<ModelSpec>> {
        let mut models = Self::parse(BUILTIN, "the builtin catalogue")?.models;

        let user_path = paths::models_dir()?.join("manifest.toml");
        if user_path.exists() {
            let text = std::fs::read_to_string(&user_path)
                .with_context(|| format!("failed to read {}", user_path.display()))?;
            let user = Self::parse(&text, &user_path.display().to_string())?;
            for spec in user.models {
                match models.iter().position(|m| m.id == spec.id) {
                    Some(index) => models[index] = spec,
                    None => models.push(spec),
                }
            }
        }

        models.retain(|spec| match spec.check_license() {
            Ok(()) => true,
            Err(err) => {
                tracing::warn!("ignoring model {}: {err:#}", spec.id);
                false
            }
        });

        Ok(models)
    }

    fn parse(text: &str, origin: &str) -> Result<Self> {
        let manifest: Manifest =
            toml::from_str(text).with_context(|| format!("failed to parse {origin}"))?;

        if manifest.manifest_version != MANIFEST_VERSION {
            bail!(
                "{origin} declares manifest_version {}, but this build understands {MANIFEST_VERSION}",
                manifest.manifest_version
            );
        }
        Ok(manifest)
    }
}

/// Licenses Optra is willing to run.
///
/// The restriction is deliberate: Optra is Apache-2.0, and a copyleft model
/// would put anyone redistributing a configured setup in an awkward position.
const ALLOWED_LICENSES: [&str; 2] = ["Apache-2.0", "MIT"];

impl ModelSpec {
    fn check_license(&self) -> Result<()> {
        if !ALLOWED_LICENSES.contains(&self.license.as_str()) {
            bail!(
                "its license is {}, and only {} are accepted",
                self.license,
                ALLOWED_LICENSES.join(" and ")
            );
        }
        Ok(())
    }

    /// Number of keypoints the model produces, where the layout is known.
    pub fn keypoint_count(&self) -> Option<usize> {
        self.output
            .keypoints
            .as_deref()
            .and_then(crate::models::keypoints::layout)
            .map(|layout| layout.count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_builtin_catalogue_parses() {
        let models = Manifest::parse(BUILTIN, "builtin")
            .expect("builtin manifest")
            .models;
        assert!(!models.is_empty());

        for spec in &models {
            assert!(
                spec.check_license().is_ok(),
                "{} has a bad license",
                spec.id
            );
            assert!(
                !spec.output.tensors.is_empty(),
                "{} declares no output tensors",
                spec.id
            );
            if spec.kind == ModelKind::Pose2d {
                assert!(
                    spec.output.keypoints.is_some(),
                    "{} is a pose model with no keypoint layout",
                    spec.id
                );
            }
        }
    }

    #[test]
    fn model_ids_are_unique() {
        let models = Manifest::parse(BUILTIN, "builtin")
            .expect("builtin manifest")
            .models;
        let mut ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
        ids.sort_unstable();
        let before = ids.len();
        ids.dedup();
        assert_eq!(before, ids.len(), "duplicate model ids in the catalogue");
    }

    #[test]
    fn a_copyleft_model_is_rejected() {
        let spec = ModelSpec {
            license: "GPL-3.0".to_owned(),
            ..Manifest::parse(BUILTIN, "builtin")
                .unwrap()
                .models
                .remove(0)
        };
        assert!(spec.check_license().is_err());
    }
}
