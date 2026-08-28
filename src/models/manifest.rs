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
        // The gate exists because Optra downloads and effectively redistributes
        // the catalogue. A file the user already has on their own disk is
        // theirs, is fetched from nowhere, and its licence is between them and
        // wherever they got it — so a local entry passes whatever it says.
        if matches!(self.source, ModelSource::Local { .. }) {
            return Ok(());
        }
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

    /// A ready-to-run spec for an ONNX file the user already has.
    ///
    /// Everything not asked for is filled from the conventions of the export
    /// pipeline behind each supported architecture — the mmdeploy SDK detectors
    /// take BGR letterboxed to grey, the RTMPose SimCC heads take RGB affine
    /// crops normalised to ImageNet — because those are what a checkpoint of
    /// that architecture almost certainly is. An unusual export edits the entry
    /// this writes into the user manifest, which is a far shorter road than
    /// authoring one from a blank file.
    pub fn local(
        kind: ModelKind,
        path: &str,
        name: &str,
        input_name: &str,
        width: u32,
        height: u32,
        keypoints: Option<String>,
    ) -> ModelSpec {
        let id = ModelSpec::slug(name);
        let (arch, input, output, decoder) = match kind {
            ModelKind::Detector => (
                "mmdet_end2end",
                InputSpec {
                    name: input_name.to_owned(),
                    width,
                    height,
                    color: ColorOrder::Bgr,
                    mean: [0.0; 3],
                    std: [1.0; 3],
                    resize: ResizeMode::Letterbox { pad: 114 },
                },
                OutputSpec {
                    tensors: vec!["dets".to_owned(), "labels".to_owned()],
                    keypoints: None,
                },
                BTreeMap::new(),
            ),
            ModelKind::Pose2d => (
                "simcc",
                InputSpec {
                    name: input_name.to_owned(),
                    width,
                    height,
                    color: ColorOrder::Rgb,
                    mean: [123.675, 116.28, 103.53],
                    std: [58.395, 57.12, 57.375],
                    resize: ResizeMode::AffineCrop { padding: 1.25 },
                },
                OutputSpec {
                    tensors: vec!["simcc_x".to_owned(), "simcc_y".to_owned()],
                    keypoints,
                },
                BTreeMap::from([("split_ratio".to_owned(), toml::Value::Float(2.0))]),
            ),
        };

        ModelSpec {
            id,
            name: name.to_owned(),
            kind,
            arch: arch.to_owned(),
            // Honest, and allowed: the licence gate does not apply to a file
            // nothing downloaded. See `check_license`.
            license: "unknown".to_owned(),
            license_url: String::new(),
            zoo: None,
            notes: Some("Registered from a local file.".to_owned()),
            source: ModelSource::Local {
                path: path.to_owned(),
            },
            input,
            output,
            decoder,
        }
    }

    /// The id a model registered under `name` gets: what survives of the name
    /// in lowercase, runs of anything else collapsed to single hyphens.
    pub fn slug(name: &str) -> String {
        let mut id = String::with_capacity(name.len());
        for c in name.chars() {
            if c.is_ascii_alphanumeric() {
                id.push(c.to_ascii_lowercase());
            } else if !id.ends_with('-') && !id.is_empty() {
                id.push('-');
            }
        }
        id.trim_end_matches('-').to_owned()
    }
}

impl Manifest {
    /// Adds `spec` to the user manifest, creating the file if it is absent.
    ///
    /// The entry is appended rather than merged: an id that already exists in
    /// the user manifest is refused here, because from a form the collision is
    /// an accident, and the load-time rule that a user entry replaces a builtin
    /// one is for people editing the file on purpose.
    pub fn register(spec: ModelSpec) -> Result<()> {
        let path = paths::models_dir()?.join("manifest.toml");
        Self::register_at(&path, spec)
    }

    fn register_at(path: &std::path::Path, spec: ModelSpec) -> Result<()> {
        let mut manifest = if path.exists() {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("failed to read {}", path.display()))?;
            Self::parse(&text, &path.display().to_string())?
        } else {
            Manifest {
                manifest_version: MANIFEST_VERSION,
                models: Vec::new(),
            }
        };

        if manifest.models.iter().any(|entry| entry.id == spec.id) {
            bail!(
                "the user manifest already has a model called {}; pick another name",
                spec.id
            );
        }

        let id = spec.id.clone();
        manifest.models.push(spec);
        let text = toml::to_string_pretty(&manifest)?;

        // Written beside the target and renamed, as the room profiles are, so
        // an interrupted write cannot cost the user every model they have
        // registered so far.
        let temporary = path.with_extension("toml.tmp");
        std::fs::write(&temporary, text)
            .with_context(|| format!("failed to write {}", temporary.display()))?;
        std::fs::rename(&temporary, path)
            .with_context(|| format!("failed to replace {}", path.display()))?;

        tracing::info!(model = %id, path = %path.display(), "registered a local model");
        Ok(())
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

    /// The licence gate is about what Optra downloads. A file already on the
    /// user's own disk is theirs, and refusing it over a licence string this
    /// application invented would lock out the one model it never touched.
    #[test]
    fn a_local_model_passes_the_licence_gate_whatever_it_says() {
        let spec = ModelSpec::local(
            ModelKind::Pose2d,
            "C:/models/mine.onnx",
            "My model",
            "input",
            192,
            256,
            Some("halpe26".to_owned()),
        );
        assert_eq!(spec.license, "unknown");
        assert!(spec.check_license().is_ok());
    }

    /// The template's whole promise is that the entry runs without hand
    /// editing, and an arch string no adapter answers to breaks that promise
    /// at session build, three panels away from where it was typed.
    #[test]
    fn local_templates_name_architectures_that_exist() {
        for (kind, layout) in [
            (ModelKind::Detector, None),
            (ModelKind::Pose2d, Some("halpe26".to_owned())),
        ] {
            let spec = ModelSpec::local(kind, "a.onnx", "A model", "input", 192, 256, layout);
            assert!(
                crate::infer::arch::KNOWN.contains(&spec.arch.as_str()),
                "{} is not a known architecture",
                spec.arch
            );
        }
    }

    #[test]
    fn a_name_becomes_a_usable_id() {
        assert_eq!(
            ModelSpec::slug("My RTMPose (fine-tuned) v2!"),
            "my-rtmpose-fine-tuned-v2"
        );
        assert_eq!(ModelSpec::slug("--weird--"), "weird");
    }

    fn scratch_manifest() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU32, Ordering};
        static COUNT: AtomicU32 = AtomicU32::new(0);
        let unique = COUNT.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "optra-manifest-{}-{unique}.toml",
            std::process::id()
        ))
    }

    #[test]
    fn a_registered_model_survives_the_round_trip() {
        let path = scratch_manifest();
        std::fs::remove_file(&path).ok();

        let spec = ModelSpec::local(
            ModelKind::Pose2d,
            "C:/models/mine.onnx",
            "My model",
            "input",
            192,
            256,
            Some("halpe26".to_owned()),
        );
        Manifest::register_at(&path, spec).unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        let read = Manifest::parse(&text, "test").unwrap();
        let entry = &read.models[0];
        assert_eq!(entry.id, "my-model");
        assert_eq!(entry.arch, "simcc");
        assert_eq!(entry.output.keypoints.as_deref(), Some("halpe26"));
        // The decoder settings are the part most easily lost in serialisation,
        // and a SimCC model without its split ratio decodes to garbage.
        assert_eq!(
            entry.decoder.get("split_ratio"),
            Some(&toml::Value::Float(2.0))
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn registering_the_same_name_twice_is_refused() {
        let path = scratch_manifest();
        std::fs::remove_file(&path).ok();

        let spec = ModelSpec::local(
            ModelKind::Detector,
            "a.onnx",
            "Mine",
            "input",
            640,
            640,
            None,
        );
        Manifest::register_at(&path, spec.clone()).unwrap();
        let refused = Manifest::register_at(&path, spec);
        assert!(
            refused.is_err(),
            "a colliding id from a form is an accident"
        );

        std::fs::remove_file(&path).ok();
    }
}
