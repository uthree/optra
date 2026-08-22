//! Architecture adapters.
//!
//! An adapter owns everything between a model's ONNX graph and the pipeline's
//! vocabulary: how an image becomes an input tensor, and how the output tensors
//! become boxes or keypoints. Adding support for a new model family is one file
//! here plus one line in [`build_detector`] or [`build_pose2d`]; adding a new
//! checkpoint of a family already supported is a manifest entry and no code at
//! all.

pub mod mmdet_end2end;
pub mod simcc;

use anyhow::{Result, bail};

use super::session::{self, ProviderChoice, SessionHandle};
use super::traits::{Detector, Pose2d};
use crate::models::manifest::ModelKind;
use crate::models::{ModelSpec, store};

/// Architecture names the manifest may refer to.
pub const KNOWN: [&str; 2] = ["mmdet_end2end", "simcc"];

/// Loads a model and wraps it in the adapter its manifest entry names.
pub fn build_detector(spec: &ModelSpec, provider: ProviderChoice) -> Result<Box<dyn Detector>> {
    if spec.kind != ModelKind::Detector {
        bail!("{} is not a detector", spec.id);
    }
    let handle = load(spec, provider)?;

    match spec.arch.as_str() {
        "mmdet_end2end" => Ok(Box::new(mmdet_end2end::MmdetEnd2End::new(
            spec.clone(),
            handle,
        )?)),
        other => bail!("{} names unknown detector architecture {other}", spec.id),
    }
}

pub fn build_pose2d(spec: &ModelSpec, provider: ProviderChoice) -> Result<Box<dyn Pose2d>> {
    if spec.kind != ModelKind::Pose2d {
        bail!("{} is not a pose model", spec.id);
    }
    let handle = load(spec, provider)?;

    match spec.arch.as_str() {
        "simcc" => Ok(Box::new(simcc::Simcc::new(spec.clone(), handle)?)),
        other => bail!("{} names unknown pose architecture {other}", spec.id),
    }
}

/// Loads the session and checks the graph against what the manifest claims.
///
/// Catching a mismatch here turns a wrong-looking skeleton at runtime into a
/// specific error naming the offending tensor.
fn load(spec: &ModelSpec, provider: ProviderChoice) -> Result<SessionHandle> {
    let path = store::local_path(spec)?;
    if !path.is_file() {
        bail!("{} is not installed", spec.id);
    }

    let handle = session::load(&path, provider)?;

    let inputs: Vec<&str> = handle.session.inputs().iter().map(|i| i.name()).collect();
    if !inputs.contains(&spec.input.name.as_str()) {
        bail!(
            "{} declares input {}, but the graph has {inputs:?}",
            spec.id,
            spec.input.name
        );
    }

    let outputs: Vec<&str> = handle.session.outputs().iter().map(|o| o.name()).collect();
    for declared in &spec.output.tensors {
        if !outputs.contains(&declared.as_str()) {
            bail!(
                "{} declares output {declared}, but the graph has {outputs:?}",
                spec.id
            );
        }
    }

    Ok(handle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Manifest;

    #[test]
    fn every_catalogue_entry_names_a_known_architecture() {
        for spec in Manifest::load().expect("the catalogue") {
            assert!(
                KNOWN.contains(&spec.arch.as_str()),
                "{} names architecture {}, which no adapter provides",
                spec.id,
                spec.arch
            );
        }
    }
}
