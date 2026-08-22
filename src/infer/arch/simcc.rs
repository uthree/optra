//! RTMPose and other models using SimCC coordinate classification.
//!
//! Instead of a heatmap, these emit two 1D distributions per keypoint, one over
//! x and one over y, at `split_ratio` times the input resolution. Decoding is
//! an argmax on each axis, which is both cheaper and more precise than reading
//! a heatmap peak.

use anyhow::{Result, bail};
use ort::value::Tensor;

use super::super::preprocess::{self, Mapping};
use super::super::traits::{Detection, ImageView, Keypoint, Keypoints2d, Pose2d};
use crate::infer::session::SessionHandle;
use crate::models::manifest::ResizeMode;
use crate::models::{Layout, ModelSpec, keypoints};

pub struct Simcc {
    handle: SessionHandle,
    spec: ModelSpec,
    layout: &'static Layout,
    split_ratio: f32,
    /// Keypoints scoring below this are treated as not seen.
    confidence_threshold: f32,
}

impl Simcc {
    pub fn new(spec: ModelSpec, handle: SessionHandle) -> Result<Self> {
        if spec.output.tensors.len() != 2 {
            bail!(
                "{} should declare two output tensors, got {:?}",
                spec.id,
                spec.output.tensors
            );
        }

        let name = spec
            .output
            .keypoints
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("{} declares no keypoint layout", spec.id))?;
        let layout = keypoints::layout(name)
            .ok_or_else(|| anyhow::anyhow!("{} refers to unknown layout {name}", spec.id))?;

        let split_ratio = spec
            .decoder
            .get("split_ratio")
            .and_then(|value| value.as_float())
            .unwrap_or(2.0) as f32;
        let confidence_threshold = spec
            .decoder
            .get("confidence_threshold")
            .and_then(|value| value.as_float())
            .unwrap_or(0.3) as f32;

        Ok(Self {
            handle,
            spec,
            layout,
            split_ratio,
            confidence_threshold,
        })
    }

    pub fn backend(&self) -> crate::infer::session::Backend {
        self.handle.backend
    }

    pub fn layout(&self) -> &'static Layout {
        self.layout
    }

    fn prepare(&self, image: &ImageView<'_>, person: &Detection) -> Result<(Vec<f32>, Mapping)> {
        let input = match self.spec.input.resize {
            ResizeMode::AffineCrop { padding } => {
                preprocess::affine_crop(image, person, &self.spec.input, padding)
            }
            ResizeMode::Letterbox { .. } => {
                bail!("{} is a pose model and needs an affine crop", self.spec.id)
            }
        };
        Ok((input.data, input.mapping))
    }
}

impl Pose2d for Simcc {
    fn estimate(&mut self, people: &[(ImageView<'_>, Detection)]) -> Result<Vec<Keypoints2d>> {
        if people.is_empty() {
            return Ok(Vec::new());
        }

        // The graph takes a dynamic batch, so every camera's crop goes through
        // in one call.
        let (width, height) = (
            self.spec.input.width as usize,
            self.spec.input.height as usize,
        );
        let mut batch = Vec::with_capacity(people.len() * 3 * width * height);
        let mut mappings = Vec::with_capacity(people.len());

        for (image, person) in people {
            let (data, mapping) = self.prepare(image, person)?;
            batch.extend_from_slice(&data);
            mappings.push(mapping);
        }

        let shape = vec![people.len() as i64, 3, height as i64, width as i64];
        let tensor = Tensor::from_array((shape, batch))
            .map_err(|err| anyhow::anyhow!("failed to build the input tensor: {err}"))?;

        let outputs = self
            .handle
            .session
            .run(ort::inputs![self.spec.input.name.clone() => tensor])
            .map_err(|err| anyhow::anyhow!("pose estimation failed: {err}"))?;

        let (x_shape, simcc_x) = outputs[self.spec.output.tensors[0].as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|err| anyhow::anyhow!("failed to read simcc x: {err}"))?;
        let (y_shape, simcc_y) = outputs[self.spec.output.tensors[1].as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|err| anyhow::anyhow!("failed to read simcc y: {err}"))?;

        if x_shape.len() != 3 || y_shape.len() != 3 {
            bail!("expected 3D simcc tensors, got {x_shape:?} and {y_shape:?}");
        }

        let keypoint_count = x_shape[1] as usize;
        let x_bins = x_shape[2] as usize;
        let y_bins = y_shape[2] as usize;

        if keypoint_count != self.layout.count {
            bail!(
                "{} produced {keypoint_count} keypoints, but its layout declares {}",
                self.spec.id,
                self.layout.count
            );
        }

        let mut results = Vec::with_capacity(people.len());
        for (person_index, mapping) in mappings.iter().enumerate() {
            let mut keypoints = Keypoints2d::default();

            for (joint, source_index) in &self.layout.joints {
                let x_offset = (person_index * keypoint_count + source_index) * x_bins;
                let y_offset = (person_index * keypoint_count + source_index) * y_bins;

                let (x_bin, x_score) = argmax(&simcc_x[x_offset..x_offset + x_bins]);
                let (y_bin, y_score) = argmax(&simcc_y[y_offset..y_offset + y_bins]);

                // The weaker axis decides: a keypoint the model is sure about
                // horizontally and lost about vertically is not a keypoint.
                let confidence = x_score.min(y_score);
                if confidence < self.confidence_threshold {
                    continue;
                }

                let (x, y) = mapping.to_source(
                    x_bin as f32 / self.split_ratio,
                    y_bin as f32 / self.split_ratio,
                );
                keypoints.set(*joint, Keypoint { x, y, confidence });
            }

            results.push(keypoints);
        }

        Ok(results)
    }
}

/// Index and value of the largest element.
fn argmax(values: &[f32]) -> (usize, f32) {
    values.iter().enumerate().fold(
        (0, f32::NEG_INFINITY),
        |(best_index, best), (index, value)| {
            if *value > best {
                (index, *value)
            } else {
                (best_index, best)
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_finds_the_peak() {
        assert_eq!(argmax(&[0.1, 0.9, 0.4]), (1, 0.9));
        assert_eq!(argmax(&[-3.0, -1.0, -2.0]), (1, -1.0));
    }

    #[test]
    fn argmax_keeps_the_first_of_equal_peaks() {
        assert_eq!(argmax(&[1.0, 1.0]).0, 0);
    }
}
