//! Detectors exported by mmdeploy with their postprocessing inside the graph.
//!
//! These models emit boxes that have already been through NMS, which is why
//! there is no decode step here: `dets` is `[batch, n, 5]` of
//! `(x1, y1, x2, y2, score)` in model input pixels, and `labels` is the class
//! of each box.

use anyhow::{Context, Result, bail};
use ort::value::Tensor;

use super::super::preprocess::{self, Mapping};
use super::super::traits::{Detection, Detector, ImageView};
use crate::infer::session::SessionHandle;
use crate::models::ModelSpec;
use crate::models::manifest::ResizeMode;

pub struct MmdetEnd2End {
    handle: SessionHandle,
    spec: ModelSpec,
    /// Boxes below this score are dropped.
    score_threshold: f32,
    /// Class id treated as a person.
    person_class: i64,
}

impl MmdetEnd2End {
    pub fn new(spec: ModelSpec, handle: SessionHandle) -> Result<Self> {
        if spec.output.tensors.len() != 2 {
            bail!(
                "{} should declare two output tensors, got {:?}",
                spec.id,
                spec.output.tensors
            );
        }

        let score_threshold = spec
            .decoder
            .get("score_threshold")
            .and_then(|value| value.as_float())
            .unwrap_or(0.35) as f32;
        let person_class = spec
            .decoder
            .get("person_class")
            .and_then(|value| value.as_integer())
            .unwrap_or(0);

        Ok(Self {
            handle,
            spec,
            score_threshold,
            person_class,
        })
    }

    pub fn backend(&self) -> crate::infer::session::Backend {
        self.handle.backend
    }

    fn prepare(&self, image: &ImageView<'_>) -> Result<(Vec<f32>, Mapping)> {
        let input = match self.spec.input.resize {
            ResizeMode::Letterbox { pad } => preprocess::letterbox(image, &self.spec.input, pad),
            ResizeMode::AffineCrop { .. } => {
                bail!(
                    "{} is a detector and needs a letterbox resize",
                    self.spec.id
                )
            }
        };
        Ok((input.data, input.mapping))
    }
}

impl Detector for MmdetEnd2End {
    fn detect(&mut self, images: &[ImageView<'_>]) -> Result<Vec<Vec<Detection>>> {
        let mut results = Vec::with_capacity(images.len());

        // These graphs are exported with a fixed batch of one, so the cameras
        // are run one after another rather than batched.
        for image in images {
            let (data, mapping) = self.prepare(image)?;
            let shape = vec![
                1i64,
                3,
                self.spec.input.height as i64,
                self.spec.input.width as i64,
            ];

            let tensor = Tensor::from_array((shape, data))
                .map_err(|err| anyhow::anyhow!("failed to build the input tensor: {err}"))?;
            let outputs = self
                .handle
                .session
                .run(ort::inputs![self.spec.input.name.clone() => tensor])
                .map_err(|err| anyhow::anyhow!("detection failed: {err}"))?;

            let (dets_shape, dets) = outputs[self.spec.output.tensors[0].as_str()]
                .try_extract_tensor::<f32>()
                .map_err(|err| anyhow::anyhow!("failed to read the boxes: {err}"))?;
            let (_, labels) = outputs[self.spec.output.tensors[1].as_str()]
                .try_extract_tensor::<i64>()
                .map_err(|err| anyhow::anyhow!("failed to read the labels: {err}"))?;

            let stride = *dets_shape.last().context("the box tensor has no shape")? as usize;
            if stride < 5 {
                bail!("the box tensor has {stride} values per box, expected at least 5");
            }

            let mut found = Vec::new();
            for (index, box_values) in dets.chunks_exact(stride).enumerate() {
                let score = box_values[4];
                if score < self.score_threshold {
                    // mmdeploy pads the output to a fixed length with zero
                    // scores, so the first low score ends the real boxes.
                    break;
                }
                if labels.get(index).copied().unwrap_or(self.person_class) != self.person_class {
                    continue;
                }

                let (x1, y1) = mapping.to_source(box_values[0], box_values[1]);
                let (x2, y2) = mapping.to_source(box_values[2], box_values[3]);
                found.push(Detection {
                    x1: x1.clamp(0.0, image.width as f32),
                    y1: y1.clamp(0.0, image.height as f32),
                    x2: x2.clamp(0.0, image.width as f32),
                    y2: y2.clamp(0.0, image.height as f32),
                    score,
                });
            }

            results.push(found);
        }

        Ok(results)
    }
}
