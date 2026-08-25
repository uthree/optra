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
    fn backend(&self) -> crate::infer::Backend {
        self.handle.backend
    }

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

            results.push(decode(
                dets,
                labels,
                stride,
                &mapping,
                Bounds {
                    width: image.width as f32,
                    height: image.height as f32,
                },
                self.score_threshold,
                self.person_class,
            ));
        }

        Ok(results)
    }
}

/// The frame a box has to end up inside.
struct Bounds {
    width: f32,
    height: f32,
}

/// Turns one image's box tensor into detections in source-image pixels.
///
/// Pulled out of the session call so it can be tested. Three of the things it
/// does are only visible in a picture and silent otherwise: mmdeploy pads its
/// output to a fixed length rather than reporting a count, the class column
/// decides whether a box is a person or a chair, and the letterbox has to be
/// undone in the right direction.
#[allow(clippy::too_many_arguments)]
fn decode(
    dets: &[f32],
    labels: &[i64],
    stride: usize,
    mapping: &Mapping,
    bounds: Bounds,
    score_threshold: f32,
    person_class: i64,
) -> Vec<Detection> {
    let mut found = Vec::new();

    for (index, box_values) in dets.chunks_exact(stride).enumerate() {
        let score = box_values[4];
        if score < score_threshold {
            // mmdeploy pads the output to a fixed length with zero scores and
            // sorts by score, so the first low one ends the real boxes.
            break;
        }
        if labels.get(index).copied().unwrap_or(person_class) != person_class {
            continue;
        }

        let (x1, y1) = mapping.to_source(box_values[0], box_values[1]);
        let (x2, y2) = mapping.to_source(box_values[2], box_values[3]);
        found.push(Detection {
            x1: x1.clamp(0.0, bounds.width),
            y1: y1.clamp(0.0, bounds.height),
            x2: x2.clamp(0.0, bounds.width),
            y2: y2.clamp(0.0, bounds.height),
            score,
        });
    }

    found
}

#[cfg(test)]
mod tests {
    use super::*;

    const STRIDE: usize = 5;

    fn identity() -> Mapping {
        Mapping {
            scale_x: 1.0,
            scale_y: 1.0,
            offset_x: 0.0,
            offset_y: 0.0,
        }
    }

    fn frame() -> Bounds {
        Bounds {
            width: 640.0,
            height: 480.0,
        }
    }

    fn boxes(rows: &[[f32; STRIDE]]) -> Vec<f32> {
        rows.iter().flatten().copied().collect()
    }

    #[test]
    fn a_confident_person_comes_back_in_source_pixels() {
        let dets = boxes(&[[10.0, 20.0, 110.0, 320.0, 0.9]]);
        let found = decode(&dets, &[0], STRIDE, &identity(), frame(), 0.3, 0);

        assert_eq!(found.len(), 1);
        assert_eq!(
            (found[0].x1, found[0].y1, found[0].x2, found[0].y2),
            (10.0, 20.0, 110.0, 320.0)
        );
        assert_eq!(found[0].score, 0.9);
    }

    /// The output is a fixed-length buffer, sorted by score and padded with
    /// zeros. Reading past the first weak box is reading padding, and a
    /// hundred zero-sized detections at the origin is what comes back.
    #[test]
    fn the_padding_after_the_last_real_box_is_not_read() {
        let dets = boxes(&[
            [10.0, 20.0, 110.0, 320.0, 0.9],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]);
        let found = decode(&dets, &[0, 0, 0], STRIDE, &identity(), frame(), 0.3, 0);
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn something_that_is_not_a_person_is_skipped() {
        let dets = boxes(&[
            [10.0, 20.0, 110.0, 320.0, 0.9],
            [200.0, 30.0, 300.0, 200.0, 0.8],
        ]);
        // The second box is class 56, a chair.
        let found = decode(&dets, &[0, 56], STRIDE, &identity(), frame(), 0.3, 0);

        assert_eq!(found.len(), 1);
        assert_eq!(found[0].x1, 10.0);
    }

    /// A model is run on a letterboxed copy, so every box comes back in the
    /// model's coordinates and has to be brought back. Getting the direction
    /// wrong puts every person in the top left corner.
    #[test]
    fn the_letterbox_is_undone_rather_than_applied_again() {
        let dets = boxes(&[[10.0, 20.0, 110.0, 320.0, 0.9]]);
        let mapping = Mapping {
            scale_x: 2.0,
            scale_y: 2.0,
            offset_x: 0.0,
            offset_y: -40.0,
        };
        let found = decode(&dets, &[0], STRIDE, &mapping, frame(), 0.3, 0);

        assert_eq!(
            (found[0].x1, found[0].y1, found[0].x2, found[0].y2),
            (20.0, 0.0, 220.0, 480.0)
        );
    }

    /// A box that runs past the frame is a person standing at the edge, not a
    /// reason to drop them — but a crop taken from outside the image is not a
    /// crop, so it is held to the frame.
    #[test]
    fn a_box_running_off_the_frame_is_held_to_it() {
        let dets = boxes(&[[-30.0, -10.0, 900.0, 700.0, 0.9]]);
        let found = decode(&dets, &[0], STRIDE, &identity(), frame(), 0.3, 0);

        assert_eq!(
            (found[0].x1, found[0].y1, found[0].x2, found[0].y2),
            (0.0, 0.0, 640.0, 480.0)
        );
    }

    /// A tensor with more columns than the five that are read — some exports
    /// carry the class in the box row as well — must not shift every box by
    /// one column.
    #[test]
    fn extra_columns_do_not_shift_the_boxes() {
        let dets: Vec<f32> = vec![
            10.0, 20.0, 110.0, 320.0, 0.9, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let found = decode(&dets, &[0, 0], 6, &identity(), frame(), 0.3, 0);

        assert_eq!(found.len(), 1);
        assert_eq!((found[0].x1, found[0].y1), (10.0, 20.0));
    }

    #[test]
    fn an_image_with_nobody_in_it_produces_nothing() {
        let dets = boxes(&[[0.0, 0.0, 0.0, 0.0, 0.0]]);
        assert!(decode(&dets, &[0], STRIDE, &identity(), frame(), 0.3, 0).is_empty());
    }
}
