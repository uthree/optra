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
    fn backend(&self) -> crate::infer::Backend {
        self.handle.backend
    }

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

        Ok(decode(
            self.layout,
            &mappings,
            Axis {
                scores: simcc_x,
                bins: x_bins,
            },
            Axis {
                scores: simcc_y,
                bins: y_bins,
            },
            self.split_ratio,
            self.confidence_threshold,
        ))
    }
}

/// One axis of a SimCC output: every keypoint's bin scores, laid end to end,
/// for every person in the batch.
struct Axis<'a> {
    scores: &'a [f32],
    /// Bins per keypoint. The two axes may differ, and do whenever the model's
    /// input is not square.
    bins: usize,
}

impl Axis<'_> {
    /// One keypoint's bins, for one person in the batch.
    fn slice(&self, person: usize, keypoints: usize, index: usize) -> &[f32] {
        let start = (person * keypoints + index) * self.bins;
        &self.scores[start..start + self.bins]
    }
}

/// Turns the two SimCC axes into keypoints in source-image pixels.
///
/// A free function rather than a method, and the argument list is the price:
/// everything it does is a place to be quietly wrong — the model's own keypoint
/// order against the canonical one, the bin-to-pixel scale, which axis is which,
/// and what a low score means — and a method would be reachable only by building
/// an ONNX session. Every one of those faults produces keypoints that are
/// perfectly plausible and in the wrong place.
fn decode(
    layout: &Layout,
    mappings: &[Mapping],
    x: Axis<'_>,
    y: Axis<'_>,
    split_ratio: f32,
    threshold: f32,
) -> Vec<Keypoints2d> {
    let mut results = Vec::with_capacity(mappings.len());

    for (person, mapping) in mappings.iter().enumerate() {
        let mut keypoints = Keypoints2d::default();

        for (joint, source_index) in &layout.joints {
            let (x_bin, x_score) = argmax(x.slice(person, layout.count, *source_index));
            let (y_bin, y_score) = argmax(y.slice(person, layout.count, *source_index));

            // The weaker axis decides: a keypoint the model is sure about
            // horizontally and lost about vertically is not a keypoint.
            let confidence = x_score.min(y_score);
            if confidence < threshold {
                continue;
            }

            let (x, y) = mapping.to_source(x_bin as f32 / split_ratio, y_bin as f32 / split_ratio);
            keypoints.set(*joint, Keypoint { x, y, confidence });
        }

        results.push(keypoints);
    }

    results
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

    use crate::models::Joint;
    use std::collections::BTreeMap;

    /// A model emitting three keypoints, whose second one is the canonical
    /// left ankle. The gap in the indices is the point: a decode that walked
    /// the tensors in canonical order rather than looking each index up would
    /// pass every test built on a layout that happens to be in order.
    fn layout() -> Layout {
        Layout {
            count: 3,
            joints: BTreeMap::from([
                (Joint::Nose, 0usize),
                (Joint::LeftAnkle, 2),
                (Joint::RightAnkle, 1),
            ]),
        }
    }

    /// Scores for one person: `peaks` gives the bin each keypoint peaks at, and
    /// how strongly.
    fn scores(bins: usize, peaks: &[(usize, f32)]) -> Vec<f32> {
        let mut out = vec![0.0; bins * peaks.len()];
        for (keypoint, (bin, score)) in peaks.iter().enumerate() {
            out[keypoint * bins + bin] = *score;
        }
        out
    }

    fn identity() -> Mapping {
        Mapping {
            scale_x: 1.0,
            scale_y: 1.0,
            offset_x: 0.0,
            offset_y: 0.0,
        }
    }

    #[test]
    fn a_keypoint_is_read_from_the_index_its_layout_names() {
        let layout = layout();
        // Model index 2 is the left ankle, and it peaks somewhere unlike the
        // other two.
        let x = scores(8, &[(0, 0.9), (1, 0.9), (6, 0.9)]);
        let y = scores(8, &[(0, 0.9), (1, 0.9), (4, 0.9)]);

        let found = decode(
            &layout,
            &[identity()],
            Axis {
                scores: &x,
                bins: 8,
            },
            Axis {
                scores: &y,
                bins: 8,
            },
            1.0,
            0.3,
        );

        let ankle = found[0].get(Joint::LeftAnkle).expect("the left ankle");
        assert_eq!((ankle.x, ankle.y), (6.0, 4.0));
        let nose = found[0].get(Joint::Nose).expect("the nose");
        assert_eq!((nose.x, nose.y), (0.0, 0.0));
    }

    /// The two axes are not interchangeable, and swapping them puts every
    /// keypoint somewhere a body could be, which is what makes it hard to see.
    #[test]
    fn the_two_axes_do_not_get_swapped() {
        let x = scores(8, &[(7, 0.9)]);
        let y = scores(8, &[(1, 0.9)]);
        let layout = Layout {
            count: 1,
            joints: BTreeMap::from([(Joint::Nose, 0usize)]),
        };

        let found = decode(
            &layout,
            &[identity()],
            Axis {
                scores: &x,
                bins: 8,
            },
            Axis {
                scores: &y,
                bins: 8,
            },
            1.0,
            0.3,
        );

        let nose = found[0].get(Joint::Nose).unwrap();
        assert_eq!((nose.x, nose.y), (7.0, 1.0));
    }

    /// The bins are at `split_ratio` times the input resolution, and the
    /// mapping is in input pixels, so the division has to happen before it.
    #[test]
    fn the_bins_are_scaled_before_the_crop_is_undone() {
        let layout = Layout {
            count: 1,
            joints: BTreeMap::from([(Joint::Nose, 0usize)]),
        };
        let x = scores(16, &[(8, 0.9)]);
        let y = scores(16, &[(4, 0.9)]);
        let mapping = Mapping {
            scale_x: 3.0,
            scale_y: 3.0,
            offset_x: 100.0,
            offset_y: 50.0,
        };

        let found = decode(
            &layout,
            &[mapping],
            Axis {
                scores: &x,
                bins: 16,
            },
            Axis {
                scores: &y,
                bins: 16,
            },
            2.0,
            0.3,
        );

        // Bin 8 at a split ratio of 2 is input pixel 4, which is source pixel
        // 4 * 3 + 100.
        let nose = found[0].get(Joint::Nose).unwrap();
        assert_eq!((nose.x, nose.y), (112.0, 56.0));
    }

    /// A keypoint the model is sure of horizontally and lost about vertically
    /// is not a keypoint, and reporting it would put a confident ray through
    /// the wrong place.
    #[test]
    fn the_weaker_axis_decides_the_confidence() {
        let layout = Layout {
            count: 1,
            joints: BTreeMap::from([(Joint::Nose, 0usize)]),
        };
        let x = scores(8, &[(2, 0.95)]);
        let y = scores(8, &[(3, 0.10)]);

        let axes = |threshold| {
            decode(
                &layout,
                &[identity()],
                Axis {
                    scores: &x,
                    bins: 8,
                },
                Axis {
                    scores: &y,
                    bins: 8,
                },
                1.0,
                threshold,
            )
        };

        assert!(
            axes(0.3)[0].get(Joint::Nose).is_none(),
            "0.10 passed a 0.3 gate"
        );
        let weak = axes(0.05)[0].get(Joint::Nose).expect("below the gate");
        assert_eq!(weak.confidence, 0.10);
    }

    /// Every person in the batch reads its own slice. Getting the stride wrong
    /// gives the second person the first one's pose, which looks like tracking
    /// until two people stand apart.
    #[test]
    fn each_person_in_the_batch_reads_their_own_scores() {
        let layout = Layout {
            count: 1,
            joints: BTreeMap::from([(Joint::Nose, 0usize)]),
        };
        let mut x = scores(8, &[(1, 0.9)]);
        x.extend(scores(8, &[(6, 0.9)]));
        let mut y = scores(8, &[(2, 0.9)]);
        y.extend(scores(8, &[(5, 0.9)]));

        let found = decode(
            &layout,
            &[identity(), identity()],
            Axis {
                scores: &x,
                bins: 8,
            },
            Axis {
                scores: &y,
                bins: 8,
            },
            1.0,
            0.3,
        );

        assert_eq!(found.len(), 2);
        let first = found[0].get(Joint::Nose).unwrap();
        let second = found[1].get(Joint::Nose).unwrap();
        assert_eq!((first.x, first.y), (1.0, 2.0));
        assert_eq!((second.x, second.y), (6.0, 5.0));
    }

    /// The two axes have different bin counts whenever the model's input is
    /// not square, which is exactly the case for every pose model here.
    #[test]
    fn the_axes_may_have_different_bin_counts() {
        let layout = Layout {
            count: 1,
            joints: BTreeMap::from([(Joint::Nose, 0usize)]),
        };
        let x = scores(12, &[(3, 0.9)]);
        let y = scores(16, &[(9, 0.9)]);

        let found = decode(
            &layout,
            &[identity()],
            Axis {
                scores: &x,
                bins: 12,
            },
            Axis {
                scores: &y,
                bins: 16,
            },
            1.0,
            0.3,
        );

        let nose = found[0].get(Joint::Nose).unwrap();
        assert_eq!((nose.x, nose.y), (3.0, 9.0));
    }
}
