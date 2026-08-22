//! Turning images into model input tensors.
//!
//! Each transform also produces the mapping back, because a detection or a
//! keypoint is meaningless until it is expressed in the source image again.

use crate::models::manifest::{ColorOrder, InputSpec};

use super::traits::{Detection, ImageView};

/// Maps a point in model input space back to the source image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mapping {
    /// Source pixels per model pixel.
    pub scale_x: f32,
    pub scale_y: f32,
    /// Source coordinate of model coordinate zero.
    pub offset_x: f32,
    pub offset_y: f32,
}

impl Mapping {
    pub fn to_source(&self, x: f32, y: f32) -> (f32, f32) {
        (
            x * self.scale_x + self.offset_x,
            y * self.scale_y + self.offset_y,
        )
    }
}

/// An NCHW float tensor plus the mapping that produced it.
pub struct Input {
    pub data: Vec<f32>,
    pub mapping: Mapping,
}

/// Fits a whole image into the model's square input, preserving aspect ratio.
///
/// The image is aligned to the top left and the remainder is padded, which is
/// what mmdetection's export pipeline does. Centring it instead would shift
/// every box by half the padding.
pub fn letterbox(image: &ImageView<'_>, spec: &InputSpec, pad: u8) -> Input {
    let (width, height) = (spec.width as usize, spec.height as usize);
    let scale =
        (spec.width as f32 / image.width as f32).min(spec.height as f32 / image.height as f32);

    let mut data = vec![0f32; 3 * width * height];
    let pad_value = pad as f32;

    for y in 0..height {
        let source_y = (y as f32 / scale).floor() as i32;
        let inside_y = source_y < image.height as i32;
        for x in 0..width {
            let source_x = (x as f32 / scale).floor() as i32;
            let pixel = if inside_y && source_x < image.width as i32 {
                image.sample(source_x, source_y)
            } else {
                [pad, pad, pad]
            };
            write_pixel(&mut data, width, height, x, y, pixel, pad_value, spec);
        }
    }

    Input {
        data,
        mapping: Mapping {
            scale_x: 1.0 / scale,
            scale_y: 1.0 / scale,
            offset_x: 0.0,
            offset_y: 0.0,
        },
    }
}

/// Crops a person out of an image into the model's aspect ratio.
///
/// The box is grown by `padding` and then widened or heightened to match the
/// model, so that the person keeps their proportions; a squashed person is a
/// person the model has never seen.
pub fn affine_crop(
    image: &ImageView<'_>,
    person: &Detection,
    spec: &InputSpec,
    padding: f32,
) -> Input {
    let (width, height) = (spec.width as usize, spec.height as usize);
    let (center_x, center_y) = person.center();

    let aspect = spec.width as f32 / spec.height as f32;
    let mut box_w = person.width().max(1.0) * padding;
    let mut box_h = person.height().max(1.0) * padding;
    if box_w > box_h * aspect {
        box_h = box_w / aspect;
    } else {
        box_w = box_h * aspect;
    }

    let scale_x = box_w / spec.width as f32;
    let scale_y = box_h / spec.height as f32;
    let offset_x = center_x - box_w * 0.5;
    let offset_y = center_y - box_h * 0.5;

    let mut data = vec![0f32; 3 * width * height];
    for y in 0..height {
        let source_y = (y as f32 * scale_y + offset_y).round() as i32;
        for x in 0..width {
            let source_x = (x as f32 * scale_x + offset_x).round() as i32;
            let pixel = image.sample(source_x, source_y);
            write_pixel(&mut data, width, height, x, y, pixel, 0.0, spec);
        }
    }

    Input {
        data,
        mapping: Mapping {
            scale_x,
            scale_y,
            offset_x,
            offset_y,
        },
    }
}

/// Writes one pixel into the NCHW tensor, applying channel order and
/// normalization.
#[inline]
#[allow(clippy::too_many_arguments)]
fn write_pixel(
    data: &mut [f32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    rgb: [u8; 3],
    _pad: f32,
    spec: &InputSpec,
) {
    let ordered = match spec.color {
        ColorOrder::Rgb => [rgb[0], rgb[1], rgb[2]],
        ColorOrder::Bgr => [rgb[2], rgb[1], rgb[0]],
    };

    let plane = width * height;
    let index = y * width + x;
    for channel in 0..3 {
        let value = (ordered[channel] as f32 - spec.mean[channel]) / spec.std[channel];
        data[channel * plane + index] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::manifest::ResizeMode;

    fn spec(width: u32, height: u32, color: ColorOrder) -> InputSpec {
        InputSpec {
            name: "input".to_owned(),
            width,
            height,
            color,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
            resize: ResizeMode::Letterbox { pad: 114 },
        }
    }

    /// A 4x2 image of a single colour, so any sampling error shows up as a
    /// wrong channel rather than a wrong pixel.
    fn image(width: u32, height: u32, color: [u8; 3]) -> (u32, u32, Vec<u8>) {
        let mut rgb = Vec::with_capacity((width * height * 3) as usize);
        for _ in 0..width * height {
            rgb.extend_from_slice(&color);
        }
        (width, height, rgb)
    }

    #[test]
    fn letterbox_preserves_aspect_ratio_and_pads_the_rest() {
        let (w, h, rgb) = image(4, 2, [10, 20, 30]);
        let view = ImageView::new(w, h, &rgb);
        let spec = spec(4, 4, ColorOrder::Rgb);

        let input = letterbox(&view, &spec, 114);
        let plane = 16;

        // The image occupies the top half; the bottom half is padding.
        assert_eq!(input.data[0], 10.0);
        assert_eq!(input.data[plane], 20.0);
        assert_eq!(input.data[2 * plane], 30.0);
        assert_eq!(
            input.data[3 * 4],
            114.0,
            "the padded rows carry the pad value"
        );

        // Scale is 1 here, so a model coordinate is a source coordinate.
        let (x, y) = input.mapping.to_source(2.0, 1.0);
        assert_eq!((x, y), (2.0, 1.0));
    }

    #[test]
    fn letterbox_maps_coordinates_back_through_the_scale() {
        let (w, h, rgb) = image(8, 4, [1, 2, 3]);
        let view = ImageView::new(w, h, &rgb);
        let spec = spec(4, 4, ColorOrder::Rgb);

        let input = letterbox(&view, &spec, 114);
        // The 8-wide image fits into 4 pixels, so one model pixel is two source
        // pixels.
        let (x, y) = input.mapping.to_source(1.0, 1.0);
        assert_eq!((x, y), (2.0, 2.0));
    }

    #[test]
    fn bgr_models_get_their_channels_swapped() {
        let (w, h, rgb) = image(2, 2, [10, 20, 30]);
        let view = ImageView::new(w, h, &rgb);
        let spec = spec(2, 2, ColorOrder::Bgr);

        let input = letterbox(&view, &spec, 0);
        let plane = 4;
        assert_eq!(input.data[0], 30.0, "the first channel should be blue");
        assert_eq!(
            input.data[2 * plane],
            10.0,
            "the last channel should be red"
        );
    }

    #[test]
    fn normalization_is_applied_per_channel() {
        let (w, h, rgb) = image(2, 2, [100, 100, 100]);
        let view = ImageView::new(w, h, &rgb);
        let mut spec = spec(2, 2, ColorOrder::Rgb);
        spec.mean = [50.0, 0.0, 100.0];
        spec.std = [2.0, 4.0, 1.0];

        let input = letterbox(&view, &spec, 0);
        let plane = 4;
        assert_eq!(input.data[0], 25.0);
        assert_eq!(input.data[plane], 25.0);
        assert_eq!(input.data[2 * plane], 0.0);
    }

    #[test]
    fn a_crop_maps_its_centre_back_to_the_box_centre() {
        let (w, h, rgb) = image(200, 200, [5, 5, 5]);
        let view = ImageView::new(w, h, &rgb);
        let spec = spec(192, 256, ColorOrder::Rgb);
        let person = Detection {
            x1: 40.0,
            y1: 60.0,
            x2: 120.0,
            y2: 180.0,
            score: 0.9,
        };

        let input = affine_crop(&view, &person, &spec, 1.25);
        let (x, y) = input
            .mapping
            .to_source(spec.width as f32 * 0.5, spec.height as f32 * 0.5);

        let (cx, cy) = person.center();
        assert!((x - cx).abs() < 0.5, "{x} should be about {cx}");
        assert!((y - cy).abs() < 0.5, "{y} should be about {cy}");
    }

    #[test]
    fn a_crop_keeps_the_person_from_being_squashed() {
        let (w, h, rgb) = image(200, 200, [5, 5, 5]);
        let view = ImageView::new(w, h, &rgb);
        let spec = spec(192, 256, ColorOrder::Rgb);

        // A wide box in a tall model input: the crop has to grow vertically
        // rather than squeeze horizontally.
        let person = Detection {
            x1: 0.0,
            y1: 90.0,
            x2: 160.0,
            y2: 110.0,
            score: 0.9,
        };

        let input = affine_crop(&view, &person, &spec, 1.0);
        let crop_w = input.mapping.scale_x * spec.width as f32;
        let crop_h = input.mapping.scale_y * spec.height as f32;

        assert!((crop_w - 160.0).abs() < 0.5, "the width should be kept");
        assert!(
            (crop_w / crop_h - spec.width as f32 / spec.height as f32).abs() < 1e-3,
            "the crop should match the model aspect ratio"
        );
    }
}
