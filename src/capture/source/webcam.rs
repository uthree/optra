//! Media Foundation capture via nokhwa.

use std::time::Instant;

use anyhow::{Result, anyhow};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{
    ApiBackend, CameraFormat, CameraInfo, FrameFormat, RequestedFormat, RequestedFormatType,
};
use nokhwa::{Camera, query};

use super::{FrameSource, NegotiatedFormat, RawFrame};
use crate::config::CameraConfig;

pub struct WebcamSource {
    camera: Camera,
    negotiated: NegotiatedFormat,
}

impl WebcamSource {
    pub fn open(config: &CameraConfig, device_path: &str, device_name: &str) -> Result<Self> {
        init_com();

        let info = find_device(device_path).ok_or_else(|| {
            anyhow!("camera \"{device_name}\" is not connected (device path {device_path})")
        })?;

        // Open before choosing a format: what a device advertises through
        // Media Foundation rarely matches what its spec sheet claims, so the
        // choice is made from the real list rather than from a guess.
        let mut camera = Camera::new(
            info.index().clone(),
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate),
        )
        .map_err(|err| anyhow!("failed to open \"{device_name}\": {err}"))?;

        let available = camera
            .compatible_camera_formats()
            .map_err(|err| anyhow!("failed to list the formats of \"{device_name}\": {err}"))?;

        let format = match pick_format(&available, config.width, config.height, config.fps) {
            // The resolved format is taken from the setter's return value.
            // Re-reading it from the device afterwards is not reliable: Media
            // Foundation reports a frame rate of 1 for formats it is happily
            // streaming at 30.
            Some(chosen) => camera
                .set_camera_requset(RequestedFormat::new::<RgbFormat>(
                    RequestedFormatType::Exact(chosen),
                ))
                .map_err(|err| anyhow!("failed to select {chosen} on \"{device_name}\": {err}"))?,
            None => {
                tracing::warn!(
                    camera = %config.id,
                    "the device reported no usable formats; keeping its default"
                );
                camera.camera_format()
            }
        };

        camera
            .open_stream()
            .map_err(|err| anyhow!("failed to start the stream on \"{device_name}\": {err}"))?;

        let negotiated = NegotiatedFormat {
            width: format.resolution().width(),
            height: format.resolution().height(),
            fps: format.frame_rate(),
            pixel_format: format.format().to_string(),
        };

        Ok(Self { camera, negotiated })
    }
}

impl FrameSource for WebcamSource {
    fn next_frame(&mut self) -> Result<RawFrame> {
        let buffer = self
            .camera
            .frame()
            .map_err(|err| anyhow!("failed to read a frame: {err}"))?;

        let resolution = buffer.resolution();
        let (width, height) = (resolution.width(), resolution.height());

        let started = Instant::now();
        let mut rgb = vec![0u8; width as usize * height as usize * 3];
        buffer
            .decode_image_to_buffer::<RgbFormat>(&mut rgb)
            .map_err(|err| anyhow!("failed to decode a frame: {err}"))?;

        Ok(RawFrame {
            width,
            height,
            rgb,
            decode: started.elapsed(),
        })
    }

    fn negotiated(&self) -> NegotiatedFormat {
        self.negotiated.clone()
    }
}

/// Chooses the format closest to what was requested.
///
/// Resolution matters most, then frame rate, and only then the pixel format.
/// Between otherwise equal candidates the least bus-hungry encoding wins, since
/// several cameras have to share the USB controller.
fn pick_format(
    available: &[CameraFormat],
    width: u32,
    height: u32,
    fps: u32,
) -> Option<CameraFormat> {
    available.iter().copied().min_by_key(|format| {
        let resolution = format.resolution();
        (
            resolution.width().abs_diff(width) as u64 + resolution.height().abs_diff(height) as u64,
            format.frame_rate().abs_diff(fps) as u64,
            bandwidth_rank(format.format()),
        )
    })
}

fn bandwidth_rank(format: FrameFormat) -> u8 {
    match format {
        FrameFormat::MJPEG => 0,
        FrameFormat::NV12 => 1,
        FrameFormat::YUYV => 2,
        FrameFormat::RAWRGB | FrameFormat::RAWBGR => 3,
        FrameFormat::GRAY => 4,
    }
}

/// Lists the connected capture devices.
pub fn list_devices() -> Result<Vec<CameraInfo>> {
    init_com();
    query(ApiBackend::MediaFoundation)
        .map_err(|err| anyhow!("failed to list capture devices: {err}"))
}

/// Lists the formats a device supports, deduplicated and sorted.
///
/// Opening the device is the only way to ask, so this cannot be called while
/// the camera is streaming.
pub fn list_formats(device_path: &str) -> Result<Vec<CameraFormat>> {
    init_com();

    let info =
        find_device(device_path).ok_or_else(|| anyhow!("camera {device_path} is not connected"))?;

    let mut camera = Camera::new(
        info.index().clone(),
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate),
    )
    .map_err(|err| anyhow!("failed to open the camera: {err}"))?;

    let mut formats = camera
        .compatible_camera_formats()
        .map_err(|err| anyhow!("failed to list the formats: {err}"))?;

    formats.sort_by_key(|format| {
        (
            std::cmp::Reverse(format.resolution().width()),
            std::cmp::Reverse(format.resolution().height()),
            std::cmp::Reverse(format.frame_rate()),
            bandwidth_rank(format.format()),
        )
    });
    formats.dedup();
    Ok(formats)
}

/// Finds a device by its Media Foundation symbolic link.
fn find_device(device_path: &str) -> Option<CameraInfo> {
    list_devices()
        .ok()?
        .into_iter()
        .find(|info| info.misc() == device_path)
}

/// Initializes COM for the calling thread.
///
/// nokhwa initializes Media Foundation once per process, which leaves every
/// other capture thread without a COM apartment. Doing it here keeps one thread
/// per camera workable. Repeat calls and `RPC_E_CHANGED_MODE` are both expected
/// and ignored.
#[cfg(windows)]
pub(super) fn init_com() {
    use windows::Win32::System::Com::{
        COINIT_APARTMENTTHREADED, COINIT_DISABLE_OLE1DDE, CoInitializeEx,
    };

    unsafe {
        let _ = CoInitializeEx(None, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    }
}
