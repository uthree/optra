//! Device property control (exposure, gain, white balance, focus).
//!
//! This matters more than it looks. A webcam left on automatic exposure
//! stretches its exposure time in dim light, which both halves the frame rate
//! and smears the subject across the frame. Both are fatal for triangulating a
//! moving person: the C920 used during development sits at 15 fps and heavy
//! motion blur on auto, and at its full 30 fps once exposure is pinned.
//!
//! nokhwa cannot do this. Its setter reuses whatever auto/manual flag the
//! property currently has, so writing an exposure value to a property that is
//! in automatic mode is silently ignored. Optra therefore talks to the device's
//! DirectShow property interfaces itself, through a media source opened
//! alongside the streaming one.

#![cfg(windows)]

use anyhow::{Result, anyhow};
use windows::Win32::Media::DirectShow::{
    CameraControl_Exposure, CameraControl_Flags_Auto, CameraControl_Flags_Manual,
    CameraControl_Focus, CameraControl_Pan, CameraControl_Tilt, CameraControl_Zoom,
    IAMCameraControl, IAMVideoProcAmp, VideoProcAmp_BacklightCompensation, VideoProcAmp_Brightness,
    VideoProcAmp_Contrast, VideoProcAmp_Flags_Auto, VideoProcAmp_Flags_Manual, VideoProcAmp_Gain,
    VideoProcAmp_Saturation, VideoProcAmp_Sharpness, VideoProcAmp_WhiteBalance,
};
use windows::Win32::Media::MediaFoundation::{
    IMFMediaSource, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
    MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID,
    MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, MF_VERSION, MFCreateAttributes,
    MFCreateDeviceSource, MFSTARTUP_NOSOCKET, MFStartup,
};
use windows::core::{HSTRING, Interface};

use super::{ControlInfo, ControlSession};
use crate::config::ControlName;

#[derive(Clone, Copy)]
enum Interfaces {
    Camera(i32),
    ProcAmp(i32),
}

/// Which DirectShow interface and property id backs a control.
fn interface_of(name: ControlName) -> Interfaces {
    match name {
        ControlName::Exposure => Interfaces::Camera(CameraControl_Exposure.0),
        ControlName::Focus => Interfaces::Camera(CameraControl_Focus.0),
        ControlName::Zoom => Interfaces::Camera(CameraControl_Zoom.0),
        ControlName::Pan => Interfaces::Camera(CameraControl_Pan.0),
        ControlName::Tilt => Interfaces::Camera(CameraControl_Tilt.0),
        ControlName::Gain => Interfaces::ProcAmp(VideoProcAmp_Gain.0),
        ControlName::Brightness => Interfaces::ProcAmp(VideoProcAmp_Brightness.0),
        ControlName::Contrast => Interfaces::ProcAmp(VideoProcAmp_Contrast.0),
        ControlName::Saturation => Interfaces::ProcAmp(VideoProcAmp_Saturation.0),
        ControlName::Sharpness => Interfaces::ProcAmp(VideoProcAmp_Sharpness.0),
        ControlName::WhiteBalance => Interfaces::ProcAmp(VideoProcAmp_WhiteBalance.0),
        ControlName::BacklightCompensation => {
            Interfaces::ProcAmp(VideoProcAmp_BacklightCompensation.0)
        }
    }
}

/// A property session on one device, independent of the streaming session.
pub struct DeviceControls {
    camera: Option<IAMCameraControl>,
    proc_amp: Option<IAMVideoProcAmp>,
    // Held so the interfaces above stay alive.
    _source: IMFMediaSource,
}

impl DeviceControls {
    /// Opens a property session on the device with the given symbolic link.
    pub fn open(device_path: &str) -> Result<Self> {
        startup()?;

        let source = unsafe {
            let mut attributes = None;
            MFCreateAttributes(&mut attributes, 2)
                .map_err(|err| anyhow!("failed to create device attributes: {err}"))?;
            let attributes =
                attributes.ok_or_else(|| anyhow!("failed to create device attributes"))?;

            attributes
                .SetGUID(
                    &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                    &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID,
                )
                .map_err(|err| anyhow!("failed to set the device source type: {err}"))?;
            attributes
                .SetString(
                    &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK,
                    &HSTRING::from(device_path),
                )
                .map_err(|err| anyhow!("failed to set the device path: {err}"))?;

            MFCreateDeviceSource(&attributes)
                .map_err(|err| anyhow!("failed to open {device_path} for control: {err}"))?
        };

        // A device may implement one interface and not the other; a webcam with
        // no motor has no IAMCameraControl at all.
        let camera = source.cast::<IAMCameraControl>().ok();
        let proc_amp = source.cast::<IAMVideoProcAmp>().ok();
        if camera.is_none() && proc_amp.is_none() {
            return Err(anyhow!("{device_path} exposes no controllable properties"));
        }

        Ok(Self {
            camera,
            proc_amp,
            _source: source,
        })
    }
}

impl ControlSession for DeviceControls {
    /// Reads one property, or `None` if the device does not support it.
    fn get(&self, name: ControlName) -> Option<ControlInfo> {
        let (mut min, mut max, mut step, mut default, mut caps) = (0, 0, 0, 0, 0);
        let (mut value, mut flags) = (0, 0);

        unsafe {
            match interface_of(name) {
                Interfaces::Camera(property) => {
                    let camera = self.camera.as_ref()?;
                    camera
                        .GetRange(
                            property,
                            &mut min,
                            &mut max,
                            &mut step,
                            &mut default,
                            &mut caps,
                        )
                        .ok()?;
                    camera.Get(property, &mut value, &mut flags).ok()?;
                    Some(ControlInfo {
                        name,
                        min: min as i64,
                        max: max as i64,
                        step: step.max(1) as i64,
                        default: default as i64,
                        value: value as i64,
                        auto: flags & CameraControl_Flags_Auto.0 != 0,
                        auto_supported: caps & CameraControl_Flags_Auto.0 != 0,
                        manual_supported: caps & CameraControl_Flags_Manual.0 != 0,
                    })
                }
                Interfaces::ProcAmp(property) => {
                    let proc_amp = self.proc_amp.as_ref()?;
                    proc_amp
                        .GetRange(
                            property,
                            &mut min,
                            &mut max,
                            &mut step,
                            &mut default,
                            &mut caps,
                        )
                        .ok()?;
                    proc_amp.Get(property, &mut value, &mut flags).ok()?;
                    Some(ControlInfo {
                        name,
                        min: min as i64,
                        max: max as i64,
                        step: step.max(1) as i64,
                        default: default as i64,
                        value: value as i64,
                        auto: flags & VideoProcAmp_Flags_Auto.0 != 0,
                        auto_supported: caps & VideoProcAmp_Flags_Auto.0 != 0,
                        manual_supported: caps & VideoProcAmp_Flags_Manual.0 != 0,
                    })
                }
            }
        }
    }

    /// Everything the device reports.
    fn list(&self) -> Vec<ControlInfo> {
        ControlName::ALL
            .iter()
            .filter_map(|name| self.get(*name))
            .collect()
    }

    /// Writes a property. `auto` selects the device's own regulation; the value
    /// is ignored in that case, which is why the flag has to be explicit.
    fn set(&self, name: ControlName, value: i64, auto: bool) -> Result<()> {
        unsafe {
            match interface_of(name) {
                Interfaces::Camera(property) => {
                    let camera = self
                        .camera
                        .as_ref()
                        .ok_or_else(|| anyhow!("this device has no {} control", name.label()))?;
                    let flags = if auto {
                        CameraControl_Flags_Auto
                    } else {
                        CameraControl_Flags_Manual
                    };
                    camera
                        .Set(property, value as i32, flags.0)
                        .map_err(|err| anyhow!("failed to set {}: {err}", name.label()))
                }
                Interfaces::ProcAmp(property) => {
                    let proc_amp = self
                        .proc_amp
                        .as_ref()
                        .ok_or_else(|| anyhow!("this device has no {} control", name.label()))?;
                    let flags = if auto {
                        VideoProcAmp_Flags_Auto
                    } else {
                        VideoProcAmp_Flags_Manual
                    };
                    proc_amp
                        .Set(property, value as i32, flags.0)
                        .map_err(|err| anyhow!("failed to set {}: {err}", name.label()))
                }
            }
        }
    }
}

/// Media Foundation has to be started once per process before any of this
/// works. nokhwa does the same thing behind its own flag; both are reference
/// counted, so doing it twice is harmless.
fn startup() -> Result<()> {
    use std::sync::OnceLock;

    static STARTED: OnceLock<Result<(), String>> = OnceLock::new();

    super::webcam::init_com();
    STARTED
        .get_or_init(|| unsafe {
            MFStartup(MF_VERSION, MFSTARTUP_NOSOCKET).map_err(|err| err.to_string())
        })
        .clone()
        .map_err(|err| anyhow!("failed to start Media Foundation: {err}"))
}
