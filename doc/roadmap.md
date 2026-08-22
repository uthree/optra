# Optra Implementation Roadmap

Milestones are ordered so that each one is independently testable and leaves the
application in a runnable state. See [design.md](design.md) for the architecture
these steps build toward.

| Milestone | Status |
|---|---|
| M0 - Project skeleton | done |
| M1 - Camera capture | done |
| M2 - Inference | next |
| M3 - VR link and calibration | |
| M4 - Fusion | |
| M5 - Tracker output | |
| M6 - Tuning and release | |

## M0 - Project skeleton

- Workspace-less single crate, `Cargo.toml` with the dependency set from the
  design document.
- `tracing` logging, `anyhow` error handling, config load/save under
  `%APPDATA%/optra`.
- `eframe` window with the six empty panels and a shared `AppState`.
- Thread supervision helper: named threads, shutdown signal, panic reporting
  back to the UI.

**Done when:** the app starts, shows the panel layout, and persists window and
config state across restarts.

## M1 - Camera capture

- Device enumeration and format negotiation via `nokhwa`.
- One capture thread per camera with monotonic timestamps and a single-slot
  mailbox.
- Synthetic frame source: a virtual room rendered from a ceiling corner, so the
  multi-camera paths can be developed and tested without the hardware.
- Device property control (exposure, gain, focus, white balance) through
  DirectShow, applied at open and adjustable live.
- Camera panel: device list, per-camera resolution/FPS/format selection, lens
  type, rotation, property sliders, live preview grid, measured FPS and
  missed-frame counters.
- Camera identity persisted by device path.

**Done when:** cameras of different models, resolutions and frame rates stream
simultaneously with stable measured FPS, and the UI makes bandwidth problems
visible.

Two findings from this milestone changed the design, both recorded in
[design.md](design.md):

- Media Foundation misreports the frame rate of a format after it is set, and
  cannot be trusted for what the camera panel shows. The value returned by the
  setter is used instead.
- Automatic exposure halves the frame rate in normal room lighting. Manual
  exposure control is not optional, and nokhwa cannot provide it.

## M2 - Inference

- `ort` session setup with the DirectML execution provider and a CPU fallback.
- Model registry: `manifest.toml` and model spec parsing, download with
  progress, SHA-256 verification, license display and gate, graph validation
  against the declared tensors.
- `Detector` / `Pose2d` / `MultiPose2d` traits and the architecture adapter
  registry. First adapters: `yolox` and `simcc`, then `heatmap`, `movenet`,
  `rtdetr`.
- Batching grouped by model, per-camera model assignment and inference stride.
- Canonical keypoint mapping driven by `keypoints.toml`, covering COCO-17,
  COCO-WholeBody-133 and MoveNet-17.
- Runtime model swap: background session build and warm-up, atomic replace.
- Model panel with per-model benchmark (ms/frame, achieved FPS) and local ONNX
  registration.
- Keypoint overlay on the camera previews.

**Done when:** all cameras show live 2D skeletons, each camera can run a
different model, and swapping a model at runtime neither blocks the UI nor
interrupts tracking. Adding a new checkpoint of a supported architecture
requires editing only the manifest.

## M3 - VR link and calibration

- OpenVR background client reading HMD and controller poses.
- Correspondence recorder for the calibration walk.
- Lens models: radial-tangential and equidistant fisheye, selected per camera.
- DLT resection with RANSAC, RQ decomposition into `K`, `R`, `t`.
- Levenberg-Marquardt bundle refinement on angular residuals, including
  distortion and the HMD-to-head-keypoint offset.
- Per-camera latency estimation by cross-correlation.
- Calibration wizard UI: guidance, live coverage map, residual reporting,
  degenerate-configuration detection, profile save/load.
- 3D viewport showing camera frusta, play-space bounds and the floor grid.

**Done when:** a calibration run converges to a low RMS reprojection error and
the reconstructed camera positions match the physical room layout.

## M4 - Fusion

- Fixed-rate fusion clock with per-camera interpolation to a common timestamp.
- Angular-weighted triangulation with RANSAC and non-linear refinement, with
  per-camera contribution weights exposed to the UI.
- Bone-length measurement step and constrained skeleton fit.
- One Euro filtering and the constant-velocity Kalman predictor.
- 3D viewport shows the live skeleton with per-joint confidence and residuals.

**Done when:** the 3D skeleton tracks the user smoothly, holds up under partial
occlusion, and per-joint residuals stay within a few pixels.

## M5 - Tracker output

- Limb-frame orientation derivation for each tracker role.
- `TrackerSink` trait, VRChat OSC backend, VMT backend.
- Output thread with configurable send rate and prediction horizon.
- Output panel: sink selection, tracker enable list (3-8), per-tracker offsets.

**Done when:** VRChat full-body tracking works end to end with hips and both
feet, and the same session works through VMT under SteamVR.

## M6 - Tuning and release

- Room profile management, multiple named profiles.
- Startup self-check: cameras present, calibration loaded, models available.
- Performance pass: allocation reuse in the hot path, preview downscaling,
  optional per-camera inference rate limits.
- User documentation: camera placement guide, calibration guide,
  troubleshooting.
- Release build configuration and packaging.

**Done when:** a fresh install can be taken from download to working full-body
tracking using only the in-app documentation.

## Deferred

These are deliberately out of scope until the milestones above are complete:

- CUDA / TensorRT execution providers.
- Linux support.
- Multi-person tracking.
- Hand and finger tracking.
