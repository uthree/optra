# Optra Implementation Roadmap

Milestones are ordered so that each one is independently testable and leaves the
application in a runnable state. See [design.md](design.md) for the architecture
these steps build toward.

| Milestone | Status |
|---|---|
| M0 - Project skeleton | done |
| M1 - Camera capture | done |
| M2 - Inference | done |
| M3 - VR link and calibration | in progress |
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
- `Detector` and `Pose2d` traits and the architecture adapter registry, with the
  `mmdet_end2end` and `simcc` adapters.
- Per-camera model assignment and a detector stride, with the box carried by the
  previous keypoints between detector runs.
- Canonical keypoint mapping driven by `keypoints.toml`, covering COCO-17,
  Halpe-26 and COCO-WholeBody-133.
- Runtime model swap: background session build and warm-up, and a camera keeps
  its current model until the replacement is ready.
- Model panel: catalogue with license badges, install progress, execution
  provider and default model selection.
- Keypoint overlay on the camera previews.
- A still-image source, so the stages downstream can be tested against a known
  scene.

**Done when:** all cameras show live 2D skeletons, each camera can run a
different model, and swapping a model at runtime neither blocks the UI nor
interrupts tracking. Adding a new checkpoint of a supported architecture
requires editing only the manifest.

What this milestone found:

- PINTO's model zoo publishes every conversion of a model in one archive, which
  runs to gigabytes for a single ONNX file. The catalogue therefore fetches the
  same models from their upstream publishers and records the zoo entry
  alongside. See [design.md](design.md).
- Halpe 26 replaced the 133-point whole-body model as the default: it carries
  the same heel and toe points without the face and hand keypoints.
- Only the `mmdet_end2end` and `simcc` adapters were needed for the shipped
  catalogue; the others in the design remain unwritten until a model needs them.

Left for later: a per-model benchmark in the model panel, and a UI for
registering a local ONNX file. The manifest already supports a local source, so
that one is a hand-edited entry in the user manifest for now.

## M3 - VR link and calibration

- OpenVR background client reading HMD and controller poses. Done: the runtime
  is loaded at run time, poses are sampled into a short history, and the
  calibration panel shows the live device list.
- Correspondence recorder for the calibration walk. Done.
- Lens models: radial-tangential and equidistant fisheye, selected per camera.
  Done.
- DLT resection with RANSAC, RQ decomposition into `K`, `R`, `t`. Done.
- Levenberg-Marquardt bundle refinement on angular residuals, including
  distortion and the HMD-to-head-keypoint offset. Done.
- Per-camera latency estimation by cross-correlation.
- Calibration wizard UI: guidance, live coverage map, residual reporting,
  degenerate-configuration detection, profile save/load. Done.
- 3D viewport showing camera frusta, play-space bounds and the floor grid.

**Done when:** a calibration run converges to a low RMS reprojection error and
the reconstructed camera positions match the physical room layout.

Progress: the solver is written and verified against synthetic rooms whose
answer is known — lens models, projection, triangulation, resection and the
joint refinement. `tests/calibration.rs` runs the whole procedure on four
unlike cameras and recovers their positions to within 3 mm and the head offset
to within 3 mm, including a run with a camera blocked for half the walk and one
keypoint in twelve thrown somewhere else.

The SteamVR link is written and verified against a Quest 3: the headset and
both controllers are read with correct classes, roles and poses. Testing it
turned up two things, both recorded in [design.md](design.md):

- An OpenVR connection is a process singleton rather than a handle. A second
  one is now refused instead of crashing the first.
- A loop asking to sleep 8 ms was running at 64 Hz, because that is the rate
  Windows wakes sleeping threads at by default. This affects the fusion clock
  and the output thread as much as pose sampling, so the fix lives in
  `worker::timing`: a raised timer resolution for the process, and a ticker
  that keeps a schedule rather than an interval.

The recorder and the solver are written: a walk is turned into a room profile
by `calib::recorder` and `calib::solve`, with `tests/calibration.rs` running a
synthetic recording through the whole path and recovering four camera positions
to within 5 mm and every rig offset to within 5 mm.

The wizard is written and the whole path is reachable from the UI: check the
prerequisites, record a walk with a live coverage map and a warning when the
walk cannot be solved, solve on a worker thread, review, and save a room
profile. `tests/panels.rs` lays every panel out headlessly, which is how an
immediate-mode UI gets tested at all — a panel nobody is looking at is never
drawn, and a layout bug in it waits until someone is mid-walk.

What is left: latency estimation, the 3D viewport, and a walk through a real
room.

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
