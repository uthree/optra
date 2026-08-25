# Optra Implementation Roadmap

Milestones are ordered so that each one is independently testable and leaves the
application in a runnable state. See [design.md](design.md) for the architecture
these steps build toward.

| Milestone | Status |
|---|---|
| M0 - Project skeleton | done |
| M1 - Camera capture | done |
| M2 - Inference | done |
| M3 - VR link and calibration | done, pending a room |
| M4 - Fusion | done, pending a room |
| M5 - Tracker output | done, pending a consumer |
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
- Per-camera latency estimation. Done, as a search over the reprojection error
  rather than a correlation; see [design.md](design.md).
- Calibration wizard UI: guidance, live coverage map, residual reporting,
  degenerate-configuration detection, profile save/load. Done.
- 3D viewport showing camera frusta and the floor grid. Done. Play-space
  bounds need the chaperone interface, which is a separate function table and
  is left until something else needs it.

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

What is left is a walk through a real room with the cameras where they will
live. Everything the milestone calls for is written and green against synthetic
rooms and against a Quest 3.

## M4 - Fusion

- Fixed-rate fusion clock with per-camera interpolation to a common timestamp.
  Done. The clock runs behind real time by the slowest camera's measured delay
  plus a slack, because interpolating onto an instant needs a frame after it.
- Angular-weighted triangulation with RANSAC and non-linear refinement, with
  per-camera contribution weights exposed to the UI. Done.
- Bone-length measurement step and constrained skeleton fit. Done, without the
  T-pose step the design called for; see [design.md](design.md).
- One Euro filtering and the constant-velocity Kalman predictor. Done, in that
  order rather than the design's; see [design.md](design.md).
- 3D viewport shows the live skeleton with per-joint confidence and residuals.
  Done, with the predicted skeleton drawn behind the measured one.

**Done when:** the 3D skeleton tracks the user smoothly, holds up under partial
occlusion, and per-joint residuals stay within a few pixels.

What this milestone found, all recorded in [design.md](design.md):

- The linear triangulation is biased towards the nearest cameras, and inverting
  the refinement's normal matrix gives the position covariance for free. The
  square root of its largest eigenvalue — how far a joint could be wrong along
  the direction the cameras constrain least — turned out to be far more useful
  than the reprojection residual. Two cameras a hand's width apart agree
  perfectly about a point neither can place: the residual says zero and the
  uncertainty says a third of a metre.
- The Kalman filter has to come before the One Euro filter, not after. Reversed,
  the velocity is measured from the smoothed signal and every prediction comes
  out short by exactly the amount the smoothing lagged.
- The smoothing costs no latency, because a first-order low pass lags by exactly
  its own time constant and that can simply be added to the prediction horizon.
- The speed that opens the adaptive cutoff has to be low-passed, or a joint
  standing still shows enough apparent speed to open the filter. At the one
  hertz that filter is usually given, a swinging leg is attenuated enough that
  every stride lags; three hertz is the trade.
- Bone lengths belong in their own file, not the room profile. A body belongs to
  a person and a room profile to a set of cameras.

`tests/fusion.rs` runs a simulated walk past three unlike cameras — 30/60/25 fps,
three resolutions, three fields of view, delays of 0, 40 and 90 ms — through the
whole chain and recovers the body to 4 mm. Treating those delays as zero gives
6.7 cm, which is the argument for temporal alignment in one number. Hiding a
knee from two of the three cameras leaves the fit to place it, and it does, to
4 cm.

What is left is the same thing M3 is waiting on: a real room.

## M5 - Tracker output

- Limb-frame orientation derivation for each tracker role.
- `TrackerSink` trait, VRChat OSC backend, VMT backend.
- Output thread with configurable send rate and prediction horizon.
- Output panel: sink selection, tracker enable list (3-8), per-tracker offsets.

**Done when:** VRChat full-body tracking works end to end with hips and both
feet, and the same session works through VMT under SteamVR.

Three things came out of building it, all recorded in
[design.md](design.md):

- **A knee and an elbow need opposite signs.** They are the same three points
  and the same cross product, and the bend plane normal points forwards for one
  and backwards for the other, because the kneecap is on the front of the body
  and the point of the elbow is on the back. Caught by a test, not by reading.
- **The prediction horizon should never have been a camera setting.** The
  distance from a frame being exposed to a reconstruction existing is measured
  by the fusion stage. Only the hop out and the consumer's own delay are
  unknowable from here, so that is all the setting covers now, and its default
  dropped from 60 ms to 20.
- **The fallbacks are the common case, not the exception.** A straight knee has
  no bend plane and a standing person's knees are straight; a foot without heel
  and toe keypoints has no yaw of its own. Both fall back to the hip axis.

First run in VRChat: the trackers were recognised over OSC and the coordinate
conversions held, but the body vibrated on the spot. The cause was in the
filter rather than in anything this milestone added -- the prediction was
riding on an unsmoothed velocity, and the term that pays back the smoothing lag
is scaled by the largest number in the filter exactly when the joint is still.
See section 9.5 of [design.md](design.md); the fix cost 1.9 cm on a simulated
walk and made a standing body steadier than its own measurement noise.

The layout tests could not have caught it, and neither could the filter tests as
they stood: they judged the *smoothed position*, and what the output stage sends
is the *prediction*, which reaches further. Testing the wrong end of a filter is
an easy thing to do for a long time.

Untested against a real consumer: everything downstream is another process, so
the tests send to a loopback socket and decode what arrived. The conversions
they check are the ones this project believes are right, which is not the same
as the ones VRChat implements.

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
