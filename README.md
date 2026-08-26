# Optra

[![CI](https://github.com/uthree/optra/actions/workflows/ci.yml/badge.svg)](https://github.com/uthree/optra/actions/workflows/ci.yml)

Optra turns 2-4 fixed webcams into lower-body tracking for VRChat.

Cameras mounted at the corners of the ceiling watch you from above. Optra runs
ONNX pose models on each view, triangulates the keypoints into a 3D skeleton in
your SteamVR play space, and streams virtual trackers to VRChat over OSC or to
SteamVR through a virtual tracker driver. It is meant to complement the 3-point
tracking of a Quest-class headset, not to replace it.

## Requirements

- Windows 11
- An AMD or NVIDIA GPU (inference runs on ONNX Runtime DirectML)
- SteamVR with a working headset
- 2-4 USB webcams

## Install

Not yet released. To build from source:

```
cargo build --release
```

`DirectML.dll` is placed beside the executable by the build and has to travel
with it. Pose models are downloaded from within the application; none are
bundled.

## Documentation

- [Placing the cameras](doc/cameras.md)
- [Calibrating a room](doc/calibration.md)
- [Troubleshooting](doc/troubleshooting.md)
- [Design](doc/design.md)
- [Roadmap](doc/roadmap.md)

## License

Apache-2.0. See [LICENSE](LICENSE).
