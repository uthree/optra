# Optra

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

Pose models are downloaded from within the application; none are bundled.

## Documentation

- [Design](doc/design.md)
- [Roadmap](doc/roadmap.md)

## License

Apache-2.0. See [LICENSE](LICENSE).
