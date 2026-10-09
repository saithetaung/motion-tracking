# motionTrace

Real-time **person tracking with speed and distance estimation** from a single camera.

motionTrace detects people, gives each one a persistent ID, estimates how far they are from the camera and how fast they are moving, and draws it all on the video in a way that is readable at a glance. Optionally it replaces the ID with a name when it recognizes a face you enrolled.

It runs on a laptop webcam today. The same pipeline is designed to take a SIYI A8 Mini gimbal camera (RTSP).

![Example overlay](docs/overlay_preview.png)

*Example overlay on a synthetic frame: one running person approaching the camera, two walking, one standing still.*

---

## What you get

For every tracked person, on every frame:

| Output | Meaning |
|---|---|
| **Name or ID** | Name if the face matches a photo in `known_faces/`, otherwise `ID n` |
| **Speed (km/h)** | Smoothed speed, plus a label: STILL, WALKING, JOGGING or RUNNING |
| **Distance (m)** | Estimated distance from the camera |
| **Direction arrow** | Points the way the person moves. Right or left means sideways, down means toward the camera, up means away. A longer arrow means faster |
| **Closer / farther** | Shown next to the distance when the person is clearly approaching or moving away |
| **Color** | The whole overlay for a person is tinted by their motion state, from gray (still) to green, amber and red (running) |
| **Trail** | Short fading path of where they just walked |

A small status panel in the top-left corner shows FPS and the number of people tracked.

Default speed bands (change them in `src/visualization/theme.py`):

| State | Speed |
|---|---|
| STILL | under 1 km/h |
| WALKING | 1 to 6.5 km/h |
| JOGGING | 6.5 to 12 km/h |
| RUNNING | 12 km/h and above |

## How it works

```
Camera ─▶ YOLO26 + ByteTrack ─▶ Face naming (optional) ─▶ Depth Anything V2 ─▶ Distance ─▶ Speed ─▶ Overlay
```

- **Detection and tracking:** [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO26 (person class only) with ByteTrack, so each person keeps the same ID across frames.
- **Distance:** a monocular depth model ([Depth Anything V2](https://huggingface.co/depth-anything), metric indoor checkpoint, via Hugging Face Transformers) produces a depth map. Distance is read at the person's position. There is no depth sensor and no manual calibration of distances.
- **Speed:** net movement over a short real-time window (0.5 s), combining sideways motion with the change in distance. It is smoothed and has a small dead zone, so someone standing still reads 0 instead of jittering.
- **Face naming (optional):** OpenCV's YuNet face detector and SFace recognizer. It only looks at the head region of people who do not have a name yet, and locks a name onto the track after consistent matches.
- **Camera abstraction:** webcam and RTSP cameras share one interface, so the rest of the pipeline never changes when you switch cameras.

## Requirements

- Python 3.11
- An NVIDIA GPU is strongly recommended. It also runs on CPU, but the depth model will be slow.
- Windows is what this was developed on. Linux and macOS should work, but are untested.

## Quick start

```bash
git clone <your-repo-url>
cd motionTrace

python -m venv venv
venv\Scripts\activate            # Linux/macOS: source venv/bin/activate

pip install -r requirements.txt
```

Install PyTorch **last**, from the index that matches your GPU, so nothing swaps it back to a CPU build:

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

- Pick the right index at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/). RTX 50-series (Blackwell) GPUs need a CUDA 13.x build such as `cu130`, which in turn needs a recent NVIDIA driver. Older GPUs can use older CUDA builds, and `cpu` works with no GPU.
- Always install `torch` and `torchvision` together from the same index.

Check the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Optional: a free [Hugging Face token](https://huggingface.co/settings/tokens) avoids download rate limits. Create a `.env` file in the project root:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

## Run

```bash
python main.py
```

Press `q` in the video window to quit. The first run downloads the YOLO weights, the depth model and (if face naming is on) the face models, so it can take a few minutes. Later runs start fast.

Choose the camera and options in `main.py`:

```python
pipeline = build_pipeline(
    single_person_mode=True,    # False tracks everyone in view
    camera_type="webcam",       # or "a8mini"
    identify_faces=True,        # False turns face naming off
)
```

Preview the overlay design without a camera or any models:

```bash
python preview_overlay.py       # writes overlay_preview.png
```

## Face naming

Create a `known_faces/` folder in the project root and add a clear, front-facing photo per person. The file name becomes the displayed name:

```
known_faces/
├── alex.jpg              # shown as "alex"
└── maria/                # or one folder per person with several photos
    ├── 1.jpg
    └── 2.jpg
```

- Use well-lit photos with the face large in the frame and no sunglasses or mask. Two or three photos per person (different angle or light) work noticeably better than one.
- Recognition needs a reasonably large face (roughly 60 to 80 px wide). It works at webcam range, but **not** at typical  distances.
- If someone is not enrolled, they keep their `ID n`.
- `known_faces/` is git-ignored on purpose. These are biometric images: only enroll people who agreed to it, and never commit them.

## Using the SIYI A8 Mini (RTSP)

> Status: code and config are in place, but this has **not yet been tested on real hardware**.

1. Connect the camera by Ethernet and give your PC a static IP on its subnet, for example `192.168.144.20` with mask `255.255.255.0`.
2. Check it is reachable: `ping 192.168.144.25`.
3. Open `rtsp://192.168.144.25:8554/main.264` in VLC. If video plays there, the network and camera are fine.
4. Set `camera_type="a8mini"` in `main.py` and run it.

Settings live in `configs/camera_a8mini.yaml`. The A8 Mini has a different lens from a webcam, so run its own calibration (below) and keep the zoom fixed while tracking.

## Optional: camera calibration

Distance comes from the depth model and does not need calibration. **Speed** converts sideways pixel movement into meters using the camera's focal length, and that is more accurate when calibrated.

```bash
python scripts/auto_calibrate.py
```

Hold a printed 9x6 checkerboard in front of the camera for a few seconds. The result is saved to `configs/camera_webcam.yaml` (git-ignored, because it belongs to your specific camera). Skipping this still works, with a rough default focal length.

## Project structure

```
motionTrace/
├── main.py                       entry point
├── preview_overlay.py            render the overlay on a sample frame
├── requirements.txt
├── configs/
│   ├── camera_webcam.yaml        per-machine calibration (git-ignored)
│   └── camera_a8mini.yaml        RTSP address and calibration for the A8 Mini
├── known_faces/                  your enrolled photos (git-ignored)
├── models/                       auto-downloaded face models (git-ignored)
├── scripts/
│   └── auto_calibrate.py         one-time checkerboard calibration
├── src/
│   ├── pipeline.py               runs the whole loop, build_pipeline() factory
│   ├── worker.py                 background-thread runner for a future service layer
│   ├── schemas.py                Track: the data passed between stages
│   ├── camera/                   webcam and A8 Mini RTSP sources (threaded capture)
│   ├── detection_tracking/       YOLO26 + ByteTrack
│   ├── recognition/              optional face naming
│   ├── metrics/                  depth model, distance and speed estimators
│   └── visualization/            overlay renderer, theme, drawing helpers
└── tests/
```

## Configuration

| What | Where | Default |
|---|---|---|
| Speed bands, colors, card opacity | `Theme` in `src/visualization/theme.py` | see above |
| How often the depth model runs | `depth_every_n_frames` in `src/pipeline.py` | every 6th frame |
| Speed smoothing window and dead zone | `SpeedEstimator` in `src/metrics/speed_estimator.py` | 0.5 s, 3 cm |
| Face match strictness | `match_threshold` in `FaceIdentifier` | 0.363 (higher is stricter) |
| Depth model size | `MonocularDepthEstimator(model_size=...)` | `small` |

## Accuracy and limitations

Please read this before trusting the numbers.

- **Distance and speed are estimates, not measurements.** The depth model is best at normal indoor range and gets less reliable with distance. The metric checkpoint used here is trained mostly on indoor scenes. Measure a few real distances with a tape and compare before relying on it.
- **The camera must be static.** The math assumes the camera itself does not move. A panning gimbal or a flying  breaks it, and handling that (GPS/IMU or optical-flow compensation) is a separate piece of work, not a setting.
- **Zoom changes invalidate calibration.** Keep the zoom fixed on the A8 Mini.
- **Closer/farther can flicker** for a still person because depth readings are noisy. Raise `trend_min_mps` in `Theme` if it does.
- **Track IDs can swap** when two people cross paths, and a face name stays with the track ID.
- Observed on a laptop GPU: detection about 30 ms per frame and depth about 60 ms, with depth running only every few frames. Your numbers will vary.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run from the project root with `python main.py`, not from inside a subfolder |
| `cudaErrorNotSupported`, or a warning that your GPU is not compatible with this PyTorch | Your PyTorch CUDA build and NVIDIA driver do not match. Update the driver, then reinstall `torch` and `torchvision` from a matching index |
| `Could not run 'torchvision::nms' with arguments from the 'CUDA' backend` | `torchvision` is a different build from `torch`. Reinstall both together from the same index |
| Hugging Face warning about unauthenticated requests | Harmless. Add `HF_TOKEN` to `.env` to remove it |
| Video is laggy | Raise `depth_every_n_frames`, or use a smaller depth model |
| A8 Mini will not connect | Check `ping 192.168.144.25`, the static IP on your adapter, and test the stream in VLC first |

## Tests

```bash
pip install pytest
python -m pytest tests
```

## Roadmap

- [x] **Phase 0:** laptop webcam, detection, tracking, distance, speed, overlay
- [x] Face naming from a folder of photos
- [ ] **Phase 1:** SIYI A8 Mini over RTSP (code ready, hardware test and calibration pending)
- [ ] **Phase 2:** arial-mounted camera (needs motion compensation, not started)
- [ ] Service layer so other systems can consume the tracking results
