import urllib.request
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.schemas import Track

MODEL_DIR = Path("models")
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
SFACE_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_recognition_sface/face_recognition_sface_2021dec.onnx")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _ensure_model(url: str) -> str:
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / url.rsplit("/", 1)[-1]
    if not path.exists():
        print(f"[FaceIdentifier] Downloading {path.name} (one-time)...")
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not download {path.name}: {e}\n"
                f"Download it manually from github.com/opencv/opencv_zoo "
                f"(models/face_detection_yunet, models/face_recognition_sface) "
                f"and put it in '{MODEL_DIR}/'."
            ) from e
    return str(path)


class FaceIdentifier:
    """
    Assigns names to tracks by matching faces against ./known_faces.

    - Face detection runs only on the head region of each *unnamed* track,
      every few frames, so cost is small and drops to zero once a name locks.
    - A name locks onto a track_id after `votes_needed` consistent matches,
      which avoids one-frame mislabels.
    """

    def __init__(self, known_faces_dir: str = "known_faces",
                 match_threshold: float = 0.363,   # SFace cosine threshold (OpenCV docs)
                 votes_needed: int = 2,
                 retry_every_n_frames: int = 5,
                 head_ratio: float = 0.45):        # top 45% of the person box
        self.match_threshold = match_threshold
        self.votes_needed = votes_needed
        self.retry_every_n_frames = retry_every_n_frames
        self.head_ratio = head_ratio

        self.detector = cv2.FaceDetectorYN.create(
            _ensure_model(YUNET_URL), "", (320, 320), 0.8, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(_ensure_model(SFACE_URL), "")

        self._names, self._gallery = self._load_gallery(Path(known_faces_dir))
        self._frame_count = 0
        self._locked: dict[int, str] = {}
        self._votes: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._last_try: dict[int, int] = {}

    # ---------- enrollment ----------
    def _load_gallery(self, root: Path):
        names, embeddings = [], []
        if not root.exists():
            print(f"[FaceIdentifier] '{root}' not found — face naming disabled.")
            return names, np.empty((0, 128), np.float32)

        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            name = (path.parent.name if path.parent != root else path.stem).replace("_", " ")

            img = cv2.imread(str(path))
            if img is None:
                print(f"[FaceIdentifier] Could not read {path}, skipped.")
                continue
            scale = 1280 / max(img.shape[:2])
            if scale < 1:                       # big phone photos: shrink for detection
                img = cv2.resize(img, None, fx=scale, fy=scale)

            face = self._largest_face(img)
            if face is None:
                print(f"[FaceIdentifier] No face found in {path.name}, skipped.")
                continue
            names.append(name)
            embeddings.append(self._embed(img, face))

        print(f"[FaceIdentifier] Enrolled {len(set(names))} people "
              f"from {len(names)} images: {sorted(set(names))}")
        gallery = (np.vstack(embeddings).astype(np.float32)
                   if embeddings else np.empty((0, 128), np.float32))
        return names, gallery

    # ---------- helpers ----------
    def _largest_face(self, img: np.ndarray):
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img)
        if faces is None or len(faces) == 0:
            return None
        return max(faces, key=lambda f: f[2] * f[3])

    def _embed(self, img: np.ndarray, face) -> np.ndarray:
        aligned = self.recognizer.alignCrop(img, face)
        feat = self.recognizer.feature(aligned).flatten()
        return feat / (np.linalg.norm(feat) + 1e-9)

    def _match(self, emb: np.ndarray) -> str | None:
        sims = self._gallery @ emb               # cosine similarity (both unit-length)
        best = int(np.argmax(sims))
        return self._names[best] if sims[best] >= self.match_threshold else None

    def _try_identify(self, frame: np.ndarray, track: Track) -> str | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in track.bbox)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        y_end = y1 + int((y2 - y1) * self.head_ratio)

        crop = frame[y1:y_end, x1:x2]
        if crop.shape[0] < 40 or crop.shape[1] < 40:
            return None
        face = self._largest_face(crop)
        if face is None:
            return None
        return self._match(self._embed(crop, face))

    # ---------- public API ----------
    def update(self, frame: np.ndarray, tracks: list[Track]) -> list[Track]:
        self._frame_count += 1
        if len(self._names) == 0:
            return tracks

        for t in tracks:
            tid = t.track_id
            if tid in self._locked:
                t.name = self._locked[tid]
                continue
            if self._frame_count - self._last_try.get(tid, -10_000) < self.retry_every_n_frames:
                continue

            self._last_try[tid] = self._frame_count
            name = self._try_identify(frame, t)
            if name:
                self._votes[tid][name] += 1
                if self._votes[tid][name] >= self.votes_needed:
                    self._locked[tid] = name
                    t.name = name
        return tracks