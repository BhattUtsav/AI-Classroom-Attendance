"""Local YuNet + SFace utilities for classroom face recognition.

Models are downloaded from the official OpenCV Zoo model repository by
`python download_sface_models.py`. No image data is sent to a third party.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
SFACE_EMBEDDINGS_PATH = MODELS_DIR / "sface_embeddings.npz"
SFACE_LABELS_PATH = MODELS_DIR / "sface_labels.txt"

# OpenCV's SFace reference cosine threshold is 0.363 for a pairwise verification
# benchmark. Classroom open-set identification needs local calibration, so the UI
# starts more conservatively and also requires a margin over the second candidate.
DEFAULT_COSINE_THRESHOLD = 0.45
MIN_CANDIDATE_MARGIN = 0.06
DETECTION_THRESHOLD = 0.82
EMBEDDING_SIZE = 112


@dataclass(frozen=True)
class MatchResult:
    student_id: str | None
    similarity: float
    margin: float
    second_student_id: str | None
    second_similarity: float


def model_files_exist() -> bool:
    return (
        YUNET_MODEL_PATH.is_file()
        and SFACE_MODEL_PATH.is_file()
        and SFACE_EMBEDDINGS_PATH.is_file()
        and SFACE_LABELS_PATH.is_file()
    )


def vision_models_exist() -> bool:
    return YUNET_MODEL_PATH.is_file() and SFACE_MODEL_PATH.is_file()


def create_face_models():
    """Return (YuNet detector, SFace recognizer), raising a helpful error if missing."""
    missing = [str(p) for p in (YUNET_MODEL_PATH, SFACE_MODEL_PATH) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "SFace model files are missing. Run `python download_sface_models.py` first. "
            "Missing: " + ", ".join(missing)
        )
    if not hasattr(cv2, "FaceDetectorYN_create") or not hasattr(cv2, "FaceRecognizerSF_create"):
        raise RuntimeError(
            "This OpenCV build lacks YuNet/SFace APIs. Install opencv-contrib-python 4.8 or newer."
        )
    detector = cv2.FaceDetectorYN_create(
        str(YUNET_MODEL_PATH), "", (320, 320), DETECTION_THRESHOLD, 0.30, 5000
    )
    recognizer = cv2.FaceRecognizerSF_create(str(SFACE_MODEL_PATH), "")
    return detector, recognizer


def detect_face_rows(detector: Any, frame: np.ndarray) -> np.ndarray:
    """Detect faces in a BGR frame and return YuNet 15-column rows."""
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return np.empty((0, 15), dtype=np.float32)
    height, width = frame.shape[:2]
    detector.setInputSize((int(width), int(height)))
    _result, faces = detector.detect(frame)
    if faces is None:
        return np.empty((0, 15), dtype=np.float32)
    return np.asarray(faces, dtype=np.float32).reshape(-1, 15)


def normalize_embedding(feature: np.ndarray) -> np.ndarray:
    vector = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("SFace produced an invalid face embedding")
    return vector / norm


def extract_aligned_face(recognizer: Any, image: np.ndarray, face_row: np.ndarray) -> np.ndarray:
    """Align a detected face using YuNet's landmarks for SFace."""
    aligned = recognizer.alignCrop(image, np.asarray(face_row, dtype=np.float32).reshape(15))
    if aligned is None or aligned.size == 0:
        raise ValueError("Could not align detected face")
    return aligned


def extract_embedding(recognizer: Any, aligned_face: np.ndarray) -> np.ndarray:
    return normalize_embedding(recognizer.feature(aligned_face))


def prepare_sample_image(
    image: np.ndarray, detector: Any, recognizer: Any
) -> tuple[np.ndarray | None, str]:
    """Return an aligned 112x112 face image and a method note.

    New enrollments are stored aligned. Older 200x200 Haar crops are kept usable
    by resizing their central square if YuNet cannot redetect landmarks. For larger
    images, no-face samples are skipped instead of embedding a whole scene.
    """
    if image is None or image.size == 0:
        return None, "empty image"
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        return None, "unsupported image format"

    h, w = image.shape[:2]
    # Keep very large source photos from making detector inference unnecessarily slow.
    if max(w, h) > 1280:
        ratio = 1280.0 / max(w, h)
        image = cv2.resize(image, (max(1, int(w * ratio)), max(1, int(h * ratio))), interpolation=cv2.INTER_AREA)
        h, w = image.shape[:2]
    # Enrollment images written by this SFace version are already aligned 112x112.
    # Only treat exact canonical size as aligned; older 200x200 crops go through YuNet.
    if (w, h) == (EMBEDDING_SIZE, EMBEDDING_SIZE):
        return cv2.resize(image, (EMBEDDING_SIZE, EMBEDDING_SIZE)), "already aligned"

    try:
        faces = detect_face_rows(detector, image)
    except cv2.error:
        faces = np.empty((0, 15), dtype=np.float32)

    if len(faces):
        faces = sorted(faces, key=lambda row: float(row[2] * row[3]), reverse=True)
        if len(faces) > 1:
            largest_area = float(faces[0][2] * faces[0][3])
            second_area = float(faces[1][2] * faces[1][3])
            # Avoid attaching a multi-face image to the wrong student's identity.
            if second_area >= largest_area * 0.55:
                return None, "multiple faces of similar size; skipped as ambiguous"
        try:
            return extract_aligned_face(recognizer, image, faces[0]), "YuNet aligned"
        except (cv2.error, ValueError) as exc:
            return None, f"alignment failed: {exc}"

    # Legacy project samples were saved as square grayscale Haar crops (usually 200x200).
    # Keep that data usable, while explicitly reporting the weaker fallback.
    if max(w, h) <= 320 and 0.72 <= w / max(h, 1) <= 1.39:
        side = min(w, h)
        x0 = max(0, (w - side) // 2)
        y0 = max(0, (h - side) // 2)
        square = image[y0:y0 + side, x0:x0 + side]
        return cv2.resize(square, (EMBEDDING_SIZE, EMBEDDING_SIZE), interpolation=cv2.INTER_AREA), "legacy crop fallback"

    return None, "no face detected"


class SFaceRuntime:
    """YuNet detector + SFace embeddings enrolled locally for each student."""

    def __init__(self):
        if not model_files_exist():
            raise FileNotFoundError(
                "SFace model/index not ready. Run `python download_sface_models.py` and "
                "then `python train_faces.py`."
            )
        self.detector, self.recognizer = create_face_models()
        with np.load(SFACE_EMBEDDINGS_PATH, allow_pickle=False) as data:
            self.template_ids = np.asarray(data["student_ids"], dtype=str)
            self.embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        if self.embeddings.ndim != 2 or len(self.template_ids) != len(self.embeddings) or len(self.embeddings) == 0:
            raise ValueError("SFace embedding index is invalid. Run `python train_faces.py` again.")
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / np.maximum(norms, 1e-8)
        self.student_ids = sorted(set(self.template_ids.tolist()))
        self.ready = True

    def detect(self, frame: np.ndarray) -> np.ndarray:
        return detect_face_rows(self.detector, frame)

    def identify(self, frame: np.ndarray, face_row: np.ndarray) -> MatchResult:
        aligned = extract_aligned_face(self.recognizer, frame, face_row)
        query = extract_embedding(self.recognizer, aligned)
        similarities = self.embeddings @ query
        per_student: list[tuple[str, float]] = []
        for sid in self.student_ids:
            sample_scores = similarities[self.template_ids == sid]
            if sample_scores.size == 0:
                continue
            # Average the top two templates so students with many samples don't win
            # solely because they have more enrolled images than other students.
            top = np.sort(sample_scores)[::-1][: min(2, sample_scores.size)]
            per_student.append((sid, float(np.mean(top))))
        per_student.sort(key=lambda item: item[1], reverse=True)
        if not per_student:
            return MatchResult(None, -1.0, 0.0, None, -1.0)
        best_id, best_score = per_student[0]
        if len(per_student) > 1:
            second_id, second_score = per_student[1]
            margin = best_score - second_score
        else:
            second_id, second_score, margin = None, -1.0, 0.0
        return MatchResult(best_id, best_score, margin, second_id, second_score)
