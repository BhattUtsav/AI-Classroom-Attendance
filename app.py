from __future__ import annotations

import re
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import numpy as np
import streamlit as st

from face_utils import (
    DEFAULT_COSINE_THRESHOLD,
    MIN_CANDIDATE_MARGIN,
    SFACE_EMBEDDINGS_PATH,
    SFACE_LABELS_PATH,
    SFaceRuntime,
    create_face_models,
    detect_face_rows,
    extract_embedding,
    model_files_exist,
    vision_models_exist,
)

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "attendance.db"
FACES_DIR = BASE_DIR / "face_samples"
MODEL_PATH = SFACE_EMBEDDINGS_PATH
LABELS_PATH = SFACE_LABELS_PATH
DEFAULT_CAMERA_URL = "http://10.128.201.119:8080/video"
DEFAULT_CONFIDENCE_LIMIT = DEFAULT_COSINE_THRESHOLD

st.set_page_config(
    page_title="AI Classroom | Smart Attendance",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A lightweight, built-in design system: no extra frontend packages required.
st.markdown(
    """
    <style>
      :root { color-scheme: dark; }
      .stApp {
        background: radial-gradient(ellipse at 10% 0%, rgba(31, 91, 117, .22), transparent 34%),
                    #0b1018;
        color: #e6edf6;
      }
      [data-testid="stHeader"] { background: rgba(11, 16, 24, .88); }
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111a27 0%, #0d131d 100%);
        border-right: 1px solid rgba(148, 163, 184, .14);
      }
      .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1440px; }
      h1, h2, h3 { letter-spacing: -.025em; }
      h1 { font-weight: 750 !important; }
      [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(24, 37, 53, .94), rgba(17, 27, 40, .92));
        border: 1px solid rgba(148, 163, 184, .18);
        padding: 18px 20px; border-radius: 16px;
        box-shadow: 0 12px 28px rgba(0,0,0,.12);
      }
      [data-testid="stMetricLabel"] { color: #a8b8cb; }
      [data-testid="stMetricValue"] { color: #f2f7ff; font-weight: 750; }
      .hero {
        background: linear-gradient(120deg, rgba(21, 54, 76, .96), rgba(18, 28, 45, .96) 58%, rgba(32, 43, 66, .94));
        border: 1px solid rgba(125, 211, 252, .22); border-radius: 22px;
        padding: 26px 28px; margin: 0 0 22px 0;
        box-shadow: 0 18px 44px rgba(0,0,0,.2);
      }
      .hero-kicker { color: #7dd3fc; text-transform: uppercase; font-size: .74rem; font-weight: 800; letter-spacing: .16em; }
      .hero-title { color: #f8fbff; font-size: clamp(1.65rem, 3vw, 2.35rem); line-height: 1.15; font-weight: 800; margin: 8px 0; }
      .hero-copy { color: #c0d0df; font-size: .98rem; max-width: 800px; margin: 0; }
      .section-note { color: #9fb0c3; font-size: .9rem; margin-top: -.45rem; }
      .pill {
        display: inline-block; padding: 5px 10px; border-radius: 999px;
        background: rgba(125, 211, 252, .10); border: 1px solid rgba(125, 211, 252, .24);
        color: #b9e8ff; font-size: .78rem; font-weight: 700; margin-top: 12px;
      }
      div.stButton > button, div.stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button {
        border-radius: 10px; min-height: 2.55rem; font-weight: 700;
        border: 1px solid rgba(125, 211, 252, .28); transition: all .15s ease;
      }
      div.stButton > button[kind="primary"], div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
        background: linear-gradient(100deg, #0284c7, #2563eb); border: 0; color: white;
      }
      div[data-testid="stTabs"] button { font-weight: 650; }
      div[data-testid="stDataFrame"] { border: 1px solid rgba(148,163,184,.16); border-radius: 12px; overflow: hidden; }
      div[data-testid="stAlert"] { border-radius: 12px; }
      .small-muted { color: #91a2b7; font-size: .82rem; }
      hr { border-color: rgba(148, 163, 184, .16); }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, timeout=10)


def create_tables() -> None:
    with get_connection() as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                attendance_date TEXT NOT NULL,
                attendance_time TEXT NOT NULL,
                status TEXT DEFAULT 'Present',
                FOREIGN KEY (student_id) REFERENCES students(student_id),
                UNIQUE (student_id, attendance_date)
            )
            """
        )


def load_students() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT student_id, name, department, created_at FROM students ORDER BY name",
            conn,
        )


def load_attendance() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT a.student_id AS 'Student ID', s.name AS 'Student Name',
                   s.department AS 'Department', a.attendance_date AS 'Date',
                   a.attendance_time AS 'Time', a.status AS 'Status'
            FROM attendance a
            LEFT JOIN students s ON a.student_id = s.student_id
            ORDER BY a.attendance_date DESC, a.attendance_time DESC
            """,
            conn,
        )


def student_folder(student_id: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", student_id)
    return FACES_DIR / safe_id


def open_phone_camera(camera_url: str) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(camera_url)
    # This property is not honored by every backend, but helps reduce stale frames where supported.
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera


def read_fresh_frame(camera: cv2.VideoCapture, discard: int = 4):
    """Drain a few buffered frames and return the newest valid frame."""
    latest = None
    for _ in range(max(1, discard + 1)):
        ok, frame = camera.read()
        if ok and frame is not None and frame.size:
            latest = frame
        elif latest is None:
            continue
    return latest


class LatestFrameCamera:
    """Continuously read the newest camera frame and reconnect after stream failures."""

    def __init__(self, camera_url: str):
        self.camera_url = camera_url.strip()
        self._capture = None
        self._lock = threading.Lock()
        # The condition wakes the AI worker only when a genuinely newer frame arrives.
        # This avoids copying full-resolution frames in a tight polling loop.
        self._condition = threading.Condition(self._lock)
        self._capture_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._latest_frame = None
        self._sequence = 0
        self._last_frame_time = 0.0
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="ip-webcam-frame-reader",
            daemon=True,
        )
        # Start the worker immediately. It will keep retrying if the phone stream
        # is temporarily unavailable instead of forcing the teacher to press Start.
        self._thread.start()

    def isOpened(self) -> bool:
        # Means that the camera worker is active, not that a network frame has
        # necessarily arrived yet. This permits automatic reconnects.
        return bool(self.camera_url) and not self._stop_event.is_set()

    def _open_capture(self):
        cap = cv2.VideoCapture()
        opened = False
        try:
            # FFmpeg timeouts prevent an unreachable IP Webcam URL from blocking
            # the whole reader indefinitely on OpenCV builds that support params.
            params = []
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                params += [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000]
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                params += [cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000]
            if params:
                opened = cap.open(self.camera_url, cv2.CAP_FFMPEG, params)
            else:
                opened = cap.open(self.camera_url)
        except (cv2.error, TypeError):
            opened = False
        if not opened or not cap.isOpened():
            # Compatibility fallback for OpenCV builds/backends that don't
            # support timeout parameters; use the default backend selection.
            try:
                cap.release()
            except Exception:
                pass
            cap = cv2.VideoCapture(self.camera_url)
            opened = cap.isOpened()
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not opened or not cap.isOpened():
            cap.release()
            return None
        return cap

    def _reader_loop(self) -> None:
        failures = 0
        while not self._stop_event.is_set():
            if self._capture is None or not self._capture.isOpened():
                cap = self._open_capture()
                if cap is None:
                    self._stop_event.wait(1.0)
                    continue
                with self._capture_lock:
                    self._capture = cap
                failures = 0

            with self._capture_lock:
                cap = self._capture
            if cap is None:
                self._stop_event.wait(0.1)
                continue
            try:
                ok, frame = cap.read()
            except Exception:
                ok, frame = False, None

            if ok and frame is not None and frame.size:
                with self._condition:
                    self._latest_frame = frame
                    self._sequence += 1
                    self._last_frame_time = time.monotonic()
                    self._condition.notify_all()
                failures = 0
            else:
                failures += 1
                if failures >= 3:
                    with self._capture_lock:
                        old_cap = self._capture
                        self._capture = None
                    if old_cap is not None:
                        try:
                            old_cap.release()
                        except Exception:
                            pass
                    failures = 0
                    self._stop_event.wait(0.5)
                else:
                    self._stop_event.wait(0.05)

    def read_latest(self, after_sequence: int = -1, timeout: float = 0.5):
        """Wait for a newer frame, then copy it exactly once for processing.

        The previous implementation copied a large frame every 20 ms even when
        no new frame was available. For MJPEG streams this wasteful polling could
        consume CPU and increase latency. This condition-based method wakes on a
        new frame and always returns the newest one, dropping obsolete frames.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            while self._sequence <= after_sequence and not self._stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            sequence = self._sequence
            age = time.monotonic() - self._last_frame_time if self._last_frame_time else None
            if self._latest_frame is not None and sequence > after_sequence:
                frame = self._latest_frame.copy()
            else:
                frame = None
        return frame, sequence, age

    def metadata(self):
        with self._lock:
            age = time.monotonic() - self._last_frame_time if self._last_frame_time else None
            return self._sequence, age, self._latest_frame is not None

    def release(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        with self._capture_lock:
            cap = self._capture
            self._capture = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if self._thread.is_alive():
            self._thread.join(timeout=1.5)


class LiveAttendanceEngine:
    """Background worker for face detection, matching, and verified attendance.

    Streamlit only renders snapshots from this object. Camera I/O and model
    inference never run in the Streamlit UI thread, and old frames are dropped.
    """

    def __init__(self, camera_url: str, recognizer, student_ids: list[str], name_map: dict[str, str],
                 confidence_limit: float, marked_today: set[str]):
        self.camera = LatestFrameCamera(camera_url)
        self.recognizer = recognizer
        self.student_ids = list(student_ids)
        self.name_map = dict(name_map)
        self.confidence_limit = float(confidence_limit)
        self.marked_today = set(marked_today)
        self._model_ready = bool(getattr(recognizer, "ready", False))
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread = None
        self._last_processed_sequence = -1
        self._annotated_jpeg = None
        self._faces = 0
        self._status = "Connecting to camera…"
        self._frame_age = None
        self._process_ms = 0.0
        self._fps = 0.0
        self._last_publish = 0.0
        self._stable: dict[str, dict[str, float | int]] = {}
        self._events: list[dict[str, str]] = []
        self._errors: list[str] = []

    def isOpened(self) -> bool:
        return self.camera.isOpened() and self._model_ready

    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._process_loop,
                name="ai-attendance-processor",
                daemon=True,
            )
            self._thread.start()

    def set_confidence_limit(self, value: float) -> None:
        with self._lock:
            self.confidence_limit = float(value)

    def snapshot(self) -> dict:
        _sequence, age, has_frame = self.camera.metadata()
        with self._lock:
            image = self._annotated_jpeg
            return {
                "image": image,
                "faces": self._faces,
                "status": self._status,
                "frame_age": age,
                "process_ms": self._process_ms,
                "fps": self._fps,
                "marked_today": set(self.marked_today),
                "events": list(self._events),
                "errors": list(self._errors),
                "has_frame": has_frame,
            }

    def stop(self) -> None:
        self._stop_event.set()
        self.camera.release()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        with self._lock:
            self._status = "Stopped"

    def _publish(self, image, face_count: int, status: str, started_at: float) -> None:
        # JPEG encoding can be relatively expensive. Do it outside the shared lock
        # so Streamlit snapshots and attendance updates never wait for image encoding.
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 60])
        finished = time.monotonic()
        elapsed = max(finished - started_at, 0.0001)
        jpeg_bytes = encoded.tobytes() if ok else None
        with self._lock:
            self._annotated_jpeg = jpeg_bytes
            self._faces = int(face_count)
            self._status = status
            self._process_ms = elapsed * 1000.0
            instant_fps = 1.0 / max(finished - self._last_publish, 0.0001) if self._last_publish else 0.0
            self._fps = instant_fps if not self._fps else (0.7 * self._fps + 0.3 * instant_fps)
            self._last_publish = finished

    def _record_attendance(self, student_id: str, name: str) -> str:
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        clock = now.strftime("%H:%M:%S")
        try:
            with sqlite3.connect(DB_PATH, timeout=8) as conn:
                conn.execute(
                    "INSERT INTO attendance (student_id, attendance_date, attendance_time, status) VALUES (?, ?, ?, 'Present')",
                    (student_id, today, clock),
                )
            event = {"Time": clock, "Student": name, "Event": "Attendance marked"}
            with self._lock:
                self.marked_today.add(student_id)
                self._events.insert(0, event)
                del self._events[10:]
            return f"{name} · PRESENT"
        except sqlite3.IntegrityError:
            with self._lock:
                self.marked_today.add(student_id)
            return f"{name} · ALREADY PRESENT"
        except sqlite3.Error as exc:
            with self._lock:
                self._errors.insert(0, f"Database write failed: {exc}")
                del self._errors[3:]
            return f"{name} · DB ERROR"

    def _process_loop(self) -> None:
        if not self._model_ready:
            with self._lock:
                self._status = "YuNet/SFace model unavailable"
            return
        last_sequence = -1
        last_process_time = 0.0
        # A modest 8–10 FPS ceiling is enough for stable 2-second confirmation.
        # The camera worker retains only the newest frame; this worker waits for a
        # new sequence instead of repeatedly copying old, full-resolution frames.
        min_interval = 0.10
        while not self._stop_event.is_set():
            remaining = min_interval - (time.monotonic() - last_process_time)
            if remaining > 0 and self._stop_event.wait(remaining):
                break
            frame, sequence, frame_age = self.camera.read_latest(
                after_sequence=last_sequence, timeout=0.5
            )
            if frame is None:
                with self._lock:
                    self._frame_age = frame_age
                    if frame_age is not None and frame_age > 2.5:
                        self._status = f"Camera stream stale ({frame_age:.1f}s)"
                    else:
                        self._status = "Waiting for a new camera frame…"
                continue

            started = time.monotonic()
            last_sequence = sequence
            last_process_time = started
            try:
                h, w = frame.shape[:2]
                # Do recognition on a smaller frame to reduce detector and embedding CPU cost.
                # 480 px keeps inference responsive for a nearby classroom camera.
                if w > 480:
                    new_h = max(1, int(h * 480 / w))
                    frame = cv2.resize(frame, (480, new_h), interpolation=cv2.INTER_AREA)
                faces = self.recognizer.detect(frame)
                frame_seen: set[str] = set()
                with self._lock:
                    limit = self.confidence_limit

                if len(faces) == 0:
                    cv2.putText(frame, "NO FACE DETECTED", (18, 36), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 165, 255), 2, cv2.LINE_AA)
                for face_row in faces:
                    x, y, bw, bh = [int(round(value)) for value in face_row[:4]]
                    x, y = max(0, x), max(0, y)
                    if bw < 40 or bh < 40 or x + bw > frame.shape[1] + 5 or y + bh > frame.shape[0] + 5:
                        continue
                    try:
                        match = self.recognizer.identify(frame, face_row)
                    except (cv2.error, ValueError) as exc:
                        with self._lock:
                            self._errors.insert(0, f"Face embedding failed: {exc}")
                            del self._errors[3:]
                        continue
                    sid = match.student_id
                    name = self.name_map.get(re.sub(r"[^A-Za-z0-9_-]", "_", sid)) if sid else None
                    similarity = float(match.similarity)
                    margin = float(match.margin)
                    accepted = bool(name and similarity >= limit and margin >= MIN_CANDIDATE_MARGIN)

                    if accepted:
                        frame_seen.add(sid)
                        with self._lock:
                            entry = self._stable.get(sid)
                            if entry is None:
                                entry = {"since": time.monotonic(), "hits": 0}
                                self._stable[sid] = entry
                            entry["hits"] = int(entry["hits"]) + 1
                            elapsed = time.monotonic() - float(entry["since"])
                            hits = int(entry["hits"])
                            already_present = sid in self.marked_today
                        if elapsed >= 2.0 and hits >= 5:
                            if not already_present:
                                status = self._record_attendance(sid, name)
                            else:
                                status = f"{name} · PRESENT"
                        else:
                            status = f"{name} · VERIFYING {min(elapsed, 2.0):.1f}/2.0s · sim {similarity:.2f} Δ{margin:.2f}"
                        color = (0, 200, 0)
                    elif name and (similarity >= limit - 0.07 or (similarity >= limit and margin < MIN_CANDIDATE_MARGIN)):
                        with self._lock:
                            self._stable.pop(sid, None)
                        status = f"POSSIBLE · {name} sim {similarity:.2f} Δ{margin:.2f}"
                        color = (0, 165, 255)
                    else:
                        with self._lock:
                            if sid:
                                self._stable.pop(sid, None)
                        status = f"UNKNOWN · sim {similarity:.2f}"
                        color = (0, 0, 255)

                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
                    cv2.putText(frame, status, (x, max(24, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, color, 2, cv2.LINE_AA)

                with self._lock:
                    for sid in list(self._stable):
                        if sid not in frame_seen:
                            self._stable.pop(sid, None)
                cv2.putText(frame, f"LIVE  {datetime.now().strftime('%H:%M:%S')}", (12, frame.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
                self._publish(frame, len(faces), "Live recognition active", started)
            except Exception as exc:
                with self._lock:
                    self._status = f"Recognition error: {exc}"
                    self._errors.insert(0, str(exc))
                    del self._errors[3:]
                self._stop_event.wait(0.1)

def load_recognizer():
    if not model_files_exist():
        return None, None, (
            "YuNet/SFace files or the SFace enrollment index are missing. Run "
            "`python download_sface_models.py`, then `python train_faces.py`."
        )
    try:
        runtime = SFaceRuntime()
    except (FileNotFoundError, RuntimeError, ValueError, cv2.error, OSError) as exc:
        return None, None, f"Could not load YuNet/SFace: {exc}"
    return runtime, runtime.student_ids, None


def recognize_and_mark_attendance(camera_url: str, confidence_limit: float = DEFAULT_CONFIDENCE_LIMIT):
    """One-shot SFace recognition; do not mark ambiguous or weak matches."""
    runtime, student_ids, error = load_recognizer()
    if error:
        return None, error
    with get_connection() as conn:
        students = conn.execute("SELECT student_id, name FROM students").fetchall()
    names = {re.sub(r"[^A-Za-z0-9_-]", "_", str(sid)): name for sid, name in students}
    camera = open_phone_camera(camera_url)
    if not camera.isOpened():
        camera.release()
        return None, "Cannot connect to phone camera. Check the URL and network."
    observations: list[tuple[str | None, float, float, tuple[int, int, int, int], object]] = []
    last_frame = None
    try:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and len(observations) < 15:
            frame = read_fresh_frame(camera, discard=1)
            if frame is None:
                time.sleep(0.04)
                continue
            h, w = frame.shape[:2]
            if w > 640:
                frame = cv2.resize(frame, (640, max(1, int(h * 640 / w))), interpolation=cv2.INTER_AREA)
            last_frame = frame
            faces = runtime.detect(frame)
            if len(faces):
                face = max(faces, key=lambda row: float(row[2] * row[3]))
                try:
                    result = runtime.identify(frame, face)
                    box = tuple(int(round(v)) for v in face[:4])
                    observations.append((result.student_id, result.similarity, result.margin, box, frame.copy()))
                except (cv2.error, ValueError):
                    pass
            time.sleep(0.05)
    finally:
        camera.release()
    if last_frame is None:
        return None, "No camera frame received. Check IP Webcam and try again."
    if not observations:
        cv2.putText(last_frame, "NO FACE DETECTED", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
        return last_frame, "No face detected. Face the camera and improve lighting."
    counts: dict[str | None, int] = {}
    for sid, _score, _margin, _box, _frame in observations:
        counts[sid] = counts.get(sid, 0) + 1
    best_id = max(counts, key=counts.get)
    group = [item for item in observations if item[0] == best_id]
    score = float(np.median([item[1] for item in group])) if group else -1.0
    margin = float(np.median([item[2] for item in group])) if group else 0.0
    agreement = len(group) / len(observations) if observations else 0.0
    _sid, _score, _margin, box, frame = group[-1] if group else observations[-1]
    name = names.get(re.sub(r"[^A-Za-z0-9_-]", "_", best_id)) if best_id else None
    x, y, w, h = box
    if not name or agreement < 0.65 or score < confidence_limit or margin < MIN_CANDIDATE_MARGIN:
        if name and score >= confidence_limit - 0.07:
            color = (0, 165, 255)
            text = f"POSSIBLE · {name} {score:.2f} Δ{margin:.2f}"
            message = f"Possible match for {name}; score={score:.2f}, margin={margin:.2f}, agreement={agreement:.0%}. No attendance marked."
        else:
            color = (0, 0, 255)
            text = f"UNKNOWN · sim {score:.2f}"
            message = f"No confident match (similarity={score:.2f}, margin={margin:.2f}, agreement={agreement:.0%}). No attendance marked."
    else:
        now = datetime.now()
        try:
            with get_connection() as conn:
                conn.execute(
                    "INSERT INTO attendance (student_id, attendance_date, attendance_time, status) VALUES (?, ?, ?, 'Present')",
                    (best_id, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")),
                )
            message = f"Attendance marked for {name} at {now.strftime('%H:%M:%S')} (SFace similarity {score:.2f})."
        except sqlite3.IntegrityError:
            message = f"{name} is already marked present today (SFace similarity {score:.2f})."
        color = (0, 200, 0)
        text = f"{name} · {score:.2f}"
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
    cv2.putText(frame, text, (x, max(28, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    return frame, message


@st.fragment(run_every=0.20)
def render_live_attendance() -> None:
    """Refresh only the live image/status frequently; keep heavy tables out of video refresh."""
    engine = st.session_state.get("live_engine")
    if engine is None or not st.session_state.get("live_active", False):
        return
    data = engine.snapshot()
    image = data["image"]
    if image is None:
        st.info(data["status"])
        if data["frame_age"] is not None and data["frame_age"] > 2.5:
            st.warning("No recent frame received. Check the phone camera and Wi-Fi.")
        return
    st.image(image, caption="Live camera · green = accepted candidate · amber = possible · red = unknown", width="stretch")
    if data["frame_age"] is not None and data["frame_age"] > 2.5:
        st.warning(f"The phone stream has not delivered a new frame for {data['frame_age']:.1f}s. Check IP Webcam/Wi-Fi.")
    elif data["status"].startswith("Recognition error"):
        st.error(data["status"])
    else:
        st.caption(f"{data['status']} · total processing {data['process_ms']:.0f} ms · {data['fps']:.1f} processed FPS")


@st.fragment(run_every=1.0)
def render_live_metrics() -> None:
    """Refresh metrics and event history once per second, not on every video frame."""
    engine = st.session_state.get("live_engine")
    if engine is None or not st.session_state.get("live_active", False):
        return
    data = engine.snapshot()
    col1, col2, col3 = st.columns(3)
    col1.metric("Faces detected", data["faces"])
    col2.metric("Students confirmed today", len(data["marked_today"]))
    col3.metric("Processing time", f"{data['process_ms']:.0f} ms")
    events = data["events"]
    if events:
        st.markdown("#### Recent live events")
        st.dataframe(pd.DataFrame(events), width="stretch", hide_index=True)
    errors = data["errors"]
    if errors:
        with st.expander("Recent diagnostics"):
            for err in errors:
                st.text(err)


create_tables()

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Computer Vision · Attendance Intelligence</div>
      <div class="hero-title">AI Smart Classroom</div>
      <p class="hero-copy">A clearer view of your classroom attendance — register students, enroll faces, review present and absent lists, and manage verified attendance from one place.</p>
      <span class="pill">● Local dashboard · Data stays on this computer</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    sidebar_today = datetime.now().strftime("%Y-%m-%d")
    sidebar_attendance = load_attendance()
    sidebar_today_count = int((sidebar_attendance["Date"] == sidebar_today).sum()) if not sidebar_attendance.empty else 0
    st.markdown("## 🎓 AI Classroom")
    st.caption("Attendance management")
    st.markdown("---")
    st.markdown("**Quick system check**")
    st.write(f"Students: **{len(load_students())}**")
    st.write(f"Today's records: **{sidebar_today_count}**")
    st.write("Face model: **SFace ready**" if model_files_exist() else "Face model: **Needs setup**")
    st.markdown("---")
    st.caption("Government Polytechnic, Junagadh · Computer Engineering")


tab1, tab2, tab3, tab4 = st.tabs(
    ["👤 Student Registration", "🧑‍💻 Face Enrollment", "📊 Attendance Dashboard", "🟢 Live Attendance"]
)

with tab1:
    st.subheader("Student registration")
    st.markdown('<p class="section-note">Add each student once using their unique enrollment number.</p>', unsafe_allow_html=True)
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        with st.form("student_form", clear_on_submit=True):
            student_id = st.text_input("Enrollment number", placeholder="e.g. 246190307024").strip()
            name = st.text_input("Full student name", placeholder="Enter name").strip()
            department = st.text_input("Department / class", value="Computer Engineering").strip()
            submitted = st.form_submit_button("＋ Register student", type="primary", width="stretch")
            if submitted:
                if not student_id or not name or not department:
                    st.error("Please fill in all fields.")
                else:
                    try:
                        with get_connection() as conn:
                            conn.execute(
                                "INSERT INTO students (student_id, name, department) VALUES (?, ?, ?)",
                                (student_id, name, department),
                            )
                        st.success(f"Registered {name} successfully.")
                        st.rerun()
                    except sqlite3.IntegrityError:
                        st.error("That enrollment number is already registered.")
    with right:
        students_df = load_students()
        st.metric("Registered students", len(students_df))
        if students_df.empty:
            st.info("No students yet. Register the first student using the form.")
        else:
            st.dataframe(students_df, width="stretch", hide_index=True)
            st.download_button(
                "⬇ Download student list (CSV)",
                data=students_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="students.csv",
                mime="text/csv",
            )

with tab2:
    st.subheader("Face enrollment")
    st.markdown('<p class="section-note">Capture varied, well-lit samples so the model can recognize the same person from normal camera angles.</p>', unsafe_allow_html=True)
    students_df = load_students()
    if students_df.empty:
        st.warning("Register at least one student before enrolling a face.")
    else:
        options = [(row["student_id"], row["name"]) for _, row in students_df.iterrows()]
        selected = st.selectbox("Student to enroll", options, format_func=lambda item: f"{item[1]} · {item[0]}")
        camera_url = st.text_input("Phone camera video URL", value=DEFAULT_CAMERA_URL, key="enrollment_camera_url")
        folder = student_folder(selected[0])
        existing_samples = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}] if folder.exists() else []
        existing_count = len(existing_samples)
        existing_numbers = []
        for sample_path in existing_samples:
            match = re.fullmatch(r"face_(\d+)\.(?:jpg|jpeg|png|bmp)", sample_path.name, re.IGNORECASE)
            if match:
                existing_numbers.append(int(match.group(1)))
        next_sample_number = max(existing_numbers, default=0) + 1
        col1, col2 = st.columns(2)
        col1.metric("Existing samples", existing_count)
        col2.metric("Recommended target", "2–3 real views")
        st.info("Capture one student at a time. Keep the face well lit and slowly turn slightly left/right during capture. Samples are aligned locally by YuNet; attendance is not marked during enrollment.")
        if st.button("📸 Capture 8 varied face samples", type="primary"):
            if not vision_models_exist():
                st.error("YuNet/SFace ONNX models missing. Run `python download_sface_models.py` in PowerShell first.")
            else:
                try:
                    detector, face_recognizer = create_face_models()
                except (FileNotFoundError, RuntimeError, cv2.error) as exc:
                    st.error(f"Cannot load YuNet/SFace: {exc}")
                    detector, face_recognizer = None, None
                camera = open_phone_camera(camera_url) if detector is not None and face_recognizer is not None else None
                if camera is not None and not camera.isOpened():
                    camera.release()
                    st.error("Cannot connect to the phone camera. Check IP Webcam and the URL.")
                elif camera is not None:
                    folder.mkdir(parents=True, exist_ok=True)
                    target_samples = 8
                    saved, attempts, preview = 0, 0, None
                    last_feature = None
                    last_save = 0.0
                    status = st.progress(0, text="Connecting to camera…")
                    instruction = st.empty()
                    instruction.info("Look at the camera, then slowly turn your head a little left and right. Avoid other faces in the frame.")
                    try:
                        while saved < target_samples and attempts < 300:
                            attempts += 1
                            frame = read_fresh_frame(camera, discard=0)
                            if frame is None:
                                time.sleep(0.04)
                                continue
                            h_frame, w_frame = frame.shape[:2]
                            if w_frame > 640:
                                frame = cv2.resize(frame, (640, max(1, int(h_frame * 640 / w_frame))), interpolation=cv2.INTER_AREA)
                            faces = detect_face_rows(detector, frame)
                            if len(faces) == 1:
                                face_row = faces[0]
                                x, y, fw, fh = [int(round(v)) for v in face_row[:4]]
                                if fw >= 50 and fh >= 50 and time.monotonic() - last_save >= 0.28:
                                    try:
                                        aligned = face_recognizer.alignCrop(frame, face_row)
                                        feature = extract_embedding(face_recognizer, aligned)
                                    except (cv2.error, ValueError):
                                        aligned = None
                                    if aligned is not None:
                                        duplicate = last_feature is not None and float(np.dot(last_feature, feature)) > 0.997
                                        if not duplicate:
                                            number = next_sample_number + saved
                                            dest = folder / f"face_{number:03d}.jpg"
                                            if cv2.imwrite(str(dest), aligned, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                                                saved += 1
                                                preview = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
                                                last_feature = feature
                                                last_save = time.monotonic()
                                                status.progress(min(saved / target_samples, 1.0), text=f"Captured {saved}/{target_samples} aligned samples")
                            elif len(faces) > 1:
                                instruction.warning("More than one face detected. Please keep only the selected student in frame.")
                            time.sleep(0.04)
                    finally:
                        camera.release()
                    if saved:
                        st.success(f"Saved {saved} aligned face samples for {selected[1]}.")
                        if preview is not None:
                            st.image(preview, caption="Last aligned enrollment sample", width=220)
                        st.warning("Rebuild the student embedding index: run `python train_faces.py` in PowerShell, then reload the dashboard.")
                    else:
                        st.error("No diverse face samples were captured. Face the camera, improve lighting, and slowly change your head angle.")



with tab3:
    st.subheader("Attendance dashboard")
    st.markdown(
        '<p class="section-note">Choose a date to see exactly who is present and who is absent. Absent status is calculated from the roster; it is not written as a separate attendance record.</p>',
        unsafe_allow_html=True,
    )
    students_df = load_students()
    attendance_df = load_attendance()
    selected_date = st.date_input("Attendance date", value=datetime.now().date(), key="attendance_view_date")
    selected_day = selected_date.strftime("%Y-%m-%d")
    daily = attendance_df[attendance_df["Date"].astype(str) == selected_day].copy() if not attendance_df.empty else attendance_df.copy()
    if not daily.empty:
        daily["Status"] = daily["Status"].fillna("Present").astype(str)
        # Only records explicitly marked Present count towards presence.
        present_ids = set(
            daily.loc[daily["Status"].str.strip().str.casefold() == "present", "Student ID"]
            .dropna().astype(str)
        )
    else:
        present_ids = set()

    if students_df.empty:
        st.info("No students are registered yet. Add students in Student Registration first.")
    else:
        roster = students_df.copy()
        roster["student_id"] = roster["student_id"].astype(str)
        present_roster = roster[roster["student_id"].isin(present_ids)].copy()
        absent_roster = roster[~roster["student_id"].isin(present_ids)].copy()
        time_by_id = {}
        if not daily.empty:
            for _, record in daily.iterrows():
                if str(record.get("Status", "Present")).strip().casefold() == "present":
                    time_by_id[str(record["Student ID"])] = str(record["Time"])

        present_list = present_roster[["student_id", "name", "department"]].rename(
            columns={"student_id": "Student ID", "name": "Student Name", "department": "Department"}
        )
        present_list["Status"] = "Present"
        present_list["Time"] = present_list["Student ID"].map(time_by_id).fillna("—")
        present_list = present_list[["Student ID", "Student Name", "Department", "Status", "Time"]].sort_values("Student Name")

        absent_list = absent_roster[["student_id", "name", "department"]].rename(
            columns={"student_id": "Student ID", "name": "Student Name", "department": "Department"}
        )
        absent_list["Status"] = "Absent"
        absent_list["Time"] = "—"
        absent_list = absent_list[["Student ID", "Student Name", "Department", "Status", "Time"]].sort_values("Student Name")

        total_students = len(roster)
        present_count = len(present_list)
        absent_count = len(absent_list)
        attendance_pct = (present_count / total_students * 100) if total_students else 0.0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total students", total_students)
        c2.metric("Present", present_count)
        c3.metric("Absent", absent_count)
        c4.metric("Attendance rate", f"{attendance_pct:.0f}%")

        present_tab, absent_tab = st.tabs([f"✅ Present ({present_count})", f"⏳ Absent ({absent_count})"])
        with present_tab:
            if present_list.empty:
                st.info(f"No students are marked present for {selected_day} yet.")
            else:
                st.dataframe(present_list, width="stretch", hide_index=True)
        with absent_tab:
            if absent_list.empty:
                st.success(f"All registered students are marked present for {selected_day}.")
            else:
                st.dataframe(absent_list, width="stretch", hide_index=True)

        daily_roll_call = pd.concat([present_list, absent_list], ignore_index=True)
        st.download_button(
            "⬇ Download daily present/absent list (CSV)",
            data=daily_roll_call.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"attendance-{selected_day}.csv",
            mime="text/csv",
        )

    st.markdown("#### Attendance history")
    if attendance_df.empty:
        st.info("Attendance records will appear here after a student is verified.")
    else:
        query = st.text_input("Search attendance history by name or enrollment number", key="attendance_search")
        shown = attendance_df
        if query.strip():
            match = attendance_df["Student Name"].fillna("").str.contains(query, case=False, regex=False) | attendance_df["Student ID"].fillna("").str.contains(query, case=False, regex=False)
            shown = attendance_df[match]
        st.dataframe(shown, width="stretch", hide_index=True)
        st.download_button("⬇ Export attendance history (CSV)", data=shown.to_csv(index=False).encode("utf-8-sig"), file_name="attendance-history.csv", mime="text/csv")
    if st.button("↻ Refresh attendance dashboard"):
        st.rerun()

with tab4:
    st.subheader("Always-on live face recognition & attendance")
    st.markdown(
        '<p class="section-note">Monitoring starts automatically when this dashboard opens. Camera capture and AI recognition run in background workers; temporary stream failures are retried automatically.</p>',
        unsafe_allow_html=True,
    )
    if not model_files_exist():
        st.error("SFace setup is incomplete. Run `python download_sface_models.py`, then `python train_faces.py`, and reload this page.")
        st.code("python download_sface_models.py\npython train_faces.py")
    else:
        camera_url = st.text_input("Phone camera URL", value=DEFAULT_CAMERA_URL, key="live_camera_url")
        confidence_limit = st.slider(
            "Minimum SFace cosine similarity", min_value=0.30, max_value=0.75,
            value=float(st.session_state.get("live_sface_similarity_threshold", DEFAULT_COSINE_THRESHOLD)), step=0.01,
            key="live_sface_similarity_threshold",
            help="Higher means stricter matching. A green result also needs a clear margin over the second-best student. Calibrate with real enrolled and unknown faces.",
        )
        st.caption("SFace compares face embeddings. Only clear matches with a sufficient score and a margin over the next student can be verified.")

        engine = st.session_state.get("live_engine")
        engine_url = st.session_state.get("live_engine_url")
        reconnect_clicked = st.button("↻ Apply URL / reconnect camera", width="content")
        if reconnect_clicked:
            if engine is not None:
                engine.stop()
            st.session_state["live_engine"] = None
            st.session_state["live_engine_url"] = None
            engine = None
            engine_url = None

        # Auto-start the background pipeline without requiring a Start Live button.
        # Existing engine is retained across Streamlit reruns and tab switches.
        if engine is None:
            recognizer, student_ids, model_error = load_recognizer()
            if model_error:
                st.error(model_error)
            else:
                with get_connection() as conn:
                    rows = conn.execute("SELECT student_id, name FROM students").fetchall()
                    today_rows = conn.execute(
                        "SELECT student_id FROM attendance WHERE attendance_date = ?",
                        (datetime.now().strftime("%Y-%m-%d"),),
                    ).fetchall()
                name_map = {
                    re.sub(r"[^A-Za-z0-9_-]", "_", sid): name
                    for sid, name in rows
                }
                engine = LiveAttendanceEngine(
                    camera_url.strip(), recognizer, student_ids, name_map,
                    float(confidence_limit), {row[0] for row in today_rows},
                )
                engine.start()
                st.session_state["live_engine"] = engine
                st.session_state["live_engine_url"] = camera_url.strip()
                st.session_state["live_active"] = True
                engine_url = camera_url.strip()
        elif engine_url != camera_url.strip():
            st.warning("Camera URL changed. Click **Apply URL / reconnect camera** to switch the running monitor to the new URL.")

        engine = st.session_state.get("live_engine")
        if engine is not None:
            engine.set_confidence_limit(float(confidence_limit))
            data = engine.snapshot()
            if data["has_frame"]:
                st.success("Automatic attendance monitoring is running. No Start button is needed; this worker keeps retrying the camera if the stream drops.")
            else:
                st.info("Automatic monitoring is running and trying to connect to the phone camera…")
            render_live_attendance()
            render_live_metrics()
        else:
            st.warning("Monitoring could not start because the face model is unavailable. Train the model, then reload the dashboard.")

        st.markdown("#### Today's attendance records")
        current_day = datetime.now().strftime("%Y-%m-%d")
        with get_connection() as conn:
            live_today = pd.read_sql_query(
                """SELECT a.student_id AS 'Student ID', s.name AS 'Student Name',
                          a.attendance_date AS 'Date', a.attendance_time AS 'Time', a.status AS 'Status'
                   FROM attendance a LEFT JOIN students s ON a.student_id = s.student_id
                   WHERE a.attendance_date = ? ORDER BY a.attendance_time DESC""",
                conn, params=(current_day,),
            )
        st.metric("Verified present today", live_today["Student ID"].nunique() if not live_today.empty else 0)
        st.dataframe(live_today, width="stretch", hide_index=True)

st.markdown("---")
st.markdown('<p class="small-muted">AI Smart Classroom Attendance · Local database: attendance.db · Check ambiguous matches before relying on results.</p>', unsafe_allow_html=True)
