from __future__ import annotations

import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import cv2

from face_utils import (
    DEFAULT_COSINE_THRESHOLD,
    MIN_CANDIDATE_MARGIN,
    SFaceRuntime,
    model_files_exist,
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "attendance.db"
DEFAULT_CAMERA_URL = "http://10.128.201.119:8080/video"
SIMILARITY_THRESHOLD = DEFAULT_COSINE_THRESHOLD
REQUIRED_SECONDS = 2.0


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(value))


def main() -> None:
    if not model_files_exist():
        raise SystemExit("SFace setup missing. Run python download_sface_models.py, then python train_faces.py.")
    runtime = SFaceRuntime()
    with sqlite3.connect(DB_PATH) as conn:
        names = {safe_id(sid): name for sid, name in conn.execute("SELECT student_id, name FROM students")}
        today = datetime.now().strftime("%Y-%m-%d")
        marked_today = {row[0] for row in conn.execute("SELECT student_id FROM attendance WHERE attendance_date = ?", (today,))}

    url = input(f"Enter phone video URL [{DEFAULT_CAMERA_URL}]: ").strip() or DEFAULT_CAMERA_URL
    camera = cv2.VideoCapture(url)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        camera.release()
        raise SystemExit("Cannot connect to camera. Check IP Webcam, URL, and Wi-Fi.")

    stable: dict[str, tuple[float, int]] = {}
    print("SFace live recognition test started. Press Q to quit.")
    print(f"Cosine similarity threshold: {SIMILARITY_THRESHOLD:.2f}; required best-vs-second margin: {MIN_CANDIDATE_MARGIN:.2f}.")
    print("A stable match must remain consistent for 2 seconds before attendance is recorded.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                time.sleep(0.03)
                continue
            h, w = frame.shape[:2]
            if w > 640:
                frame = cv2.resize(frame, (640, max(1, int(h * 640 / w))), interpolation=cv2.INTER_AREA)
            try:
                faces = runtime.detect(frame)
            except cv2.error as exc:
                print(f"Detector error: {exc}")
                time.sleep(0.1)
                continue
            visible: set[str] = set()
            now_mono = time.monotonic()
            if len(faces) == 0:
                cv2.putText(frame, "NO FACE DETECTED", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            for face in faces:
                x, y, bw, bh = [int(round(v)) for v in face[:4]]
                try:
                    match = runtime.identify(frame, face)
                except (cv2.error, ValueError) as exc:
                    print(f"Embedding error: {exc}")
                    continue
                sid = match.student_id
                name = names.get(safe_id(sid)) if sid else None
                score, margin = float(match.similarity), float(match.margin)
                accepted = bool(name and score >= SIMILARITY_THRESHOLD and margin >= MIN_CANDIDATE_MARGIN)
                if accepted:
                    visible.add(sid)
                    since, hits = stable.get(sid, (now_mono, 0))
                    hits += 1
                    stable[sid] = (since, hits)
                    if now_mono - since >= REQUIRED_SECONDS and hits >= 5:
                        if sid not in marked_today:
                            now = datetime.now()
                            try:
                                with sqlite3.connect(DB_PATH, timeout=8) as conn:
                                    conn.execute("INSERT INTO attendance (student_id, attendance_date, attendance_time, status) VALUES (?, ?, ?, 'Present')", (sid, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")))
                                marked_today.add(sid)
                                print(f"Attendance marked: {name} at {now.strftime('%H:%M:%S')}")
                            except sqlite3.IntegrityError:
                                marked_today.add(sid)
                        label, color = f"{name} · PRESENT" if sid in marked_today else f"{name} · VERIFYING", (0, 200, 0)
                    else:
                        label, color = f"{name} · VERIFYING {min(now_mono - since, REQUIRED_SECONDS):.1f}/2s", (0, 200, 0)
                elif name and score >= SIMILARITY_THRESHOLD - 0.07:
                    stable.pop(sid, None)
                    label, color = f"POSSIBLE · {name} {score:.2f} Δ{margin:.2f}", (0, 165, 255)
                else:
                    if sid:
                        stable.pop(sid, None)
                    label, color = f"UNKNOWN · sim {score:.2f}", (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
                cv2.putText(frame, label, (x, max(24, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 2, cv2.LINE_AA)
            for sid in list(stable):
                if sid not in visible:
                    stable.pop(sid, None)
            cv2.putText(frame, f"LIVE {datetime.now().strftime('%H:%M:%S')}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.imshow("AI Classroom Attendance — SFace — Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
