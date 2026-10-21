from __future__ import annotations

import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "attendance.db"
FACES_DIR = BASE_DIR / "face_samples"
MODEL_PATH = BASE_DIR / "models" / "lbph_model.yml"
LABELS_PATH = BASE_DIR / "models" / "labels.txt"
DEFAULT_CAMERA_URL = "http://10.128.201.119:8080/video"
DEFAULT_CONFIDENCE_LIMIT = 80.0

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


def load_recognizer():
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        return None, None, "Face model files not found. Capture samples and train the model first."
    student_ids = [line.strip() for line in LABELS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not student_ids:
        return None, None, "The model label file is empty. Retrain the model."
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))
    return recognizer, student_ids, None


def recognize_and_mark_attendance(camera_url: str, confidence_limit: float = DEFAULT_CONFIDENCE_LIMIT):
    """Capture several recent frames and require a stable, acceptable LBPH match."""
    recognizer, student_ids, error = load_recognizer()
    if error:
        return None, error

    with get_connection() as conn:
        students = conn.execute("SELECT student_id, name FROM students").fetchall()
    names = {
        re.sub(r"[^A-Za-z0-9_-]", "_", student_id): name
        for student_id, name in students
    }

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if detector.empty():
        return None, "OpenCV could not load the face detector."

    camera = open_phone_camera(camera_url)
    if not camera.isOpened():
        camera.release()
        return None, "Cannot connect to phone camera. Check the URL and network."

    observations: list[tuple[int, float, tuple[int, int, int, int], object]] = []
    last_frame = None
    try:
        # The first frames from IP Webcam can be stale. Use several recent frames.
        deadline = time.monotonic() + 2.5
        attempts = 0
        while time.monotonic() < deadline and attempts < 18:
            attempts += 1
            frame = read_fresh_frame(camera, discard=1)
            if frame is None:
                time.sleep(0.04)
                continue
            last_frame = frame
            gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detection_gray = cv2.equalizeHist(gray_raw)
            faces = detector.detectMultiScale(
                detection_gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )
            if len(faces) == 0:
                time.sleep(0.03)
                continue
            x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
            # Apply the same crop-level normalization used during training.
            face = cv2.equalizeHist(gray_raw[y:y + h, x:x + w])
            face = cv2.resize(face, (200, 200))
            label, confidence = recognizer.predict(face)
            observations.append((int(label), float(confidence), (int(x), int(y), int(w), int(h)), frame.copy()))
            time.sleep(0.04)
    finally:
        camera.release()

    if last_frame is None:
        return None, "No camera frame received. Check IP Webcam and try again."
    if not observations:
        cv2.putText(last_frame, "NO FACE DETECTED", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
        return last_frame, "No face detected. Face the camera, improve lighting, and try again."

    # Use the most frequent predicted label and its median distance to reduce single-frame noise.
    label_counts: dict[int, int] = {}
    for label, _confidence, _box, _frame in observations:
        label_counts[label] = label_counts.get(label, 0) + 1
    best_label = max(label_counts, key=label_counts.get)
    matching = [item for item in observations if item[0] == best_label]
    median_confidence = sorted(item[1] for item in matching)[len(matching) // 2]
    x, y, w, h = matching[-1][2]
    frame = matching[-1][3]

    is_valid_label = 0 <= best_label < len(student_ids)
    sid = student_ids[best_label] if is_valid_label else None
    name = names.get(re.sub(r"[^A-Za-z0-9_-]", "_", sid)) if sid else None
    agreement = len(matching) / len(observations)

    if not is_valid_label or name is None:
        color = (0, 0, 255)
        label_text = "Unknown face"
        message = "No registered student match. No attendance marked."
    elif agreement < 0.60:
        color = (0, 165, 255)
        label_text = f"Unstable match: {name}"
        message = f"Unstable match for {name} ({agreement:.0%} frame agreement). No attendance marked; capture again."
    elif median_confidence > confidence_limit:
        if median_confidence <= confidence_limit + 15:
            color = (0, 165, 255)
            label_text = f"Possible: {name}"
            message = (
                f"Possible match: {name} (LBPH distance {median_confidence:.1f}; limit {confidence_limit:.0f}). "
                "Not marked—improve lighting or capture more face samples."
            )
        else:
            color = (0, 0, 255)
            label_text = "Unknown / low match"
            message = (
                f"No confident match (best candidate {name}, LBPH distance {median_confidence:.1f}). "
                "No attendance marked."
            )
    else:
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")
        try:
            with get_connection() as conn:
                conn.execute(
                    "INSERT INTO attendance (student_id, attendance_date, attendance_time, status) VALUES (?, ?, ?, 'Present')",
                    (sid, today, current_time),
                )
            message = f"Attendance marked for {name} at {current_time}. Match distance {median_confidence:.1f}."
        except sqlite3.IntegrityError:
            message = f"{name} is already marked present today. Match distance {median_confidence:.1f}."
        color = (0, 200, 0)
        label_text = f"{name} | {median_confidence:.1f}"

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
    cv2.putText(frame, label_text, (x, max(28, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2)
    return frame, message


@st.fragment(run_every=0.4)
def render_live_attendance() -> None:
    """Refresh only the live-camera area repeatedly while a session is active."""
    if not st.session_state.get("live_active", False):
        return

    camera = st.session_state.get("live_camera")
    recognizer = st.session_state.get("live_recognizer")
    student_ids = st.session_state.get("live_student_ids", [])
    name_map = st.session_state.get("live_name_map", {})
    detector = st.session_state.get("live_detector")
    if camera is None or recognizer is None or detector is None:
        st.error("The live camera session is not initialized. Stop and start it again.")
        return

    frame = read_fresh_frame(camera, discard=2)
    if frame is None or not frame.size:
        failures = int(st.session_state.get("live_read_failures", 0)) + 1
        st.session_state["live_read_failures"] = failures
        st.warning(f"Waiting for camera frames… ({failures}/15). Check that IP Webcam is still streaming.")
        if failures >= 15:
            st.session_state["live_active"] = False
            camera.release()
            st.session_state["live_camera"] = None
            st.error("Camera stream stopped. Start live attendance again after checking the phone camera.")
        return

    st.session_state["live_read_failures"] = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detection_gray = cv2.equalizeHist(gray)
    faces = detector.detectMultiScale(
        detection_gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    now_mono = time.monotonic()
    seen_valid: set[str] = set()
    limit = float(st.session_state.get("live_confidence_limit", DEFAULT_CONFIDENCE_LIMIT))
    stable = st.session_state.setdefault("live_stable", {})
    marked_today: set[str] = st.session_state.setdefault("live_marked_today", set())
    event_log: list[dict[str, str]] = st.session_state.setdefault("live_event_log", [])

    if len(faces) == 0:
        cv2.putText(frame, "NO FACE DETECTED", (22, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 165, 255), 2)

    for (x, y, w, h) in faces:
        crop = cv2.equalizeHist(gray[y:y + h, x:x + w])
        face_img = cv2.resize(crop, (200, 200))
        label_index, distance = recognizer.predict(face_img)
        valid_label = 0 <= int(label_index) < len(student_ids)
        student_id = student_ids[int(label_index)] if valid_label else None
        name = name_map.get(re.sub(r"[^A-Za-z0-9_-]", "_", student_id)) if student_id else None

        if name and distance <= limit:
            seen_valid.add(student_id)
            entry = stable.get(student_id)
            if entry is None:
                entry = {"since": now_mono, "hits": 0}
                stable[student_id] = entry
            entry["hits"] = int(entry.get("hits", 0)) + 1
            elapsed = now_mono - float(entry["since"])
            if elapsed >= 2.0 and int(entry["hits"]) >= 5:
                present_already = student_id in marked_today
                if not present_already:
                    current = datetime.now()
                    try:
                        with get_connection() as conn:
                            conn.execute(
                                "INSERT INTO attendance (student_id, attendance_date, attendance_time, status) VALUES (?, ?, ?, 'Present')",
                                (student_id, current.strftime("%Y-%m-%d"), current.strftime("%H:%M:%S")),
                            )
                        present_already = False
                        event_log.insert(0, {"Time": current.strftime("%H:%M:%S"), "Student": name, "Event": "Attendance marked"})
                    except sqlite3.IntegrityError:
                        present_already = True
                        event_log.insert(0, {"Time": current.strftime("%H:%M:%S"), "Student": name, "Event": "Already present today"})
                    marked_today.add(student_id)
                    del event_log[10:]
                status = f"{name} · PRESENT" if present_already else f"{name} · PRESENT"
            else:
                status = f"{name} · VERIFYING {min(elapsed, 2.0):.1f}/2.0s"
            color = (0, 200, 0)
        elif name and distance <= limit + 15:
            stable.pop(student_id, None)
            status = f"POSSIBLE: {name} · {distance:.1f}"
            color = (0, 165, 255)
        else:
            if student_id:
                stable.pop(student_id, None)
            status = f"UNKNOWN · {distance:.1f}"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        text_y = max(28, y - 10)
        cv2.putText(frame, status, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)

    # Require uninterrupted acceptable matches for a student before marking attendance.
    for student_id in list(stable):
        if student_id not in seen_valid:
            stable.pop(student_id, None)

    cv2.putText(frame, f"LIVE  |  {datetime.now().strftime('%H:%M:%S')}", (18, frame.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Live camera — green: accepted match · amber: uncertain · red: unknown", use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Faces detected now", int(len(faces)))
    col2.metric("Students confirmed today", len(marked_today))
    col3.metric("LBPH limit", f"{limit:.0f}")

    if event_log:
        st.markdown("#### Recent live events")
        st.dataframe(pd.DataFrame(event_log), use_container_width=True, hide_index=True)


create_tables()

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Computer Vision · Attendance Intelligence</div>
      <div class="hero-title">AI Smart Classroom</div>
      <p class="hero-copy">A clearer view of your classroom attendance — register students, enroll faces, review camera frames, and manage verified attendance from one place.</p>
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
    st.write("Face model: **Ready**" if MODEL_PATH.exists() and LABELS_PATH.exists() else "Face model: **Needs training**")
    st.markdown("---")
    st.caption("Government Polytechnic, Junagadh · Computer Engineering")


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["👤 Student Registration", "🧑‍💻 Face Enrollment", "📷 Camera Preview", "📊 Attendance Dashboard", "🟢 Live Attendance"]
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
            submitted = st.form_submit_button("＋ Register student", type="primary", use_container_width=True)
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
            st.dataframe(students_df, use_container_width=True, hide_index=True)
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
        existing_samples = list(folder.glob("*.jpg")) if folder.exists() else []
        existing_count = len(existing_samples)
        existing_numbers = []
        for sample_path in existing_samples:
            match = re.fullmatch(r"face_(\d+)\.jpg", sample_path.name)
            if match:
                existing_numbers.append(int(match.group(1)))
        next_sample_number = max(existing_numbers, default=0) + 1
        col1, col2 = st.columns(2)
        col1.metric("Existing samples", existing_count)
        col2.metric("Recommended target", "25–60")
        st.info("Keep one face in frame. Include slight head turns and normal lighting changes. Enrollment stores local grayscale face samples; it does not itself mark attendance.")
        if st.button("📸 Capture 25 face samples", type="primary"):
            detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            camera = open_phone_camera(camera_url)
            if detector.empty():
                st.error("Could not load the OpenCV face detector.")
            elif not camera.isOpened():
                camera.release()
                st.error("Cannot connect to the phone camera. Check IP Webcam and the URL.")
            else:
                folder.mkdir(parents=True, exist_ok=True)
                saved, attempts, preview = 0, 0, None
                status = st.progress(0, text="Connecting to camera…")
                try:
                    while saved < 25 and attempts < 160:
                        attempts += 1
                        frame = read_fresh_frame(camera, discard=1)
                        if frame is None:
                            time.sleep(0.08)
                            continue
                        gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        detection_gray = cv2.equalizeHist(gray_raw)
                        faces = detector.detectMultiScale(detection_gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
                        if len(faces) > 0:
                            x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
                            # Save the grayscale face crop; training and prediction normalize it identically.
                            face = cv2.resize(gray_raw[y:y + h, x:x + w], (200, 200))
                            number = next_sample_number + saved
                            if cv2.imwrite(str(folder / f"face_{number:03d}.jpg"), face):
                                saved += 1
                                preview = face
                                status.progress(min(saved / 25, 1.0), text=f"Captured {saved}/25 samples")
                        time.sleep(0.10)
                finally:
                    camera.release()
                if saved:
                    st.success(f"Saved {saved} new samples for {selected[1]}.")
                    if preview is not None:
                        st.image(preview, caption="Last enrolled face sample", width=220, clamp=True)
                    st.warning("Retrain the model after enrollment: run `python train_faces.py` in your project terminal.")
                else:
                    st.error("No face samples were captured. Face the camera, improve the lighting, and try again.")

with tab3:
    st.subheader("Phone camera preview")
    st.markdown('<p class="section-note">Capture a fresh snapshot from IP Webcam. This button takes a snapshot; it is not a continuous video player.</p>', unsafe_allow_html=True)
    camera_url = st.text_input("IP Webcam video URL", value=DEFAULT_CAMERA_URL, key="preview_camera_url")
    if st.button("⟳ Capture fresh camera frame", type="primary"):
        camera = open_phone_camera(camera_url)
        if not camera.isOpened():
            camera.release()
            st.error("Cannot connect to camera. Check the phone server and URL.")
        else:
            with st.spinner("Waiting for the latest camera frame…"):
                try:
                    frame = read_fresh_frame(camera, discard=8)
                finally:
                    camera.release()
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption=f"Latest snapshot · {datetime.now().strftime('%H:%M:%S')}", use_container_width=True)
                st.success("Camera frame received. Move the phone and capture again to check for updates.")
            else:
                st.error("No frame received. Reopen IP Webcam, check Wi-Fi, and retry.")
    st.caption("Camera URL: http://10.128.201.119:8080/video")

with tab4:
    st.subheader("Attendance overview")
    attendance_df = load_attendance()
    students_df = load_students()
    today = datetime.now().strftime("%Y-%m-%d")
    today_df = attendance_df[attendance_df["Date"] == today] if not attendance_df.empty else attendance_df
    total_students = len(students_df)
    present = today_df["Student ID"].nunique() if not today_df.empty else 0
    absent = max(total_students - present, 0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total students", total_students)
    c2.metric("Present today", present)
    c3.metric("Not marked", absent)
    c4.metric("All-time records", len(attendance_df))
    st.markdown("#### Today's attendance")
    if today_df.empty:
        st.info("No attendance has been recorded today.")
    else:
        st.dataframe(today_df, use_container_width=True, hide_index=True)
    st.markdown("#### Attendance history")
    if attendance_df.empty:
        st.info("Attendance records will appear here after a student is verified.")
    else:
        query = st.text_input("Search by student name or enrollment number", key="attendance_search")
        shown = attendance_df
        if query.strip():
            match = attendance_df["Student Name"].fillna("").str.contains(query, case=False, regex=False) | attendance_df["Student ID"].fillna("").str.contains(query, case=False, regex=False)
            shown = attendance_df[match]
        st.dataframe(shown, use_container_width=True, hide_index=True)
        st.download_button("⬇ Export attendance (CSV)", data=shown.to_csv(index=False).encode("utf-8-sig"), file_name="attendance.csv", mime="text/csv")
    if st.button("↻ Refresh dashboard"):
        st.rerun()

with tab5:
    st.subheader("Live face recognition & attendance")
    st.markdown(
        '<p class="section-note">Continuous camera monitoring. Green boxes are accepted matches, amber boxes need review, and red boxes are unknown/low-confidence. Attendance is recorded only after a stable match for 2 seconds.</p>',
        unsafe_allow_html=True,
    )
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        st.error("Face model not found. Enroll samples, then run `python train_faces.py` in PowerShell.")
    else:
        camera_url = st.text_input("Phone camera URL", value=DEFAULT_CAMERA_URL, key="live_camera_url")
        confidence_limit = st.slider(
            "LBPH acceptance limit", min_value=45, max_value=110,
            value=int(st.session_state.get("live_confidence_limit", DEFAULT_CONFIDENCE_LIMIT)), step=1,
            help="LBPH distance is not a percentage: lower is a closer match. A higher limit can reduce missed matches but can cause false matches.",
        )
        st.session_state["live_confidence_limit"] = float(confidence_limit)
        st.caption("Phone and computer must be on a network that permits access to the IP Webcam stream.")
        start_col, stop_col = st.columns(2)
        with start_col:
            start_clicked = st.button(
                "▶ Start live attendance", type="primary", use_container_width=True,
                disabled=bool(st.session_state.get("live_active", False)),
            )
        with stop_col:
            stop_clicked = st.button(
                "■ Stop live attendance", use_container_width=True,
                disabled=not bool(st.session_state.get("live_active", False)),
            )

        if start_clicked:
            camera = open_phone_camera(camera_url.strip())
            if not camera.isOpened():
                camera.release()
                st.error("Cannot connect to the phone camera. Check IP Webcam and the URL.")
            else:
                recognizer, student_ids, model_error = load_recognizer()
                if model_error:
                    camera.release()
                    st.error(model_error)
                else:
                    with get_connection() as conn:
                        students = conn.execute("SELECT student_id, name FROM students").fetchall()
                        today_rows = conn.execute(
                            "SELECT student_id FROM attendance WHERE attendance_date = ?",
                            (datetime.now().strftime("%Y-%m-%d"),),
                        ).fetchall()
                    st.session_state["live_camera"] = camera
                    st.session_state["live_recognizer"] = recognizer
                    st.session_state["live_student_ids"] = student_ids
                    st.session_state["live_name_map"] = {
                        re.sub(r"[^A-Za-z0-9_-]", "_", sid): name for sid, name in students
                    }
                    st.session_state["live_detector"] = cv2.CascadeClassifier(
                        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                    )
                    st.session_state["live_marked_today"] = {row[0] for row in today_rows}
                    st.session_state["live_stable"] = {}
                    st.session_state["live_event_log"] = []
                    st.session_state["live_read_failures"] = 0
                    st.session_state["live_active"] = True
                    st.rerun()

        if stop_clicked:
            st.session_state["live_active"] = False
            active_camera = st.session_state.get("live_camera")
            if active_camera is not None:
                active_camera.release()
            st.session_state["live_camera"] = None
            st.session_state["live_stable"] = {}
            st.rerun()

        if st.session_state.get("live_active", False):
            st.success("Live attendance is running. Keep the browser tab open; use Stop live attendance to end the session.")
            render_live_attendance()
        else:
            st.info("Press **Start live attendance** to begin a continuous video feed with live green, amber, and red boxes. The old Camera Preview tab remains a one-shot snapshot tool.")

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
        st.dataframe(live_today, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown('<p class="small-muted">AI Smart Classroom Attendance · Local database: attendance.db · Check ambiguous matches before relying on results.</p>', unsafe_allow_html=True)
