import streamlit as st
import cv2
import sqlite3
import pandas as pd
import time
import re
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "attendance.db"
FACES_DIR = BASE_DIR / "face_samples"

def get_connection():
    return sqlite3.connect(DB_PATH)


def create_tables():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                attendance_date TEXT NOT NULL,
                attendance_time TEXT NOT NULL,
                status TEXT DEFAULT 'Present',
                FOREIGN KEY (student_id) REFERENCES students(student_id),
                UNIQUE (student_id, attendance_date)
            )
        """)

def load_students():
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT student_id, name, department, created_at "
            "FROM students ORDER BY name", conn
        )


def load_attendance():
    with get_connection() as conn:
        return pd.read_sql_query("""
            SELECT
                a.student_id AS 'Student ID',
                s.name AS 'Student Name',
                s.department AS 'Department',
                a.attendance_date AS 'Date',
                a.attendance_time AS 'Time',
                a.status AS 'Status'
            FROM attendance a
            LEFT JOIN students s ON a.student_id = s.student_id
            ORDER BY a.attendance_date DESC, a.attendance_time DESC
        """, conn)    

def student_folder(student_id):
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", student_id)
    return FACES_DIR / safe_id


def recognize_and_mark_attendance(camera_url):
    model_path = BASE_DIR / "models" / "lbph_model.yml"
    labels_path = BASE_DIR / "models" / "labels.txt"

    if not model_path.exists() or not labels_path.exists():
        return None, "Face model files not found."

    student_ids = labels_path.read_text(
        encoding="utf-8"
    ).splitlines()

    with get_connection() as conn:
        students = conn.execute(
            "SELECT student_id, name FROM students"
        ).fetchall()

    names = {
        re.sub(r"[^A-Za-z0-9_-]", "_", sid): name
        for sid, name in students
    }

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(model_path))

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    camera = cv2.VideoCapture(camera_url)

    if not camera.isOpened():
        camera.release()
        return None, "Cannot connect to phone camera. Check the URL."

    success, frame = camera.read()
    camera.release()

    if not success:
        return None, "Could not capture a camera frame."

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return frame, "No face detected. Try better lighting."

    # Use the largest detected face
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))

    label, confidence = recognizer.predict(face)

    # Lower LBPH confidence means a closer match
    if not (0 <= label < len(student_ids)) or confidence >= 70:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(
            frame, "Unknown", (x, max(25, y-10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        return frame, f"Unknown face (confidence: {confidence:.1f}). No attendance marked."

    sid = student_ids[label]
    name = names.get(sid)

    if name is None:
        return frame, "Recognized ID is not registered in the database."

    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    try:
        with get_connection() as conn:
            conn.execute("""
                INSERT INTO attendance
                (student_id, attendance_date, attendance_time, status)
                VALUES (?, ?, ?, 'Present')
            """, (sid, today, current_time))
        message = f"Attendance marked for {name} at {current_time}."
    except sqlite3.IntegrityError:
        message = f"{name} is already marked present today."

    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 0), 2)
    cv2.putText(
        frame, name, (x, max(25, y-10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2
    )

    return frame, message

create_tables()

st.set_page_config(
    page_title="AI Classroom Attendance",
    page_icon="🎓",
    layout="wide"
)

st.title("AI Smart Classroom Attendance System")
st.caption("Government Polytechnic Junagadh | Computer Engineering")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Student Registration",
    "Face Enrollment",
    "Camera Preview",
    "Attendance Dashboard",
    "Live Attendance"
])

with tab1:
    st.subheader("Register a Student")

    with st.form("student_form", clear_on_submit=True):
        student_id = st.text_input("Enrollment Number").strip()
        name = st.text_input("Student Name").strip()
        department = st.text_input(
            "Department / Class",
            value="Computer Engineering"
        ).strip()

        submitted = st.form_submit_button("Register Student")

        if submitted:
            if not student_id or not name or not department:
                st.error("Please fill in all fields.")
            else:
                try:
                    with get_connection() as conn:
                        conn.execute(
                            "INSERT INTO students "
                            "(student_id, name, department) VALUES (?, ?, ?)",
                            (student_id, name, department)
                        )
                    st.success(f"Student registered: {name}")
                except sqlite3.IntegrityError:
                    st.error("This enrollment number is already registered.")

    students_df = load_students()
    st.metric("Total Registered Students", len(students_df))
    st.dataframe(students_df, use_container_width=True)

    if not students_df.empty:
        st.download_button(
            "Download Student List (CSV)",
            data=students_df.to_csv(index=False).encode("utf-8"),
            file_name="students.csv",
            mime="text/csv"
        )

with tab2:
    st.subheader("Enroll a Student's Face")
    st.write(
        "Select a registered student, look at the camera, "
        "and capture face samples."
    )

    students_df = load_students()

    if students_df.empty:
        st.warning("Register at least one student first.")
    else:
        options = [
            (row["student_id"], row["name"])
            for _, row in students_df.iterrows()
        ]

        selected = st.selectbox(
            "Select student",
            options,
            format_func=lambda item: f"{item[1]} ({item[0]})"
        )

        camera_url = st.text_input(
            "IP Webcam video URL",
            value="http://10.128.201.119:8080/video",
            key="enrollment_camera_url"
        )

        folder = student_folder(selected[0])
        existing_count = (
            len(list(folder.glob("*.jpg"))) if folder.exists() else 0
        )
        st.write(f"Existing face samples: {existing_count}")

        st.info(
            "Use good lighting. Keep one face centered and look toward "
            "the camera. The app will try to capture up to 25 face samples."
        )

        if st.button("Capture Face Samples"):
            cascade_path = (
                cv2.data.haarcascades
                + "haarcascade_frontalface_default.xml"
            )
            detector = cv2.CascadeClassifier(cascade_path)

            if detector.empty():
                st.error("Could not load the face detector.")
            else:
                camera = cv2.VideoCapture(camera_url)

                if not camera.isOpened():
                    st.error("Cannot connect to the phone camera.")
                else:
                    folder.mkdir(parents=True, exist_ok=True)
                    saved = 0
                    attempts = 0
                    preview = None

                    status = st.empty()
                    with st.spinner("Capturing face samples..."):
                        while saved < 25 and attempts < 100:
                            attempts += 1
                            success, frame = camera.read()

                            if not success:
                                time.sleep(0.1)
                                continue

                            gray = cv2.cvtColor(
                                frame, cv2.COLOR_BGR2GRAY
                            )
                            gray = cv2.equalizeHist(gray)

                            detected = detector.detectMultiScale(
                                gray,
                                scaleFactor=1.1,
                                minNeighbors=5,
                                minSize=(80, 80)
                            )

                            if len(detected) > 0:
                                x, y, w, h = max(
                                    detected,
                                    key=lambda box: box[2] * box[3]
                                )
                                face = gray[y:y+h, x:x+w]
                                face = cv2.resize(face, (200, 200))

                                number = existing_count + saved + 1
                                filename = folder / f"face_{number:03d}.jpg"

                                if cv2.imwrite(str(filename), face):
                                    saved += 1
                                    preview = face

                            status.write(
                                f"Face samples captured: {saved}/25"
                            )
                            time.sleep(0.12)

                    camera.release()

                    if saved > 0:
                        st.success(
                            f"Saved {saved} new face samples for "
                            f"{selected[1]}."
                        )
                        if preview is not None:
                            st.image(
                                preview,
                                caption="Last detected face sample",
                                clamp=True,
                                width=200
                            )
                    else:
                        st.error(
                            "No face detected. Improve lighting, move "
                            "closer, and try again."
                        )

    st.caption(
        "Samples are stored locally in the face_samples folder. "
        "This step collects face images; it does not yet recognize "
        "students or mark attendance."
    )

with tab3:
    st.subheader("Phone Camera Preview")

    camera_url = st.text_input(
        "IP Webcam video URL",
        value="http://10.128.201.119:8080/video",
        key="preview_camera_url"
    )

    if st.button("Capture Camera Frame"):
        camera = cv2.VideoCapture(camera_url)

        if not camera.isOpened():
            st.error("Cannot connect to camera. Check the URL.")
        else:
            success, frame = camera.read()
            camera.release()

            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame, caption="Camera Frame", width=640)
                st.success("Camera connection is working.")
            else:
                st.error("No frame received from camera.")

with tab4:
    st.subheader("Attendance Dashboard")

    attendance_df = load_attendance()
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    today_df = attendance_df[attendance_df["Date"] == today]

    col1, col2, col3 = st.columns(3)
    col1.metric("Registered Students", len(load_students()))
    col2.metric("Present Today", today_df["Student ID"].nunique())
    col3.metric("Total Attendance Records", len(attendance_df))

    st.subheader("Today's Attendance")
    if today_df.empty:
        st.info("No attendance recorded today yet.")
    else:
        st.dataframe(today_df, use_container_width=True)

    st.subheader("All Attendance Records")
    if attendance_df.empty:
        st.info("No attendance records found.")
    else:
        st.dataframe(attendance_df, use_container_width=True)

        st.download_button(
            "Download Attendance CSV",
            data=attendance_df.to_csv(index=False).encode("utf-8"),
            file_name="attendance.csv",
            mime="text/csv"
        )

    if st.button("Refresh Attendance"):
        st.rerun()     




with tab5:
    st.subheader("Live Face Recognition & Attendance")

    model_path = BASE_DIR / "models" / "lbph_model.yml"
    labels_path = BASE_DIR / "models" / "labels.txt"

    if not model_path.exists() or not labels_path.exists():
        st.error("Model not found. Run train_faces.py first.")
    else:
        camera_url = st.text_input(
            "Phone Camera URL",
            value="http://10.128.201.119:8080/video",
            key="live_camera_url"
        )

        run_live = st.checkbox(
            "Enable live recognition",
            key="run_live_recognition"
        )

        
        if run_live:
            if st.button("Capture and Recognize Frame"):
                with st.spinner("Recognizing face..."):
                    frame, message = recognize_and_mark_attendance(camera_url)

                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(
                        frame,
                        caption="Recognition Result",
                        width=640
                    )

                if message.startswith("Attendance marked"):
                    st.success(message)
                elif "already marked" in message:
                    st.info(message)
                else:
                    st.warning(message)

        st.subheader("Today's Attendance")

        with get_connection() as conn:
            today = pd.Timestamp.now().strftime("%Y-%m-%d")

            today_df = pd.read_sql_query("""
                SELECT
                    a.student_id AS 'Student ID',
                    s.name AS 'Student Name',
                    a.attendance_date AS 'Date',
                    a.attendance_time AS 'Time',
                    a.status AS 'Status'
                FROM attendance a
                LEFT JOIN students s
                    ON a.student_id = s.student_id
                WHERE a.attendance_date = ?
                ORDER BY a.attendance_time DESC
            """, conn, params=(today,))

        st.metric(
            "Students Present Today",
            today_df["Student ID"].nunique()
        )
        st.dataframe(today_df, use_container_width=True)

        st.subheader("Today's Attendance")

        with get_connection() as conn:
            today = pd.Timestamp.now().strftime("%Y-%m-%d")
            today_df = pd.read_sql_query("""
                SELECT a.student_id AS 'Student ID',
                       s.name AS 'Student Name',
                       a.attendance_date AS 'Date',
                       a.attendance_time AS 'Time',
                       a.status AS 'Status'
                FROM attendance a
                LEFT JOIN students s ON a.student_id = s.student_id
                WHERE a.attendance_date = ?
                ORDER BY a.attendance_time DESC
            """, conn, params=(today,))

        st.metric(
            "Students Present Today",
            today_df["Student ID"].nunique()
        )
        st.dataframe(today_df, use_container_width=True)                  


st.divider()
st.caption("Local database: attendance.db")
