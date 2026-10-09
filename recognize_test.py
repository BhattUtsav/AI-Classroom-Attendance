
import cv2
import sqlite3
import re
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "lbph_model.yml"
LABELS_PATH = BASE_DIR / "models" / "labels.txt"
DB_PATH = BASE_DIR / "attendance.db"

# Check trained model
if not MODEL_PATH.exists() or not LABELS_PATH.exists():
    print("Run train_faces.py first.")
    raise SystemExit(1)

student_ids = LABELS_PATH.read_text(
    encoding="utf-8"
).splitlines()

with sqlite3.connect(DB_PATH) as conn:
    students = conn.execute(
        "SELECT student_id, name FROM students"
    ).fetchall()

names = {
    re.sub(r"[^A-Za-z0-9_-]", "_", sid): name
    for sid, name in students
}

# Face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(str(MODEL_PATH))

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

# Connect phone camera
url = input("Enter phone video URL: ").strip()
camera = cv2.VideoCapture(url)

if not camera.isOpened():
    print("Cannot connect to camera. Check the URL.")
    raise SystemExit(1)

# Attendance settings
CONFIDENCE_LIMIT = 70
REQUIRED_SECONDS = 2
first_seen = {}
marked_today = set()

print("Face recognition started. Press Q to quit.")
print("Keep your face visible for 2 seconds to mark attendance.")

while True:
    success, frame = camera.read()

    if not success:
        print("Unable to read camera frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for x, y, w, h in faces:
        face = cv2.resize(
            gray[y:y+h, x:x+w], (200, 200)
        )

        label, confidence = recognizer.predict(face)

        if (
            0 <= label < len(student_ids)
            and confidence < CONFIDENCE_LIMIT
        ):
            sid = student_ids[label]
            name = names.get(sid, sid)

            # Require continuous recognition for 2 seconds
            if sid not in first_seen:
                first_seen[sid] = time.time()

            if time.time() - first_seen[sid] >= REQUIRED_SECONDS:
                today = datetime.now().strftime("%Y-%m-%d")
                current_time = datetime.now().strftime("%H:%M:%S")

                try:
                    with sqlite3.connect(DB_PATH) as conn:
                        conn.execute("""
                            INSERT INTO attendance
                            (student_id, attendance_date,
                             attendance_time, status)
                            VALUES (?, ?, ?, 'Present')
                        """, (sid, today, current_time))

                    if sid not in marked_today:
                        print(
                            f"Attendance marked: {name} "
                            f"at {current_time}"
                        )
                        marked_today.add(sid)

                    text = f"{name} - Present"
                    color = (0, 200, 0)

                except sqlite3.IntegrityError:
                    text = f"{name} - Already Present"
                    color = (0, 200, 0)

            else:
                text = f"{name} - Confirming..."
                color = (0, 200, 0)

        else:
            text = f"Unknown ({confidence:.1f})"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame, text, (x, max(25, y-10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

    cv2.imshow("AI Classroom Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()