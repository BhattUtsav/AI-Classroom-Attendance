
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
FACES_DIR = BASE_DIR / "face_samples"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "lbph_model.yml"
LABELS_PATH = MODEL_DIR / "labels.txt"

faces = []
labels = []
student_ids = []

if not FACES_DIR.exists():
    print("Face samples folder not found.")
    raise SystemExit(1)

for student_folder in sorted(FACES_DIR.iterdir()):
    if not student_folder.is_dir():
        continue

    images = list(student_folder.glob("*.jpg"))
    if not images:
        continue

    label = len(student_ids)
    student_ids.append(student_folder.name)

    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        image = cv2.resize(image, (200, 200))
        faces.append(image)
        labels.append(label)

if not faces:
    print("No usable face samples found. Capture faces first.")
    raise SystemExit(1)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels, dtype=np.int32))

MODEL_DIR.mkdir(parents=True, exist_ok=True)
recognizer.write(str(MODEL_PATH))
LABELS_PATH.write_text(
    "\n".join(student_ids),
    encoding="utf-8"
)

print("Training complete!")
print("Students trained:", len(student_ids))
print("Face samples used:", len(faces))
print("Model saved to:", MODEL_PATH)
