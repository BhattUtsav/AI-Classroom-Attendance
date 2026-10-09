
import cv2
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent
SOURCE_DIR = BASE_DIR / "source_photos"
OUTPUT_DIR = BASE_DIR / "face_samples"

STUDENT_NAMES = {
    "246190307024": "Gresha",
    "246190307032": "Bhumi",
    "246190307069": "Diya",
}

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

if not SOURCE_DIR.exists():
    print("ERROR: source_photos folder not found.")
    raise SystemExit(1)

# Remove old test samples so they cannot affect training.
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

total_samples = 0

for student_id, name in STUDENT_NAMES.items():
    student_source = SOURCE_DIR / student_id
    student_output = OUTPUT_DIR / student_id

    if not student_source.exists():
        print(f"Folder missing for {name}: {student_id}")
        continue

    student_output.mkdir(parents=True, exist_ok=True)
    count = 0

    for photo in student_source.rglob("*"):
        if photo.suffix.lower() not in (
            ".jpg", ".jpeg", ".png", ".bmp"
        ):
            continue

        image = cv2.imread(str(photo))

        if image is None:
            print("Could not read:", photo.name)
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(faces) == 0:
            print(f"No face detected in {name}: {photo.name}")
            continue

        x, y, w, h = max(
            faces, key=lambda face: face[2] * face[3]
        )

        face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))

        for brightness in (-15, 0, 15):
            sample = cv2.convertScaleAbs(
                face, alpha=1.0, beta=brightness
            )

            count += 1
            output_path = student_output / f"face_{count:03d}.jpg"
            cv2.imwrite(str(output_path), sample)

    total_samples += count
    print(f"{name} ({student_id}): {count} samples created")

print("\nPhoto processing finished.")
print("Total samples:", total_samples)
print("Output folder:", OUTPUT_DIR)