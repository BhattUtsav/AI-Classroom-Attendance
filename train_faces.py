from __future__ import annotations

import re
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from face_utils import (
    SFACE_EMBEDDINGS_PATH,
    SFACE_LABELS_PATH,
    create_face_models,
    extract_embedding,
    prepare_sample_image,
)

BASE_DIR = Path(__file__).resolve().parent
FACES_DIR = BASE_DIR / "face_samples"
MODEL_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "attendance.db"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(value))


def main() -> None:
    if not DB_PATH.exists():
        raise SystemExit("attendance.db not found. Register students first.")
    if not FACES_DIR.exists():
        raise SystemExit("face_samples folder not found. Enroll at least one student's face first.")
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT student_id, name FROM students ORDER BY student_id").fetchall()
    registered = {safe_id(sid): (str(sid), str(name)) for sid, name in rows}
    if not registered:
        raise SystemExit("No registered students found in attendance.db.")

    try:
        detector, recognizer = create_face_models()
    except (FileNotFoundError, RuntimeError, cv2.error) as exc:
        raise SystemExit(f"Cannot load YuNet/SFace models: {exc}") from exc

    all_embeddings: list[np.ndarray] = []
    all_ids: list[str] = []
    counts: dict[str, int] = {}
    notes: list[str] = []
    skipped: list[str] = []

    for folder in sorted((p for p in FACES_DIR.iterdir() if p.is_dir()), key=lambda p: p.name):
        folder_key = safe_id(folder.name)
        if folder_key not in registered:
            notes.append(f"Ignoring folder not present in database: {folder.name}")
            continue
        student_id, student_name = registered[folder_key]
        image_paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        usable = 0
        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                skipped.append(f"{image_path.name}: unreadable image")
                continue
            aligned, method = prepare_sample_image(image, detector, recognizer)
            if aligned is None:
                skipped.append(f"{student_name}/{image_path.name}: {method}")
                continue
            try:
                feature = extract_embedding(recognizer, aligned)
            except (cv2.error, ValueError) as exc:
                skipped.append(f"{student_name}/{image_path.name}: embedding failed ({exc})")
                continue
            all_embeddings.append(feature.astype(np.float32))
            all_ids.append(student_id)
            usable += 1
            if method == "legacy crop fallback":
                notes.append(f"Legacy square face crop used without landmarks: {student_name}/{image_path.name}")
        if usable:
            counts[student_id] = usable

    trained_ids = sorted(set(all_ids))
    if not all_embeddings or not trained_ids:
        raise SystemExit(
            "No usable face embeddings were created. Ensure each folder contains a clear face photo, "
            "then run python download_sface_models.py and retry."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    backup_needed = [SFACE_EMBEDDINGS_PATH, SFACE_LABELS_PATH]
    if any(p.exists() for p in backup_needed):
        backup_dir = MODEL_DIR / "backups" / datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)
        for p in backup_needed:
            if p.exists():
                shutil.copy2(p, backup_dir / p.name)
        print(f"Previous SFace index backup: {backup_dir}")

    temp_embeddings = SFACE_EMBEDDINGS_PATH.with_name("sface_embeddings.tmp.npz")
    temp_labels = SFACE_LABELS_PATH.with_name("sface_labels.tmp.txt")
    np.savez_compressed(
        temp_embeddings,
        embeddings=np.vstack(all_embeddings).astype(np.float32),
        student_ids=np.asarray(all_ids, dtype=str),
    )
    temp_labels.write_text("\n".join(trained_ids) + "\n", encoding="utf-8")
    temp_embeddings.replace(SFACE_EMBEDDINGS_PATH)
    temp_labels.replace(SFACE_LABELS_PATH)

    print("\nSFace enrollment index built successfully.")
    print(f"Registered students in database: {len(registered)}")
    print(f"Students represented in SFace model: {len(trained_ids)}")
    print(f"Usable face templates: {len(all_embeddings)}")
    print(f"Embeddings saved to: {SFACE_EMBEDDINGS_PATH}")
    print("Recognition engine: pretrained SFace embeddings + YuNet face alignment")
    for sid in trained_ids:
        name = registered[safe_id(sid)][1]
        print(f"  {name} ({sid}): {counts.get(sid, 0)} templates")
    missing = [f"{name} ({sid})" for key, (sid, name) in registered.items() if sid not in counts]
    if missing:
        print("\nRegistered students without usable face samples:")
        for row in missing:
            print("  -", row)
    if notes:
        print("\nNotes:")
        for row in notes[:12]:
            print("  -", row)
        if len(notes) > 12:
            print(f"  - ... {len(notes) - 12} additional notes")
    if skipped:
        print("\nSkipped samples:")
        for row in skipped[:15]:
            print("  -", row)
        if len(skipped) > 15:
            print(f"  - ... {len(skipped) - 15} additional skipped files")
    print("\nImportant: only enrolled students can be identified. Test known and unknown people before relying on attendance.")


if __name__ == "__main__":
    main()
