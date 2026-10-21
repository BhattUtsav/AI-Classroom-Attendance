from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODELS = {
    "face_detection_yunet_2023mar.onnx": {
        "min_bytes": 100_000,
        "urls": [
            "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx?download=true",
            "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        ],
    },
    "face_recognition_sface_2021dec.onnx": {
        "min_bytes": 30_000_000,
        "urls": [
            "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx?download=true",
            "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        ],
    },
}


def download_file(name: str, minimum_bytes: int, urls: list[str]) -> None:
    dest = MODEL_DIR / name
    if dest.is_file() and dest.stat().st_size >= minimum_bytes:
        print(f"Already downloaded: {dest.name} ({dest.stat().st_size:,} bytes)")
        return
    last_error: Exception | None = None
    for url in urls:
        temp = dest.with_suffix(dest.suffix + ".download")
        try:
            print(f"Downloading {name} ...")
            request = urllib.request.Request(url, headers={"User-Agent": "AI-Classroom-Attendance/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, temp.open("wb") as output:
                total = 0
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
                    total += len(block)
            if total < minimum_bytes:
                raise RuntimeError(f"Downloaded file is too small ({total:,} bytes); expected model data.")
            temp.replace(dest)
            print(f"Saved {dest} ({total:,} bytes)")
            return
        except Exception as exc:
            last_error = exc
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass
            print(f"Source failed: {exc}")
            time.sleep(0.5)
    raise RuntimeError(f"Could not download {name}. Last error: {last_error}")


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for name, config in MODELS.items():
            download_file(name, int(config["min_bytes"]), list(config["urls"]))
    except Exception as exc:
        print(f"\nModel download failed: {exc}", file=sys.stderr)
        print("Check your internet connection, then run this script again.", file=sys.stderr)
        return 1
    print("\nSFace models are ready. Next run: python train_faces.py")
    print("Model sources: OpenCV Zoo (YuNet face detection and SFace face recognition).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
