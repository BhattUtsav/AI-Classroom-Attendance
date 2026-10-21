# AI Smart Classroom Attendance

A local Streamlit dashboard for student registration, face sample enrollment, camera snapshots, LBPH face recognition, and attendance export.

## Features

- Register students and export a CSV of the class list.
- Capture face samples from an IP Webcam phone stream.
- Train an OpenCV LBPH face recognizer.
- Verify a match across several recent camera frames before recording attendance.
- Display green accepted matches, amber uncertain matches, and red unrecognized faces.
- Search and export attendance records.
- Keep the SQLite database, student photos, face samples, and trained models local and out of Git.

## Setup on Windows PowerShell

Open PowerShell in the project directory:

```powershell
cd C:\Users\Acer\Desktop\AI-Classroom-Attendance
```

Activate the project's virtual environment (create it if you do not already have one):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `venv` already exists, do not run `python -m venv venv` again; activate it and install requirements instead.

Start the dashboard:

```powershell
python -m streamlit run app.py
```

Open the local URL Streamlit prints, normally `http://localhost:8501`.

## Phone camera

Start IP Webcam on your phone and use the video URL, for example:

```text
http://10.128.201.119:8080/video
```

The phone and computer must be connected to a network that allows them to reach each other. If the IP address changes, update the URL in the dashboard.

## Enrollment and training

1. Register each student in the dashboard using their enrollment number.
2. Open **Face Enrollment** and capture varied, well-lit samples for that student.
3. Repeat for each registered student.
4. In PowerShell (with the project environment activated), run:

   ```powershell
   python train_faces.py
   ```

5. Return to **Live Attendance** and test each registered student under the actual camera and lighting conditions.

Every time you add or recapture samples, retrain the model. Do not train on a folder for a student who is not registered in `attendance.db`.

## Reading the recognition score

The OpenCV LBPH score is a **distance, not a probability or percentage**. A lower distance is a closer model match. The dashboard starts at a limit of 80 and lets you adjust it for testing. Raising the limit can reduce missed matches, but can also increase false matches. The app uses a multi-frame consensus and an amber uncertain state to avoid marking a borderline match automatically. Evaluate with several known students and unknown people before relying on attendance records.

The model is sensitive to the quality and variety of enrollment samples, face size, lighting, camera angle, blur, and distance. No threshold guarantees error-free identification.

## Local data and privacy

The project is designed to store the following locally:

- `attendance.db`
- `face_samples/`
- `source_photos/`
- `models/`
- `venv/` or `.venv/`

These paths are ignored by `.gitignore`. Do not commit student photos, biometric samples, attendance databases, or secrets to a public repository.
