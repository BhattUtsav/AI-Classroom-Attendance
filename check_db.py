
import sqlite3

conn = sqlite3.connect("attendance.db")

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

conn.commit()
print("Attendance table created successfully!")

tables = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table'"
).fetchall()

print("Database tables:", tables)

conn.close()