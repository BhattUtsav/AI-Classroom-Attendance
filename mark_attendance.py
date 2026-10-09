
import sqlite3
from datetime import datetime


def mark_attendance(student_id):
    conn = sqlite3.connect("attendance.db")

    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    try:
        conn.execute("""
            INSERT INTO attendance
            (student_id, attendance_date, attendance_time, status)
            VALUES (?, ?, ?, 'Present')
        """, (student_id, today, current_time))

        conn.commit()
        print("Attendance marked successfully!")

    except sqlite3.IntegrityError:
        print("Attendance already marked today.")

    finally:
        conn.close()


if __name__ == "__main__":
    student_id = input("Enter student ID: ")
    mark_attendance(student_id)