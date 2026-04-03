import cv2
import pickle
import numpy as np
from deepface import DeepFace
import pandas as pd
from datetime import datetime
import os

EMBEDDING_FILE = "embeddings.pkl"
ATTENDANCE_FILE = "attendance.csv"

# Load embeddings
if not os.path.exists(EMBEDDING_FILE):
    print("No registered students found ❌")
    exit()

with open(EMBEDDING_FILE, "rb") as f:
    known_faces = pickle.load(f)

print("Loaded Students:", list(known_faces.keys()))

# Create attendance file if not exists
if not os.path.exists(ATTENDANCE_FILE):
    df = pd.DataFrame(columns=["Name", "Time"])
    df.to_csv(ATTENDANCE_FILE, index=False)

marked_today = set()

# Change this only
source = 1  # For USB camera
# source = "rtsp://username:password@ip:port/stream"  # For RTSP

cap = cv2.VideoCapture(source)



face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

print("Recognition Started ✅")
print("Press Q to quit")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face_crop = frame[y:y+h, x:x+w]

        try:
            embedding = DeepFace.represent(
                img_path=face_crop,
                model_name="Facenet",
                enforce_detection=False
            )[0]["embedding"]

            best_match = "Unknown"
            best_score = 0

            for name, known_embedding in known_faces.items():
                score = cosine_similarity(
                    np.array(embedding),
                    np.array(known_embedding)
                )

                if score > best_score:
                    best_score = score
                    best_match = name

            if best_score > 0.7:   # Threshold
                label = best_match

                if best_match not in marked_today:
                    now = datetime.now().strftime("%H:%M:%S")
                    df = pd.read_csv(ATTENDANCE_FILE)
                    df.loc[len(df)] = [best_match, now]
                    df.to_csv(ATTENDANCE_FILE, index=False)
                    marked_today.add(best_match)
                    print(f"{best_match} Attendance Marked ✅")
            else:
                label = "Unknown"

        except:
            label = "Processing..."

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,255,0), 2)

    cv2.imshow("AI Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()