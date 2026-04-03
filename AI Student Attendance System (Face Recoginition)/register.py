import cv2
import pickle
import numpy as np
from deepface import DeepFace
import os

EMBEDDING_FILE = "embeddings.pkl"

name = input("Enter Student Name: ")

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

print("Press S to save face | Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    face_crop = None

    for (x, y, w, h) in faces:
        face_crop = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Register Face", frame)

    key = cv2.waitKey(10) & 0xFF   # ✅ ONLY ONCE PER FRAME

    if key == ord('s') and face_crop is not None:
        print("Processing...")

        embedding = DeepFace.represent(
            img_path=face_crop,
            model_name="Facenet",
            enforce_detection=False
        )[0]["embedding"]

        if os.path.exists(EMBEDDING_FILE):
            with open(EMBEDDING_FILE, "rb") as f:
                data = pickle.load(f)
        else:
            data = {}

        data[name] = embedding

        with open(EMBEDDING_FILE, "wb") as f:
            pickle.dump(data, f)

        print(f"{name} Registered Successfully ✅")
        break

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()