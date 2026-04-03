import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
from deepface import DeepFace

rtsp_url = "rtsp://admin:admin123456@192.168.1.10:8554/profile0"

print("Connecting to RTSP...")
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("❌ Cannot open stream")
    exit()

print("✅ Connected")

print("Loading RetinaFace detector...")
detector_backend = "retinaface"

print("Recognition Started ✅")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    try:
        faces = DeepFace.extract_faces(
            img_path=frame,
            detector_backend=detector_backend,
            enforce_detection=False
        )

        for face in faces:
            x, y, w, h = face["facial_area"].values()

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    except:
        pass

    cv2.imshow("CCTV Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()