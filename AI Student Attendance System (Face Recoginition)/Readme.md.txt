📦 1. System Requirements
💻 Software Requirements
Component	Version
Python	3.9.25
pip	26.0.1
OS	Windows 10 / 11 (Recommended)
📚 2. Python Libraries (Requirements.txt)

Below is your clean and usable requirements.txt (only important packages, no unnecessary ones):

✅ 🔹 Core AI & Vision
opencv-python==4.13.0.92
opencv-contrib-python==4.13.0.92
deepface==0.0.98
tensorflow==2.20.0
keras==3.10.0
mtcnn==0.1.1
retina-face==0.0.17
✅ 🔹 Data Processing
numpy==2.0.2
pandas==2.3.3
scikit-learn==1.6.1
scipy==1.13.1
✅ 🔹 Visualization & UI
streamlit==1.50.0
matplotlib==3.9.4
✅ 🔹 Backend & Utility
pickle-mixin==1.0.2
python-dateutil==2.9.0.post0
pytz==2025.2
✅ 🔹 Optional (Performance / Support)
onnxruntime==1.19.2
mediapipe==0.10.32
⚠️ Important Note (VERY IMPORTANT ⚡)

👉 Do NOT install all 100+ packages from pip list
👉 Only install required ones (above)

📄 3. Final Requirements.txt (Ready to Copy)

Just paste this 👇

opencv-python==4.13.0.92
opencv-contrib-python==4.13.0.92
deepface==0.0.98
tensorflow==2.20.0
keras==3.10.0
mtcnn==0.1.1
retina-face==0.0.17
numpy==2.0.2
pandas==2.3.3
scikit-learn==1.6.1
scipy==1.13.1
streamlit==1.50.0
matplotlib==3.9.4
onnxruntime==1.19.2
mediapipe==0.10.32
python-dateutil==2.9.0.post0
pytz==2025.2
⚙️ 4. Installation Commands
🔹 Step 1: Create Virtual Environment
python -m venv venv
🔹 Step 2: Activate
venv\Scripts\activate
🔹 Step 3: Install Requirements
pip install -r requirements.txt
🧠 5. Why These Libraries?
Library	Why Used
OpenCV	Camera + face detection
DeepFace	Face recognition
TensorFlow	Backend for DeepFace
NumPy	Vector math
Pandas	Attendance storage
Streamlit	Dashboard UI
Scikit-learn	Cosine similarity
🚀 6. Optional Improvements (Advanced)

If you want better performance:

torch
ultralytics

👉 For YOLO-based face detection upgrade


🎓 AI Student Attendance System (Face Recognition)
🧠 1. Project Overview

This system automatically marks student attendance using face recognition instead of manual methods.

🔹 Core Flow:
Register student face
Convert face → embedding (AI vector)
Store embedding
Detect face in live camera
Compare with stored embeddings
Mark attendance if matched
⚙️ 2. Technologies Used
Component	Purpose
OpenCV (cv2)	Camera + face detection
DeepFace	Face recognition (FaceNet model)
NumPy	Vector calculations
Pandas	Attendance storage
Streamlit	Web dashboard
Pickle	Save embeddings
Sklearn	Cosine similarity
📂 3. File-wise Explanation (VERY IMPORTANT)
🟢 3.1 register.py → Face Registration
🎯 Purpose:

Register a student and store their face embedding.

🔄 Process Flow:
Open camera
Detect face using Haar Cascade
Crop face
Convert face → embedding using DeepFace
Save embedding in embeddings.pkl
🧠 Key Concepts:
🔹 Face Detection
face_cascade.detectMultiScale(gray, 1.1, 5)

Detects face region.

🔹 Face Embedding
DeepFace.represent(...)
Converts face → 128/512 vector
This is the identity of the person
🔹 Storage
data[name] = embedding

Saved as:

{
  "Nilay": [0.123, 0.567, ...]
}
✅ Output:
File created: embeddings.pkl
Student registered successfully
🔵 3.2 main.py → Basic Recognition + Attendance
🎯 Purpose:

Detect faces and mark attendance.

🔄 Process:
Load embeddings
Start camera
Detect face
Generate embedding
Compare with stored embeddings
Mark attendance
🧠 Matching Logic:
🔹 Cosine Similarity
score = np.dot(a, b) / (|a| * |b|)
Range: 0 to 1
Closer to 1 = same person
🔹 Threshold
if best_score > 0.7:

0.7 → Match

< 0.7 → Unknown
📄 Attendance System:
attendance.csv
Name	Time
Nilay	10:32:10
⚡ Optimization:
marked_today = set()

Prevents duplicate attendance.

🟡 3.3 rtsp_fast_detection.py → CCTV Detection
🎯 Purpose:

Detect faces from IP camera (RTSP stream)

🔄 Flow:
Connect to RTSP camera
Use RetinaFace (DeepFace)
Detect faces
Draw bounding boxes
🔥 Important:
No recognition
Only detection
Faster + better for CCTV
🟣 3.4 PRO_AI_ATTENDANCE_SYSTEM.py → Dashboard Version

📄 File:

🎯 Purpose:

Complete system with UI (Streamlit)

🔥 Features:

✔ Login system
✔ Register users
✔ Live recognition
✔ Attendance table
✔ Analytics (graphs)
✔ Delete users

🧠 Important Modules:
🔹 run_camera()

Main engine:

Detect face
Extract embedding
Register OR recognize
🔹 find_match()
Compares embeddings
Uses cosine similarity
Returns name + confidence
🔹 mark_attendance()
Saves:
Name | Date | Time
🔹 dashboard()

Controls:

Register
Recognition
Users
Attendance
Analytics
🌙 3.5 Night Vision Version

📄 File:

🎯 Purpose:

Improve detection in low light

🔥 Key Function:
enhance_night_vision()
🧠 Techniques Used:
CLAHE (Contrast Enhancement)
Gamma Correction
Grayscale enhancement
✅ Result:

Better face detection at night / dark areas

🔐 3.6 Secure Version (Best Version 🚀)

📄 File:

🎯 Purpose:

Add security + accuracy improvements

🔥 Features:

✔ Anti-spoof detection
✔ Multiple embeddings per user
✔ Higher threshold (0.82)
✔ Better accuracy
✔ Fake face prevention

🧠 Important Concepts:
🔹 1. Multiple Embeddings
db[name].append(embedding)

👉 Each user has MANY samples
👉 Improves accuracy

🔹 2. Liveness Detection
eye_detector.detectMultiScale(...)

✔ Real person → eyes detected
❌ Photo → no eyes → blocked

🔹 3. Strong Threshold
if best_score > 0.82:

👉 Reduces false matches

🔹 4. Frame Skipping
if frame_count % 5 == 0:

👉 Avoids duplicate embeddings
👉 Faster processing

📊 4. System Architecture
Camera Input
     ↓
Face Detection (OpenCV / RetinaFace)
     ↓
Face Cropping
     ↓
DeepFace (FaceNet)
     ↓
Embedding Vector
     ↓
Compare with Database
     ↓
Match / Unknown
     ↓
Attendance Stored (CSV)
     ↓
Dashboard (Streamlit)
📁 5. Data Files
📌 embeddings.pkl

Stores:

{ name : [embedding vectors] }
📌 attendance.csv

Stores:

Name | Date | Time
🚀 6. Key Strengths of Your Project

🔥 Real-time recognition
🔥 Works with webcam + CCTV
🔥 Dashboard UI
🔥 Night vision support
🔥 Anti-spoof security
🔥 Analytics system
🔥 Multiple embeddings (high accuracy)

⚠️ 7. Limitations
Haar cascade is not very accurate
Lighting affects performance
Threshold tuning needed
Single camera scalability
💡 8. Future Improvements

👉 Use YOLOv8 face detection
👉 Add database (MySQL)
👉 Add mask detection
👉 Add multi-camera tracking
👉 Use ArcFace (better than FaceNet)

🏁 Final Summary

Your system is:

👉 AI-based biometric attendance system
👉 Uses DeepFace (FaceNet) for recognition
👉 Uses cosine similarity for matching
👉 Stores embeddings in pickle file
👉 Provides full UI using Streamlit dashboard
👉 Includes security + night vision + CCTV support