# 🎓 AI Student Attendance System (Face Recognition)

---

## 📌 Project Overview

The **AI Student Attendance System** is a real-time face recognition-based application designed to automate attendance marking using computer vision and deep learning techniques.

This system replaces traditional manual attendance methods with an intelligent, secure, and efficient solution using facial recognition.

---

## 🚀 Features

* ✅ Real-time face recognition
* ✅ Student face registration system
* ✅ Automatic attendance marking
* ✅ Streamlit-based dashboard UI
* ✅ CCTV (RTSP) camera support
* ✅ Night vision enhancement
* ✅ Anti-spoof (liveness) detection
* ✅ Attendance analytics & reports
* ✅ Multiple embeddings for higher accuracy

---

## ⚙️ System Requirements

### 💻 Software Requirements

| Component | Version         |
| --------- | --------------- |
| Python    | 3.9.25          |
| pip       | 26.0.1          |
| OS        | Windows 10 / 11 |

---

## 📦 Installation

### 🔹 Step 1: Clone Repository

```bash
git clone https://github.com/your-username/ai-attendance-system.git
cd ai-attendance-system
```

---

### 🔹 Step 2: Create Virtual Environment

```bash
python -m venv venv
```

---

### 🔹 Step 3: Activate Environment

```bash
venv\Scripts\activate
```

---

### 🔹 Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📚 Requirements (requirements.txt)

```txt
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
```

---

## 🧠 Technologies Used

| Technology         | Purpose                          |
| ------------------ | -------------------------------- |
| OpenCV             | Face detection & camera handling |
| DeepFace (FaceNet) | Face recognition                 |
| TensorFlow / Keras | Deep learning backend            |
| NumPy              | Mathematical operations          |
| Pandas             | Attendance data handling         |
| Streamlit          | Web dashboard                    |
| Scikit-learn       | Cosine similarity                |

---

## 📂 Project Structure

```
├── register.py
├── main.py
├── rtsp_fast_detection.py
├── PRO_AI_ATTENDANCE_SYSTEM.py
├── PRO_AI_ATTENDANCE_SYSTEM_SECURE.py
├── embeddings.pkl
├── attendance.csv
├── requirements.txt
└── README.md
```

---

## 🔍 How It Works

1. Capture face from camera
2. Detect face using OpenCV / RetinaFace
3. Convert face into embedding using DeepFace (FaceNet)
4. Store embeddings in database (pickle file)
5. Compare real-time face with stored embeddings
6. Mark attendance if matched

---

## 📂 File Descriptions

### 🟢 `register.py` – Face Registration

* Captures student face using webcam
* Detects face using Haar Cascade
* Converts face to embedding using DeepFace
* Stores embedding in `embeddings.pkl`

---

### 🔵 `main.py` – Basic Recognition

* Loads stored embeddings
* Detects faces in real-time
* Matches faces using cosine similarity
* Marks attendance in `attendance.csv`

---

### 🟡 `rtsp_fast_detection.py` – CCTV Detection

* Connects to RTSP camera stream
* Uses RetinaFace for detection
* Displays detected faces
* No recognition (only detection)

---

### 🟣 `PRO_AI_ATTENDANCE_SYSTEM.py` – Dashboard

* Streamlit-based UI
* Features:

  * Login system
  * Registration
  * Recognition
  * Attendance view
  * Analytics
  * User management

---

### 🌙 Night Vision Version

* Enhances low-light images
* Uses:

  * CLAHE
  * Gamma correction
* Improves detection in dark environments

---

### 🔐 `PRO_AI_ATTENDANCE_SYSTEM_SECURE.py` – Secure System

* Advanced version with:

  * Anti-spoof detection (eye detection)
  * Multiple embeddings per user
  * Higher accuracy threshold (0.82)
  * Improved recognition performance

---

## 📊 System Architecture

```
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
Comparison (Cosine Similarity)
     ↓
Match / Unknown
     ↓
Attendance Stored (CSV)
     ↓
Dashboard (Streamlit)
```

---

## 📁 Data Files

### 📌 `embeddings.pkl`

Stores face embeddings:

```
{ name : [embedding vectors] }
```

---

### 📌 `attendance.csv`

Stores attendance records:

| Name | Date | Time |
| ---- | ---- | ---- |

---

## ▶️ Usage

### 🔹 Register Student

```bash
python register.py
```

---

### 🔹 Run Recognition

```bash
python main.py
```

---

### 🔹 Run Dashboard

```bash
streamlit run PRO_AI_ATTENDANCE_SYSTEM.py
```

---

### 🔹 Run Secure System

```bash
streamlit run PRO_AI_ATTENDANCE_SYSTEM_SECURE.py
```

---

## 🚀 Key Advantages

* Real-time processing
* High accuracy with FaceNet
* Secure (anti-spoof detection)
* Works with webcam & CCTV
* User-friendly dashboard
* Scalable architecture

---

## ⚠️ Limitations

* Haar Cascade is less accurate than modern detectors
* Performance depends on lighting conditions
* Requires threshold tuning
* Limited multi-camera scalability

---

## 🔮 Future Improvements

* Integrate YOLOv8 for face detection
* Add database (MySQL / Firebase)
* Multi-camera tracking
* Mask detection integration
* Upgrade to ArcFace model

---

## 👨‍💻 Author

**Nilay Purohit**

---

## 📌 License

This project is for educational and research purposes.

---
