# PRO_AI_ATTENDANCE_SYSTEM_SECURE.py

import streamlit as st
import cv2
import numpy as np
import pickle
import os
import pandas as pd
import datetime
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Smart Secure AI Attendance",
    layout="wide",
    page_icon="🎓"
)

# -------------------------------------------------
# CUSTOM UI STYLE
# -------------------------------------------------
st.markdown("""
<style>
body { background-color: #0E1117; }
.stButton>button {
    width: 100%;
    border-radius: 8px;
    height: 45px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# FILE PATHS
# -------------------------------------------------
EMBEDDINGS_FILE = "embeddings.pkl"
ATTENDANCE_FILE = "attendance.csv"

# -------------------------------------------------
# FACE + EYE DETECTORS
# -------------------------------------------------
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

if "mode" not in st.session_state:
    st.session_state.mode = None

if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0

# -------------------------------------------------
# LOAD / SAVE EMBEDDINGS
# -------------------------------------------------
def load_embeddings():

    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            return pickle.load(f)

    return {}

def save_embeddings(data):

    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(data, f)

# -------------------------------------------------
# ATTENDANCE SYSTEM
# -------------------------------------------------
def mark_attendance(name):

    today = datetime.date.today().strftime("%Y-%m-%d")
    now = datetime.datetime.now().strftime("%H:%M:%S")

    if not os.path.exists(ATTENDANCE_FILE):

        df = pd.DataFrame(columns=["Name","Date","Time"])
        df.to_csv(ATTENDANCE_FILE,index=False)

    df = pd.read_csv(ATTENDANCE_FILE)

    if not ((df["Name"] == name) & (df["Date"] == today)).any():

        new_row = pd.DataFrame(
            [[name,today,now]],
            columns=["Name","Date","Time"]
        )

        df = pd.concat([df,new_row],ignore_index=True)

        df.to_csv(ATTENDANCE_FILE,index=False)

# -------------------------------------------------
# FACE MATCHING
# -------------------------------------------------
def find_match(embedding, db):

    embedding = np.array(embedding).flatten()

    best_name = None
    best_score = -1

    for name, embeddings in db.items():

        for e in embeddings:

            e = np.array(e).flatten()

            if embedding.shape != e.shape:
                continue

            sim = cosine_similarity(
                embedding.reshape(1,-1),
                e.reshape(1,-1)
            )[0][0]

            if sim > best_score:

                best_score = sim
                best_name = name

    # Strong threshold to reduce false detection
    if best_score > 0.82:

        confidence = round(best_score * 100,2)

        return best_name, confidence

    return None,0

# -------------------------------------------------
# LIVENESS CHECK
# -------------------------------------------------
def check_liveness(gray_face):

    eyes = eye_detector.detectMultiScale(gray_face,1.3,5)

    # Real person normally shows 1-2 eyes
    if len(eyes) >= 1:
        return True
    else:
        return False

# -------------------------------------------------
# CAMERA LOOP
# -------------------------------------------------
def run_camera(camera_index, register_name=None):

    cap = cv2.VideoCapture(camera_index)

    frame_placeholder = st.empty()

    db = load_embeddings()

    if st.session_state.mode == "register":

        if register_name not in db:

            db[register_name] = []

    while st.session_state.camera_active:

        ret, frame = cap.read()

        if not ret:
            st.error("Camera not accessible")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray,1.2,5)

        for (x,y,w,h) in faces:

            face = frame[y:y+h, x:x+w]

            gray_face = gray[y:y+h, x:x+w]

            rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # -------------------------------------------------
            # ANTI SPOOF CHECK
            # -------------------------------------------------
            live = check_liveness(gray_face)

            if not live:

                label = "Spoof Detected"
                color = (0,0,255)

                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)

                cv2.putText(frame,label,(x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,color,2)

                continue

            # -------------------------------------------------
            # FACE EMBEDDING
            # -------------------------------------------------
            embedding = DeepFace.represent(
                rgb,
                model_name="Facenet",
                enforce_detection=False
            )[0]["embedding"]

            # -------------------------------------------------
            # REGISTER MODE
            # -------------------------------------------------
            if st.session_state.mode == "register":

                if st.session_state.frame_count % 5 == 0:

                    db[register_name].append(embedding)

                    save_embeddings(db)

                label = f"Registered: {register_name}"

                color = (0,255,0)

            # -------------------------------------------------
            # RECOGNITION MODE
            # -------------------------------------------------
            else:

                match,confidence = find_match(embedding, db)

                if match:

                    label = f"{match} ({confidence}%)"

                    color = (0,255,0)

                    mark_attendance(match)

                else:

                    label = "Unknown"

                    color = (0,0,255)

            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)

            cv2.putText(frame,label,(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,color,2)

        st.session_state.frame_count += 1

        frame_placeholder.image(
            cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),
            channels="RGB"
        )

    cap.release()

# -------------------------------------------------
# LOGIN PAGE
# -------------------------------------------------
def login_page():

    st.title("🔐 Admin Login")

    username = st.text_input("Username")

    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username == "admin" and password == "admin123":

            st.session_state.logged_in = True

            st.success("Login Successful")

            st.rerun()

        else:

            st.error("Invalid Credentials")

# -------------------------------------------------
# DASHBOARD
# -------------------------------------------------
def dashboard():

    st.sidebar.title("🎛 Control Panel")

    menu = st.sidebar.radio(
        "Navigation",
        ["Register","Recognition","Users","Attendance","Analytics","Logout"]
    )

    camera_index = 1

    # REGISTER
    if menu == "Register":

        st.title("Register User")

        name = st.text_input("Enter Name")

        col1,col2 = st.columns(2)

        with col1:

            if st.button("Start Registration"):

                st.session_state.mode = "register"

                st.session_state.camera_active = True

                run_camera(camera_index,name)

        with col2:

            if st.button("Stop Camera"):

                st.session_state.camera_active = False

    # RECOGNITION
    elif menu == "Recognition":

        st.title("Live Recognition")

        col1,col2 = st.columns(2)

        with col1:

            if st.button("Start Recognition"):

                st.session_state.mode = "recognition"

                st.session_state.camera_active = True

                run_camera(camera_index)

        with col2:

            if st.button("Stop Camera"):

                st.session_state.camera_active = False

    # USERS
    elif menu == "Users":

        st.title("Registered Users")

        db = load_embeddings()

        if not db:

            st.info("No users registered")

        else:

            for user in list(db.keys()):

                col1,col2 = st.columns([3,1])

                col1.write(f"**{user}**")

                if col2.button("Delete",key=user):

                    del db[user]

                    save_embeddings(db)

                    st.rerun()

    # ATTENDANCE
    elif menu == "Attendance":

        st.title("Attendance Records")

        if os.path.exists(ATTENDANCE_FILE):

            df = pd.read_csv(ATTENDANCE_FILE)

            st.dataframe(df)

            st.download_button(
                "Download CSV",
                df.to_csv(index=False),
                file_name="attendance.csv"
            )

        else:

            st.info("No attendance data")

    # ANALYTICS
    elif menu == "Analytics":

        st.title("Attendance Analytics")

        if os.path.exists(ATTENDANCE_FILE):

            df = pd.read_csv(ATTENDANCE_FILE)

            df["Date"] = pd.to_datetime(df["Date"])

            daily = df.groupby("Date")["Name"].nunique().reset_index()

            st.bar_chart(daily.set_index("Date"))

        else:

            st.info("No analytics available")

    # LOGOUT
    elif menu == "Logout":

        st.session_state.logged_in = False

        st.rerun()

# -------------------------------------------------
# APP START
# -------------------------------------------------
if not st.session_state.logged_in:

    login_page()

else:

    dashboard()