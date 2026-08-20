"""Facial Attendance System — Streamlit app.

Camera capture runs in the visitor's browser via streamlit-webrtc and streams
frames to the server for detection/recognition. This is what makes the app
work when deployed remotely (Streamlit Community Cloud, Docker, etc.) — a
server has no physical webcam, so the previous cv2.VideoCapture(0) +
cv2.imshow approach only ever worked on localhost.
"""
import queue
import threading
from datetime import date

import av
import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import face_utils as fu

st.set_page_config(page_title="Facial Attendance System", page_icon="🧑‍💻", layout="wide")

RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


# --------------------------------------------------------------------------
# Attendance ("Take Attendance") video processor
# --------------------------------------------------------------------------
class AttendanceProcessor:
    def __init__(self):
        self.model = self.names = self.ids = None
        if fu.has_trained_model():
            self.model, self.names, self.ids = fu.load_model()
        self.last_seen = {}  # name -> most recent recognition, for the sidebar

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.model is not None:
            for (x, y, w, h) in fu.detect_faces(img):
                crop = img[y : y + h, x : x + w]
                predicted = fu.recognize_face(self.model, crop)
                if predicted is None:
                    label, color = "Unknown", (0, 0, 255)
                else:
                    idx = self.names.index(predicted)
                    student_id = self.ids[idx]
                    fu.log_attendance(predicted, student_id)
                    self.last_seen[predicted] = student_id
                    label, color = f"{predicted} ({student_id})", (0, 200, 0)

                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --------------------------------------------------------------------------
# Enrollment ("Register New Face") video processor
# --------------------------------------------------------------------------
class CollectorProcessor:
    def __init__(self):
        self.lock = threading.Lock()
        self.collecting = False
        self.samples = []
        self.target = fu.SAMPLES_PER_USER

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        faces = fu.detect_faces(img)
        for (x, y, w, h) in faces:
            with self.lock:
                if self.collecting and len(self.samples) < self.target:
                    crop = cv2.resize(img[y : y + h, x : x + w], fu.FACE_SIZE)
                    self.samples.append(crop)
                count = len(self.samples)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, f"{count}/{self.target}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🧑‍💻 Facial Attendance System")

page = st.sidebar.radio("Menu", ["Take Attendance", "Register New Face", "Attendance Records"])

if page == "Take Attendance":
    st.subheader("Take Attendance")

    if not fu.has_trained_model():
        st.warning("No trained model yet. Register at least one face first.")
    else:
        st.caption("Allow camera access in your browser. Recognized faces are logged once per person per day.")
        ctx = webrtc_streamer(
            key="attendance",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=AttendanceProcessor,
            media_stream_constraints={"video": True, "audio": False},
        )
        if ctx.video_processor:
            st.write("Recognized this session:", ctx.video_processor.last_seen or "—")

elif page == "Register New Face":
    st.subheader("Register New Face")
    name = st.text_input("Full name")
    user_id = st.text_input("ID / roll number")

    ctx = webrtc_streamer(
        key="register",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=CollectorProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

    col1, col2 = st.columns(2)
    start_disabled = not (name and user_id and ctx.video_processor)
    if col1.button("Start capturing 50 samples", disabled=start_disabled):
        with ctx.video_processor.lock:
            ctx.video_processor.samples = []
            ctx.video_processor.collecting = True
        st.info("Move your head slightly while the box tracks your face.")

    if ctx.video_processor:
        with ctx.video_processor.lock:
            count = len(ctx.video_processor.samples)
            done = count >= ctx.video_processor.target
        st.progress(count / fu.SAMPLES_PER_USER)

        if done:
            with ctx.video_processor.lock:
                ctx.video_processor.collecting = False
                samples = list(ctx.video_processor.samples)
            if col2.button("Save & train model"):
                fu.save_face_samples(name, user_id, samples)
                fu.train_and_save_model()
                with ctx.video_processor.lock:
                    ctx.video_processor.samples = []
                st.success(f"Registered {name} (ID {user_id}) and retrained the model.")

    st.divider()
    users = fu.registered_users()
    st.write(f"**Enrolled people:** {len(users)}")
    if users:
        st.dataframe(
            {"ID": [u[0] for u in users], "Name": [u[1] for u in users]},
            hide_index=True,
        )

else:
    st.subheader("Attendance Records")
    picked_day = st.date_input("Date", value=date.today())
    df = fu.load_attendance(picked_day)
    if df.empty:
        st.info("No attendance recorded for this date.")
    else:
        st.dataframe(df, hide_index=True)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"Attendance_{picked_day.isoformat()}.csv",
            mime="text/csv",
        )
