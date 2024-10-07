import cv2
import numpy as np
import pickle
import streamlit as st
import pandas as pd
import os
from datetime import datetime

# Load the face recognition model and data
model_path = "data/model.pkl"
names_path = "data/names.pkl"
ids_path = "data/ids.pkl"
attendance_file = "Attendance/attendance.csv"

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(names_path, "rb") as f:
    names = pickle.load(f)

with open(ids_path, "rb") as f:
    ids = pickle.load(f)


# Function to log attendance
def log_attendance(name, student_id):
    # Check if the attendance directory exists; create it if it doesn't
    if not os.path.exists("Attendance"):
        os.makedirs("Attendance")

    # Append the attendance data
    with open(attendance_file, "a") as f:
        f.write(f"{name},{student_id},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


# Function to recognize faces
def recognize_faces():
    video = cv2.VideoCapture(0)  # Use 0 for the default camera

    while True:
        ret, frame = video.read()
        if not ret:
            st.write("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        facedetect = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        recognized = False  # Flag to check if a person has been recognized

        for x, y, w, h in faces:
            crop_img = frame[y : y + h, x : x + w]
            resized_img = cv2.resize(crop_img, (50, 50))  # Ensure it matches the input size for training
            reshaped_img = resized_img.flatten().reshape(1, -1)  # Flatten and reshape for prediction

            predicted_name = model.predict(reshaped_img)[0]  # Get the predicted name
            student_id = ids[names.index(predicted_name)]  # Get the corresponding ID

            # Log attendance
            log_attendance(predicted_name, student_id)

            # Draw rectangle around face and display the name
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                frame,
                predicted_name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
            )

            recognized = True  # Set recognized flag

        # Display the frame with recognized faces
        cv2.imshow("Frame", frame)

        if recognized:
            st.write(f"Attendance logged for: {predicted_name} (ID: {student_id})")
        
        # Stop video capture and exit if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()


# Streamlit App layout
st.title("Facial Attendance System")
if st.button("Start Recognizing Faces"):
    recognize_faces()

# Display attendance log
if os.path.exists(attendance_file):
    df = pd.read_csv(attendance_file, names=["Name", "ID", "Time"])
    st.dataframe(df.style.highlight_max(axis=0))
else:
    st.write("No attendance records found.")
