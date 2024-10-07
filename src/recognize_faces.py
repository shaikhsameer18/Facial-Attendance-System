import cv2
import pickle
import pandas as pd
from datetime import datetime

# Load trained model and data
with open('data/face_recognition_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('data/names.pkl', 'rb') as f:
    names = pickle.load(f)

with open('data/ids.pkl', 'rb') as f:
    ids = pickle.load(f)

# Start video capture
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Create attendance DataFrame
attendance = []

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]  # Keep it colored for consistency
        # Resize and flatten the image to match the training input shape
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)  # 50x50x3 for RGB images

        # Predict name
        predicted_name = model.predict(resized_img)[0]

        # Log attendance
        attendance.append((predicted_name, datetime.now()))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
        cv2.putText(frame, predicted_name, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
    
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

# Log attendance to CSV
attendance_df = pd.DataFrame(attendance, columns=["Name", "Timestamp"])
date = datetime.now().strftime("%d-%m-%Y")
attendance_df.to_csv("Attendance/Attendance_" + date + ".csv", index=False)
print("Attendance logged to Attendance/Attendance_" + date + ".csv")
