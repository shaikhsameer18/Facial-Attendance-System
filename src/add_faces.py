import cv2
import pickle
import numpy as np
import os

# Initialize video capture and face detector
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

faces_data = []
i = 0

# Get user input for name and ID
name = input("Enter Your Name: ")
user_id = input("Enter Your ID: ")

# Capture images
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        crop_img = frame[y : y + h, x : x + w, :]
        resized_img = cv2.resize(crop_img, (50, 50))

        if len(faces_data) < 50 and i % 10 == 0:
            faces_data.append(resized_img)

        i += 1
        cv2.putText(
            frame,
            str(len(faces_data)),
            (50, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (50, 50, 255),
            1,
        )
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)

    if k == ord("q") or len(faces_data) >= 50:
        break

video.release()
cv2.destroyAllWindows()

faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(50, -1)

# Save names and faces data
if "names.pkl" not in os.listdir("data/"):
    names = [name] * 50
    ids = [user_id] * 50  # New: Store user IDs
    with open("data/names.pkl", "wb") as f:
        pickle.dump(names, f)
    with open("data/ids.pkl", "wb") as f:  # New: Save IDs
        pickle.dump(ids, f)
else:
    with open("data/names.pkl", "rb") as f:
        names = pickle.load(f)
    with open("data/ids.pkl", "rb") as f:  # Load existing IDs
        ids = pickle.load(f)
    names += [name] * 50
    ids += [user_id] * 50
    with open("data/names.pkl", "wb") as f:
        pickle.dump(names, f)
    with open("data/ids.pkl", "wb") as f:  # Save updated IDs
        pickle.dump(ids, f)

# Save face data
if "faces_data.pkl" not in os.listdir("data/"):
    with open("data/faces_data.pkl", "wb") as f:
        pickle.dump(faces_data, f)
else:
    with open("data/faces_data.pkl", "rb") as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)
    with open("data/faces_data.pkl", "wb") as f:
        pickle.dump(faces, f)
