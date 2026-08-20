"""Optional offline enrollment via local webcam window (no browser needed).

Requires the non-headless `opencv-python` package for cv2.imshow to work.
For normal use, prefer the "Register New Face" page in app.py instead —
it works the same way locally and remotely.
"""
import cv2

import face_utils as fu

video = cv2.VideoCapture(0)
if not video.isOpened():
    raise RuntimeError("Could not open webcam. Is another app using it?")

name = input("Enter Your Name: ").strip()
user_id = input("Enter Your ID: ").strip()

samples = []
i = 0
try:
    while len(samples) < fu.SAMPLES_PER_USER:
        ret, frame = video.read()
        if not ret:
            break

        for (x, y, w, h) in fu.detect_faces(frame):
            crop = cv2.resize(frame[y : y + h, x : x + w], fu.FACE_SIZE)
            if i % 10 == 0:
                samples.append(crop)
            i += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
            cv2.putText(frame, f"{len(samples)}/{fu.SAMPLES_PER_USER}", (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)

        cv2.imshow("Register Face - press q to cancel", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    video.release()
    cv2.destroyAllWindows()

if len(samples) < fu.SAMPLES_PER_USER:
    print(f"Only captured {len(samples)} samples, aborting (need {fu.SAMPLES_PER_USER}).")
else:
    fu.save_face_samples(name, user_id, samples)
    fu.train_and_save_model()
    print(f"Registered {name} (ID {user_id}) and retrained model.pkl")
