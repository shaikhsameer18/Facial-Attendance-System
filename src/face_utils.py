"""Shared face enrollment / recognition / attendance logic.

Used by both the Streamlit app (app.py) and the optional CLI scripts.
Keeping this in one place avoids the model-path drift and duplicate-logging
bugs that existed when each entry point re-implemented this logic.
"""
import os
import pickle
from datetime import date, datetime

import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

DATA_DIR = "data"
ATTENDANCE_DIR = "Attendance"
CASCADE_PATH = os.path.join(DATA_DIR, "haarcascade_frontalface_default.xml")
FACES_PATH = os.path.join(DATA_DIR, "faces_data.pkl")
NAMES_PATH = os.path.join(DATA_DIR, "names.pkl")
IDS_PATH = os.path.join(DATA_DIR, "ids.pkl")
MODEL_PATH = os.path.join(DATA_DIR, "model.pkl")

FACE_SIZE = (50, 50)
SAMPLES_PER_USER = 50
# Euclidean distance (on flattened 50x50x3 pixel vectors) above which a face
# is treated as "unknown" rather than forced into the nearest registered
# class. Tuned empirically for this pixel-based KNN approach.
DISTANCE_THRESHOLD = 4500

_face_detector = None


def get_face_detector():
    global _face_detector
    if _face_detector is None:
        _face_detector = cv2.CascadeClassifier(CASCADE_PATH)
    return _face_detector


def detect_faces(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return get_face_detector().detectMultiScale(gray, 1.3, 5)


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pickle(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def has_trained_model():
    return all(os.path.exists(p) for p in (MODEL_PATH, NAMES_PATH, IDS_PATH))


def load_model():
    return _load_pickle(MODEL_PATH), _load_pickle(NAMES_PATH), _load_pickle(IDS_PATH)


def registered_users():
    """Return sorted list of (id, name) for every unique enrolled person."""
    if not (os.path.exists(NAMES_PATH) and os.path.exists(IDS_PATH)):
        return []
    names = _load_pickle(NAMES_PATH)
    ids = _load_pickle(IDS_PATH)
    seen = dict(zip(ids, names))
    return sorted(seen.items())


def save_face_samples(name, user_id, samples_bgr):
    """samples_bgr: list of BGR crops already resized to FACE_SIZE."""
    new_faces = np.asarray(samples_bgr).reshape(len(samples_bgr), -1)
    new_names = [name] * len(samples_bgr)
    new_ids = [str(user_id)] * len(samples_bgr)

    if os.path.exists(FACES_PATH):
        new_faces = np.append(_load_pickle(FACES_PATH), new_faces, axis=0)
    _save_pickle(new_faces, FACES_PATH)

    if os.path.exists(NAMES_PATH):
        new_names = _load_pickle(NAMES_PATH) + new_names
    _save_pickle(new_names, NAMES_PATH)

    if os.path.exists(IDS_PATH):
        new_ids = _load_pickle(IDS_PATH) + new_ids
    _save_pickle(new_ids, IDS_PATH)


def train_and_save_model():
    faces_data = _load_pickle(FACES_PATH)
    faces_data = faces_data.reshape(faces_data.shape[0], -1)
    names = _load_pickle(NAMES_PATH)

    n_neighbors = max(1, min(5, len(set(names))))
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(faces_data, names)
    _save_pickle(model, MODEL_PATH)
    return model


def recognize_face(model, crop_bgr):
    """Return predicted name, or None if the face is not confidently known."""
    resized = cv2.resize(crop_bgr, FACE_SIZE).flatten().reshape(1, -1)
    distances, _ = model.kneighbors(resized, n_neighbors=1)
    if distances[0][0] > DISTANCE_THRESHOLD:
        return None
    return model.predict(resized)[0]


def attendance_file_for(day=None):
    day = day or date.today()
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    return os.path.join(ATTENDANCE_DIR, f"Attendance_{day.isoformat()}.csv")


def load_attendance(day=None):
    path = attendance_file_for(day)
    if not os.path.exists(path):
        return pd.DataFrame(columns=["Name", "ID", "Time"])
    return pd.read_csv(path)


def already_logged_today(user_id, day=None):
    df = load_attendance(day)
    return str(user_id) in df["ID"].astype(str).values


def log_attendance(name, user_id, day=None):
    """Append one row, but only once per person per day."""
    if already_logged_today(user_id, day):
        return False
    path = attendance_file_for(day)
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        if is_new:
            f.write("Name,ID,Time\n")
        f.write(f"{name},{user_id},{datetime.now().strftime('%H:%M:%S')}\n")
    return True
