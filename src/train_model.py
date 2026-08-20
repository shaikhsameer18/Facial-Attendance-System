"""Retrain data/model.pkl from data/faces_data.pkl + data/names.pkl.

Only needed if you enrolled faces without using the app (e.g. via
add_faces.py) — the Streamlit app retrains automatically after each
registration.
"""
import face_utils as fu

fu.train_and_save_model()
print("Model trained and saved to", fu.MODEL_PATH)
