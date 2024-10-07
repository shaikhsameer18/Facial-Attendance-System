import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Load the face data and names
with open('data/faces_data.pkl', 'rb') as f:
    faces_data = pickle.load(f)

with open('data/names.pkl', 'rb') as f:
    names = pickle.load(f)

# Flatten the faces data for the model
faces_data = faces_data.reshape(faces_data.shape[0], -1)

# Create and train the model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(faces_data, names)

# Save the trained model
with open('data/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")
