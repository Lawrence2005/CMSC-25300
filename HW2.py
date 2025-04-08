import numpy as np

# #### Part a #####
# Load in training data and labels
# File available on Canvas
face_data_dict = np. load ("face_emotion_data.npz")
face_features = face_data_dict ["X"]
face_labels = face_data_dict ["y"]
n, p = face_features.shape
# Solve the least - squares solution . weights is the array of weight coefficients

weights = (face_features.transpose() @ face_features).inv() @ face_features.transpose() @ face_labels
print ( f" Part 4a. Found weights :\n{ weights }")