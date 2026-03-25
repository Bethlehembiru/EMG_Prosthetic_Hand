import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib  # for saving/loading models

# -----------------------------
# SETTINGS
# -----------------------------
FEATURE_DIR = "data/features"
MODEL_DIR = "data/models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_FILENAME = os.path.join(MODEL_DIR, "svm_emg_model.pkl")

# -----------------------------
# LOAD FEATURE VECTORS
# -----------------------------
X_list = []
y_list = []

for participant_folder in os.listdir(FEATURE_DIR):
    participant_path = os.path.join(FEATURE_DIR, participant_folder)
    if not os.path.isdir(participant_path):
        continue

    for file in os.listdir(participant_path):
        if file.endswith(".npz"):
            file_path = os.path.join(participant_path, file)
            data = np.load(file_path)
            X_list.append(data["features"])
            y_list.append(data["labels"])

# Concatenate all participants
X = np.vstack(X_list)
y = np.hstack(y_list)

print(f"Total samples: {X.shape[0]}, Features per sample: {X.shape[1]}")

# -----------------------------
# NORMALIZATION
# -----------------------------
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)  # compute mean/std across training set

# -----------------------------
# TRAIN CLASSIFIER
# -----------------------------
clf = SVC(kernel="linear", probability=True)
clf.fit(X_norm, y)
print("Classifier trained successfully.")

# -----------------------------
# SAVE MODEL + SCALER
# -----------------------------
joblib.dump({"scaler": scaler, "classifier": clf}, MODEL_FILENAME)
print(f"Trained model saved to {MODEL_FILENAME}")

# -----------------------------
# USAGE EXAMPLE (later for testing)
# -----------------------------
# To load the model and apply normalization on new data:
# model_data = joblib.load(MODEL_FILENAME)
# scaler = model_data["scaler"]
# clf = model_data["classifier"]
# X_new_norm = scaler.transform(X_new)
# predictions = clf.predict(X_new_norm)
