import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

from SEMG.classification.classifier import EMGClassifier


# -----------------------------
# PATHS (SOURCE + SAVE)
# -----------------------------
FEATURE_DIR = r"C:\Users\HP\OneDrive\Desktop\EMG_Prosthetic_Hand\data\features"
SAVE_DIR = r"C:\Users\HP\OneDrive\Desktop\EMG_Prosthetic_Hand\models"

os.makedirs(SAVE_DIR, exist_ok=True)


# -----------------------------
# LOAD DATA
# -----------------------------
X_all = []
y_all = []

for file in os.listdir(FEATURE_DIR):
    if not file.endswith(".npz"):
        continue

    path = os.path.join(FEATURE_DIR, file)
    data = np.load(path)

    X_all.append(data["X"])
    y_all.append(data["y"])

X = np.vstack(X_all)
y = np.concatenate(y_all)

print("Loaded dataset:")
print("X shape:", X.shape)
print("y shape:", y.shape)


# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# -----------------------------
# TRAIN MODELS
# -----------------------------
classifier = EMGClassifier()
classifier.train(X_train, y_train)


# -----------------------------
# EVALUATION
# -----------------------------
print("\nModel evaluation:")
models = classifier.get_models()

for name, model in models.items():
    acc = model.score(X_test, y_test)
    print(f"{name} accuracy: {acc:.3f}")


# -----------------------------
# SAVE MODELS
# -----------------------------
print("\nSaving models...")

for name, model in models.items():
    path = os.path.join(SAVE_DIR, f"{name}.joblib")
    joblib.dump(model, path)
    print(f"Saved {name} → {path}")

print("\nAll models saved successfully.")