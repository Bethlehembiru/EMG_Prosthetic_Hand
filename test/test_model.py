import numpy as np
import os
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix

# ==============================
# SETTINGS
# ==============================
FEATURES_DIR = "data/features"  # where your test feature files are stored
MODEL_PATH = "model_train_test77/best_svm_model.joblib"
SCALER_PATH = "model_train_test77/best_svm_model.joblib"

# ==============================
# LOAD TEST DATA
# ==============================
# Assuming you saved test features as X_test.npy and y_test.npy
X_test = np.load(os.path.join(FEATURES_DIR, "X_test.npy"))
y_test = np.load(os.path.join(FEATURES_DIR, "y_test.npy"))

# ==============================
# LOAD TRAINED SCALER
# ==============================
scaler = joblib.load(SCALER_PATH)

# ==============================
# NORMALIZE TEST FEATURES
# ==============================
X_test_scaled = scaler.transform(X_test)

# ==============================
# LOAD TRAINED CLASSIFIER
# ==============================
classifier = joblib.load(MODEL_PATH)

# ==============================
# PREDICT
# ==============================
y_pred = classifier.predict(X_test_scaled)

# ==============================
# EVALUATE
# ==============================
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Test Accuracy: {acc*100:.2f}%")
print("Confusion Matrix:")
print(cm)