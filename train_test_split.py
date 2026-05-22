import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_DIR = os.path.join(BASE_DIR, "data", "features77")
SAVE_DIR = os.path.join(BASE_DIR, "model_train_test77")

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================
X_all, y_all, groups = [], [], []

print("\nLoading dataset...\n")

for file in sorted(os.listdir(FEATURE_DIR)):
    if not file.endswith(".npz"):
        continue

    path = os.path.join(FEATURE_DIR, file)
    data = np.load(path)

    X = data["X"]
    y = data["y"]
    participant_id = file.replace("_features.npz", "")

    X_all.append(X)
    y_all.append(y)
    groups.extend([participant_id] * len(y))

    print(participant_id, "Samples:", len(y), "Shape:", X.shape)

X_all = np.vstack(X_all)
y_all = np.concatenate(y_all)
groups = np.array(groups)

print("\nFINAL DATASET")
print("X:", X_all.shape, "(Should be N samples, 4 features)")
print("y:", y_all.shape)

# =========================================================
# PCA VISUALIZATION
# =========================================================
def plot_pca(X, y, title):
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(7, 6))
    for label in np.unique(y):
        idx = y == label
        plt.scatter(
            X_pca[idx, 0],
            X_pca[idx, 1],
            s=12,
            alpha=0.7,
            label=f"Class {label}"
        )

    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

print("\nGenerating PCA plot for the 4-feature set...")
plot_pca(X_all, y_all, "PCA - MAV, WL, ZC, SSC Features")

# =========================================================
# TRAIN TEST SPLIT (80 / 20)
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_all,
    y_all,
    test_size=0.2,
    random_state=42,
    stratify=y_all
)

print("\n========================================")
print("TRAIN TEST SPLIT")
print("========================================")
print("Train:", len(X_train))
print("Test :", len(X_test))

# =========================================================
# SVM (GRID SEARCH WITH PROBABILITY)
# =========================================================
print("\nTraining Optimized RBF SVM (This may take a moment)...")

# CRITICAL: probability=True is required for the Deadzone logic in real-time
svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(kernel="rbf", class_weight="balanced", probability=True))
])

param_grid = {
    "model__C": [0.1, 1, 10, 50, 100],
    "model__gamma": ["scale", 0.1, 0.01, 0.001]
}

grid = GridSearchCV(
    estimator=svm_pipeline,
    param_grid=param_grid,
    cv=4,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)
best_svm = grid.best_estimator_

print("Best SVM Params:", grid.best_params_)

# =========================================================
# EVALUATE
# =========================================================
print("\n========================================")
print("FINAL RESULTS")
print("========================================")

y_pred = best_svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", round(acc, 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion Matrix
cm_norm = confusion_matrix(y_test, y_pred, normalize="true")

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues"
)
plt.title("RBF SVM Confusion Matrix")
plt.xlabel("Predicted State")
plt.ylabel("True State")
plt.show()

# =========================================================
# SAVE PIPELINE
# =========================================================
save_path = os.path.join(SAVE_DIR, "best_svm_model.joblib")
joblib.dump(best_svm, save_path)

print("\n========================================")
print("Pipeline (Scaler + SVM) saved successfully.")
print("Saved to:", save_path)
print("========================================")