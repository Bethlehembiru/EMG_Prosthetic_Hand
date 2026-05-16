import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from SEMG.classification.lda import get_lda
from SEMG.classification.knn import get_knn


# =========================
# PATHS
# =========================
FEATURE_DIR = r"C:\Users\HP\OneDrive\Desktop\EMG_Prosthetic_Hand\data\features"
SAVE_DIR = r"/models2"

os.makedirs(SAVE_DIR, exist_ok=True)


# =========================
# LOAD DATA
# =========================
X_all, y_all = [], []

for file in os.listdir(FEATURE_DIR):
    if file.endswith(".npz"):
        data = np.load(os.path.join(FEATURE_DIR, file))
        X_all.append(data["X"])
        y_all.append(data["y"])

X = np.vstack(X_all)
y = np.concatenate(y_all)

print("\nDataset Loaded")
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nClass distribution:")
for c in np.unique(y):
    print(f"Class {c}: {np.sum(y == c)}")


# =========================
# PCA VISUALIZATION (SEPARABILITY CHECK)
# =========================
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(7, 6))

for label in np.unique(y):
    idx = y == label
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], s=10, label=f"Class {label}")

plt.title("PCA Feature Separability")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid()
plt.show()


# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# SVM (GRID SEARCH = BEST MODEL)
# =========================
print("\n================ SVM GRID SEARCH ================")

svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(kernel="rbf", class_weight="balanced"))
])

param_grid = {
    "model__C": [0.1, 1, 10, 50, 100],
    "model__gamma": ["scale", 0.1, 0.01, 0.001]
}

grid = GridSearchCV(
    svm_pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\nBest SVM Params:", grid.best_params_)
print("Best CV Score:", grid.best_score_)


# =========================
# MODELS TO COMPARE
# =========================
models = {
    "SVM (Best)": best_model,

    "LDA": Pipeline([
        ("scaler", StandardScaler()),
        ("model", get_lda())
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", get_knn())
    ])
}


# =========================
# TRAIN + EVALUATE
# =========================
results = {}

for name, model in models.items():
    print(f"\n================ {name} ================\n")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"Accuracy: {acc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    cm_norm = confusion_matrix(y_test, y_pred, normalize="true")

    print("Confusion Matrix:\n", cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues")

    plt.title(f"{name} - Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # save model
    model_path = os.path.join(SAVE_DIR, f"{name.replace(' ', '_').lower()}.joblib")
    joblib.dump(model, model_path)

    print(f"{name} saved → {model_path}")


# =========================
# FINAL SUMMARY
# =========================
print("\nFINAL RESULTS SUMMARY")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")


# =========================
# SAVE BEST MODEL EXPLICITLY
# =========================
best_name = max(results, key=results.get)
print("\nBEST MODEL OVERALL:", best_name)

joblib.dump(
    models[best_name],
    os.path.join(SAVE_DIR, "best_model.joblib")
)

print("Best model saved as best_model.joblib")