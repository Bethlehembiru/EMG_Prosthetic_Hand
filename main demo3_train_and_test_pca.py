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

from SEMG.classification.lda import get_lda
from SEMG.classification.knn import get_knn
from SEMG.classification.RF import get_rf
from SEMG.classification.GB import get_gb
from SEMG.classification.QDA import get_qda


# =========================================================
# PATHS
# =========================================================
FEATURE_DIR = r"C:\Users\hp\PycharmProjects\EMG_Prosthetic_Hand\data\features_multi_normalized\mixed"
SAVE_DIR = r"C:\Users\hp\PycharmProjects\EMG_Prosthetic_Hand\models\split"

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
print("X:", X_all.shape)
print("y:", y_all.shape)



# =========================================================
# 80/20 SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_all,
    y_all,
    test_size=0.2,
    random_state=42,
    stratify=y_all
)

print("\nDATA SPLIT")
print("Train:", X_train.shape, "Test:", X_test.shape)


# =========================================================
# SVM (GRID SEARCH)
# =========================================================
svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95)),
    ("model", SVC(kernel="rbf", class_weight="balanced"))
])

param_grid = {
    "model__C": [0.1, 1, 10, 50, 100],
    "model__gamma": ["scale", 0.1, 0.01, 0.001]
}

grid = GridSearchCV(
    svm_pipeline,
    param_grid,
    cv=4,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)
svm_model = grid.best_estimator_


# =========================================================
# OTHER MODELS
# =========================================================
lda_model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
        ("model", get_lda())
    ])
knn_model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
        ("model", get_knn())
    ])
rf_model = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95)),
    ("model", get_rf())
])

gb_model = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95)),
    ("model", get_gb())
])

qda_model = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95)),
    ("model", get_qda())
])


models = {
    "SVM": svm_model,
    "LDA": lda_model,
    "KNN": knn_model,
    "RF": rf_model,
    "GB": gb_model,
    "QDA": qda_model
}


# =========================================================
# TRAIN + EVALUATE
# =========================================================
best_model = None
best_model_name = None
best_accuracy = 0

for name, model in models.items():

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"\n--- {name} ---")
    print("Accuracy:", round(acc, 4))
    print(classification_report(y_test, y_pred, zero_division=0))

    cm_norm = confusion_matrix(y_test, y_pred, normalize="true")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues")
    plt.title(f"{name} - 80/20 Split")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name


# =========================================================
# FINAL RESULTS
# =========================================================
print("\n========================================")
print("FINAL RESULTS (80/20 SPLIT)")
print("========================================")
print("Best Model:", best_model_name)
print("Best Accuracy:", round(best_accuracy, 4))


# =========================================================
# SAVE MODEL
# =========================================================
joblib.dump(best_model, os.path.join(SAVE_DIR, "best_model_multi_pca_2.joblib"))