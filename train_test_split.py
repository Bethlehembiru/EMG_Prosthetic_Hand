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
FEATURE_DIR = r"C:\Users\HP\OneDrive\Desktop\EMG_Prosthetic_Hand\data\features2"
SAVE_DIR = r"C:\Users\HP\OneDrive\Desktop\EMG_Prosthetic_Hand\model_train_test2"

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
# PCA FUNCTION
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
            label=f"Class {label}"
        )

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.legend()
    plt.grid()

    plt.show()


# =========================================================
# FEATURE SPLITS
# =========================================================
filtered_X = X_all[:, 0:14]
rectified_X = X_all[:, 14:26]
envelope_X = X_all[:, 26:31]


# =========================================================
# PCA VISUALIZATION
# =========================================================
print("\nGenerating PCA plots...")

plot_pca(X_all, y_all, "PCA - Full Feature Set")
plot_pca(filtered_X, y_all, "PCA - Filtered Features")
plot_pca(rectified_X, y_all, "PCA - Rectified Features")
plot_pca(envelope_X, y_all, "PCA - Envelope Features")


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
# SCORE STORAGE
# =========================================================
scores = {
    "SVM": 0,
    "LDA": 0,
    "KNN": 0,
    "RF": 0,
    "GB": 0,
    "QDA": 0
}

best_model = None
best_model_name = None
best_accuracy = 0


# =========================================================
# SVM (GRID SEARCH)
# =========================================================
svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(kernel="rbf", class_weight="balanced"))
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

svm_model = grid.best_estimator_

# FIXED ERROR HERE
print("\nBest SVM Params:", grid.best_params_)


# =========================================================
# OTHER MODELS
# =========================================================
lda_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", get_lda())
])

knn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", get_knn())
])

rf_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", get_rf())
])

gb_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", get_gb())
])

qda_model = Pipeline([
    ("scaler", StandardScaler()),
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
for name, model in models.items():

    print("\n========================================")
    print(f"{name} RESULTS")
    print("========================================")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    scores[name] = acc

    print("Accuracy:", round(acc, 4))

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================
    cm_norm = confusion_matrix(
        y_test,
        y_pred,
        normalize="true"
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues"
    )

    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.show()

    # =====================================================
    # TRACK BEST MODEL
    # =====================================================
    if acc > best_accuracy:

        best_accuracy = acc
        best_model = model
        best_model_name = name


# =========================================================
# FINAL RESULTS
# =========================================================
print("\n========================================")
print("FINAL RESULTS")
print("========================================")

for name, acc in scores.items():

    print(f"{name}: {round(acc, 4)}")


# =========================================================
# BEST MODEL
# =========================================================
print("\n========================================")
print("BEST MODEL")
print("========================================")

print("Model:", best_model_name)
print("Accuracy:", round(best_accuracy, 4))


# =========================================================
# SAVE BEST MODEL
# =========================================================
save_path = os.path.join(
    SAVE_DIR,
    "best_model.joblib"
)

joblib.dump(best_model, save_path)

print("\nBest model saved successfully.")
print("Saved to:", save_path)