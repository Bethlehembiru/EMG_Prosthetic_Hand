import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from SEMG.classification.lda import get_lda
from SEMG.classification.knn import get_knn


# =========================================================
# PATHS
# =========================================================
FEATURE_DIR = r"C:\Users\hp\PycharmProjects\EMG_Prosthetic_Hand\data\features"
SAVE_DIR = r"C:\Users\hp\PycharmProjects\EMG_Prosthetic_Hand\models_loso"

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
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], s=12, label=f"Class {label}")

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid()
    plt.show()


# =========================================================
# FEATURE SPLITS (CRITICAL)
# Must match EMGFeatureExtractor exactly
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
# LOSO SETUP
# =========================================================
logo = LeaveOneGroupOut()

svm_scores, lda_scores, knn_scores = [], [], []

best_model = None
best_model_name = None
best_accuracy = 0


# =========================================================
# LOSO LOOP
# =========================================================
for fold, (train_idx, test_idx) in enumerate(logo.split(X_all, y_all, groups), 1):

    print("\n========================================")
    print(f"LOSO FOLD {fold}")
    print("========================================")

    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    test_subject = groups[test_idx][0]

    print("Testing on:", test_subject)
    print("Train:", len(X_train), "Test:", len(X_test))


    # =====================================================
    # SVM GRID SEARCH
    # =====================================================
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
        cv=4,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    svm_model = grid.best_estimator_

    print("Best SVM Params:", grid.best_params_)


    # =====================================================
    # OTHER MODELS
    # =====================================================
    lda_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", get_lda())
    ])

    knn_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", get_knn())
    ])

    models = {
        "SVM": svm_model,
        "LDA": lda_model,
        "KNN": knn_model
    }


    # =====================================================
    # TRAIN + EVALUATE
    # =====================================================
    for name, model in models.items():

        print(f"\n--- {name} ---")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        print("Accuracy:", round(acc, 4))
        print(classification_report(y_test, y_pred, zero_division=0))

        cm = confusion_matrix(y_test, y_pred)
        cm_norm = confusion_matrix(y_test, y_pred, normalize="true")

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues")
        plt.title(f"{name} - {test_subject}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

        if name == "SVM":
            svm_scores.append(acc)
        elif name == "LDA":
            lda_scores.append(acc)
        elif name == "KNN":
            knn_scores.append(acc)

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = name


# =========================================================
# FINAL RESULTS
# =========================================================
print("\n========================================")
print("FINAL LOSO RESULTS")
print("========================================")

print("SVM:", np.mean(svm_scores))
print("LDA:", np.mean(lda_scores))
print("KNN:", np.mean(knn_scores))


# =========================================================
# SAVE BEST MODEL
# =========================================================
joblib.dump(
    best_model,
    os.path.join(SAVE_DIR, "best_model.joblib")
)

print("\nBest Model:", best_model_name)
print("Best Accuracy:", best_accuracy)