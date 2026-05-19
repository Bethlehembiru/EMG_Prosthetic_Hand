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
from SEMG.classification.RF import get_rf
from SEMG.classification.GB import get_gb
from SEMG.classification.QDA import get_qda


# =========================================================
# PATHS
# =========================================================
FEATURE_DIR = r"C:\Users\HP\OneDrive\Desktop\EMG_Prosthetic_Hand\data\features2"
SAVE_DIR = r"C:\Users\HP\OneDrive\Desktop\EMG_Prosthetic_Hand\models_loso2"

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
# FEATURE SPLITS
# =========================================================
filtered_X = X_all[:, 0:14]
rectified_X = X_all[:, 14:26]
envelope_X = X_all[:, 26:31]


print("\nGenerating PCA plots...")
plot_pca(X_all, y_all, "PCA - Full Feature Set")
plot_pca(filtered_X, y_all, "PCA - Filtered Features")
plot_pca(rectified_X, y_all, "PCA - Rectified Features")
plot_pca(envelope_X, y_all, "PCA - Envelope Features")


# =========================================================
# LOSO SETUP
# =========================================================
logo = LeaveOneGroupOut()

# ---- score storage (ALL MODELS) ----
scores = {
    "SVM": [],
    "LDA": [],
    "KNN": [],
    "RF": [],
    "GB": [],
    "QDA": []
}

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
    # SVM (GRID SEARCH)
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


    # =====================================================
    # OTHER MODELS
    # =====================================================
    lda_model = Pipeline([("scaler", StandardScaler()), ("model", get_lda())])
    knn_model = Pipeline([("scaler", StandardScaler()), ("model", get_knn())])
    rf_model  = Pipeline([("scaler", StandardScaler()), ("model", get_rf())])
    gb_model  = Pipeline([("scaler", StandardScaler()), ("model", get_gb())])
    qda_model = Pipeline([("scaler", StandardScaler()), ("model", get_qda())])

    models = {
        "SVM": svm_model,
        "LDA": lda_model,
        "KNN": knn_model,
        "RF": rf_model,
        "GB": gb_model,
        "QDA": qda_model
    }


    # =====================================================
    # TRAIN + EVALUATE
    # =====================================================
    for name, model in models.items():

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        scores[name].append(acc)

        print(f"\n--- {name} ---")
        print("Accuracy:", round(acc, 4))
        print(classification_report(y_test, y_pred, zero_division=0))

        cm_norm = confusion_matrix(y_test, y_pred, normalize="true")

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues")
        plt.title(f"{name} - {test_subject}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

        # track best single fold model
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = name


# =========================================================
# FINAL LOSO RESULTS
# =========================================================
print("\n========================================")
print("FINAL LOSO RESULTS")
print("========================================")

results = {}

for name, vals in scores.items():
    vals = np.array(vals)
    mean = np.mean(vals)
    std = np.std(vals)

    results[name] = mean - std  # stability score

    print(f"\n{name}")
    print("Mean:", round(mean, 4))
    print("Std :", round(std, 4))
    print("Min :", round(np.min(vals), 4))
    print("Max :", round(np.max(vals), 4))


# =========================================================
# BEST MODEL (STABILITY-BASED)
# =========================================================
best_model_name_final = max(results, key=results.get)

print("\n========================================")
print("BEST MODEL (STABILITY BASED)")
print("Model:", best_model_name_final)
print("Score:", results[best_model_name_final])
print("========================================")


# rebuild model reference safely
model_builder = {
    "SVM": svm_model,
    "LDA": lda_model,
    "KNN": knn_model,
    "RF": rf_model,
    "GB": gb_model,
    "QDA": qda_model
}

best_model = model_builder[best_model_name_final]

joblib.dump(best_model, os.path.join(SAVE_DIR, "best_model.joblib"))