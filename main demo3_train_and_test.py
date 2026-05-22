import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
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
FEATURE_DIR = r"C:\Users\hp\PycharmProjects\EMG_Prosthetic_Hand\data\features_close\mixed"
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
# FEATURE TYPE DETECTION
# =========================================================
def detect_feature_type(n_features):
    if n_features == 5:
        return "envelope"
    elif n_features == 12:
        return "amplitude"
    elif n_features == 14:
        return "filtered"
    elif n_features == 31:
        return "combined"
    else:
        return "unknown"


feature_type = detect_feature_type(X_all.shape[1])

print("\n================================")
print("FEATURE TYPE:", feature_type)
print("FEATURE SHAPE:", X_all.shape)
print("================================")


# =========================================================
# SAFE PCA VISUALIZATION
# =========================================================
def plot_pca(X, y, title):

    if X.shape[1] < 2:
        print(f"Skipping PCA: {title} (not enough features)")
        return

    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    print(f"\n{title}")
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Total variance:", np.sum(pca.explained_variance_ratio_))

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


print("\nGenerating PCA plot...")
plot_pca(X_all, y_all, f"PCA - {feature_type}")


# =========================================================
# TRAIN / TEST SPLIT (80/20)
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all,
    test_size=0.2,
    random_state=42,
    stratify=y_all
)


# =========================================================
# MODELS
# =========================================================
svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(kernel="rbf", class_weight="balanced"))
])

models = {
    "SVM": svm_pipeline,
    "LDA": Pipeline([("scaler", StandardScaler()), ("model", get_lda())]),
    "KNN": Pipeline([("scaler", StandardScaler()), ("model", get_knn())]),
    "RF": Pipeline([("scaler", StandardScaler()), ("model", get_rf())]),
    "GB": Pipeline([("scaler", StandardScaler()), ("model", get_gb())]),
    "QDA": Pipeline([("scaler", StandardScaler()), ("model", get_qda())])
}


# =========================================================
# TRAIN + EVALUATE
# =========================================================
scores = {}

for name, model in models.items():

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    scores[name] = acc

    print(f"\n--- {name} ---")
    print("Accuracy:", round(acc, 4))
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, normalize="true")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues")
    plt.title(f"{name} - 80/20 Split")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


# =========================================================
# BEST MODEL
# =========================================================
best_model_name = max(scores, key=scores.get)
best_model = models[best_model_name]

print("\n================================")
print("BEST MODEL:", best_model_name)
print("ACCURACY:", scores[best_model_name])
print("================================")

joblib.dump(best_model, os.path.join(SAVE_DIR, "best_model_close.joblib"))