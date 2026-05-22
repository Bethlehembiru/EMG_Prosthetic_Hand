import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

file_path = input("Enter feature .npz file path:\n")

data = np.load(file_path)

X = data["X"]
y = data["y"]

print("Feature shape:", X.shape)

# =====================================================
# FEATURE NAMES (MATCH YOUR EXTRACTOR ORDER)
# =====================================================

feature_names = (

    # ---------------- FILTERED ----------------
    ["ZC", "SSC", "WAMP"] +
    ["Hjorth_Activity", "Hjorth_Mobility", "Hjorth_Complexity"] +
    ["MNF", "MDF", "PeakFreq", "SpectralEntropy"] +
    [f"AR_{i+1}" for i in range(4)] +

    # ---------------- RECTIFIED ----------------
    ["Min_R", "Max_R", "Mean_R", "Std_R",
     "Var_R", "RMS", "MAV", "IEMG",
     "WL", "AAC", "Skew_R", "Kurt_R"] +

    # ---------------- ENVELOPE ----------------
    ["AFB", "Env_Mean", "Env_Max", "Env_Min", "Env_Std"]
)

# safety check
if X.shape[1] != len(feature_names):
    print("WARNING: Feature count mismatch!")
    print("X columns:", X.shape[1])
    print("Feature names:", len(feature_names))

# =====================================================
# 1. HEATMAP (FEATURE STRUCTURE VIEW)
# =====================================================
plt.figure(figsize=(12, 6))

plt.imshow(X.T, aspect='auto', interpolation='nearest')

plt.title("Feature Heatmap (Structured Features vs Windows)")
plt.xlabel("Windows")
plt.ylabel("Features")

plt.yticks(
    ticks=np.arange(len(feature_names)),
    labels=feature_names,
    fontsize=8
)

plt.colorbar()
plt.tight_layout()
plt.show()

# =====================================================
# 2. RAW FEATURE SCATTER (FIRST TWO FEATURES)
# =====================================================
plt.figure()

f1, f2 = 0, 1

for label in np.unique(y):
    idx = y == label
    plt.scatter(
        X[idx, f1],
        X[idx, f2],
        label=f"Class {label}",
        s=10
    )

plt.title("Feature Space (Raw Features)")
plt.xlabel(feature_names[f1])
plt.ylabel(feature_names[f2])
plt.legend()
plt.grid()
plt.show()

# =====================================================
# 3. PCA VISUALIZATION (GLOBAL STRUCTURE)
# =====================================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure()

for label in np.unique(y):
    idx = y == label
    plt.scatter(
        X_pca[idx, 0],
        X_pca[idx, 1],
        label=f"Class {label}",
        s=10
    )

plt.title("PCA Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid()
plt.show()

# =====================================================
# 4. PCA INFO (VERY IMPORTANT FOR THESIS)
# =====================================================
print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)
print("Total variance captured (PC1+PC2):",
      np.sum(pca.explained_variance_ratio_))