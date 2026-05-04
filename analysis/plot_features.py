import numpy as np
import matplotlib.pyplot as plt

file_path = input("Enter feature .npz file path:\n")

data = np.load(file_path)

X = data["X"]
y = data["y"]

print("Feature shape:", X.shape)

# =========================
# 1. HEATMAP (BEST OVERALL VIEW)
# =========================
plt.figure()
plt.imshow(X.T, aspect='auto')
plt.title("Feature Heatmap (Features vs Windows)")
plt.xlabel("Windows")
plt.ylabel("Features")
plt.colorbar()
plt.show()

# =========================
# 2. CLASS SEPARATION (first 2 features only)
# =========================
plt.figure()

for label in np.unique(y):
    idx = y == label
    plt.scatter(X[idx, 0], X[idx, 1], label=f"Class {label}", s=10)

plt.title("Feature Space (Feature 1 vs Feature 2)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid()
plt.show()