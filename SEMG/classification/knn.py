from sklearn.neighbors import KNeighborsClassifier


def get_knn():

    return KNeighborsClassifier(
        n_neighbors=1,          # k=1 best for feature-based EMG
        weights="distance",     # closer samples matter more
        metric="minkowski",     # standard, stable for EMG feature spaces
        p=2                     # Euclidean distance
    )