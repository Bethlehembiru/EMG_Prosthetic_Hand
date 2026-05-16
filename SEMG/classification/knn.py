from sklearn.neighbors import KNeighborsClassifier

def get_knn():
    return KNeighborsClassifier(n_neighbors=5)