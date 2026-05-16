from sklearn.svm import SVC

def get_svm():
    return SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")