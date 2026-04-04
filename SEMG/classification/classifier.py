from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

class EMGClassifier:
    def __init__(self):
        self.models = {
            "lda": LinearDiscriminantAnalysis(),
            "svm": SVC(kernel='rbf', gamma='scale'),
            "knn": KNeighborsClassifier(n_neighbors=5)
        }

    def train(self, X, y):
        for name, model in self.models.items():
            model.fit(X, y)

    def get_models(self):
        return self.models