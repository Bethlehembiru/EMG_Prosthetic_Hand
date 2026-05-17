from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def get_lda():

    return LinearDiscriminantAnalysis(
        solver="lsqr",
        shrinkage="auto"
    )