from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def get_qda():

    return QuadraticDiscriminantAnalysis(
        reg_param=0.1
    )