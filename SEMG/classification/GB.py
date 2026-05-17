from sklearn.ensemble import GradientBoostingClassifier


def get_gb():

    return GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )