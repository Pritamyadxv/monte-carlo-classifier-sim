"""Classifier registry (scikit-learn)."""
from __future__ import annotations

from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

CLASSIFIERS = {
    "lda":           lambda: LinearDiscriminantAnalysis(),
    "qda":           lambda: QuadraticDiscriminantAnalysis(reg_param=1e-3),
    "logistic":      lambda: LogisticRegression(max_iter=500),
    "gaussian_nb":   lambda: GaussianNB(),
    "knn":           lambda: KNeighborsClassifier(n_neighbors=5),
    "decision_tree": lambda: DecisionTreeClassifier(max_depth=5, random_state=0),
}


def get_classifier(name: str):
    if name not in CLASSIFIERS:
        raise KeyError(f"Unknown classifier '{name}'. Options: {list(CLASSIFIERS)}")
    return CLASSIFIERS[name]()
