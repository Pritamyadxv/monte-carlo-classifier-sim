"""Monte Carlo engine.

For each (distribution, trial) a fresh train/test dataset is generated and
EVERY classifier is fit on the same data (paired comparison). Trial t uses the
same random seed stream for every distribution (common random numbers), which
reduces variance when comparing distributions.
"""
from __future__ import annotations

import warnings
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .classifiers import CLASSIFIERS, get_classifier
from .distributions import DISTRIBUTIONS, make_dataset


def monte_carlo(dists: Sequence[str] | None = None,
                clfs: Sequence[str] | None = None,
                n_trials: int = 200, n_train: int = 100, n_test: int = 1000,
                n_features: int = 2, separation: float = 1.5, std: float = 1.0,
                seed: int = 0) -> pd.DataFrame:
    """Run the simulation. Returns one row per (dist, clf, trial)."""
    dists = list(dists or DISTRIBUTIONS)
    clfs = list(clfs or CLASSIFIERS)
    seeds = np.random.SeedSequence(seed).spawn(n_trials)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for d in dists:
            for t in range(n_trials):
                rng = np.random.default_rng(seeds[t])
                Xtr, ytr = make_dataset(d, rng, n_train, n_features, separation, std)
                Xte, yte = make_dataset(d, rng, n_test, n_features, separation, std)
                for c in clfs:
                    model = get_classifier(c).fit(Xtr, ytr)
                    pred = model.predict(Xte)
                    score = model.predict_proba(Xte)[:, 1]
                    rows.append({
                        "dist": d, "clf": c, "trial": t,
                        "accuracy": accuracy_score(yte, pred),
                        "f1": f1_score(yte, pred),
                        "auc": roc_auc_score(yte, score),
                    })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, metric: str = "accuracy") -> pd.DataFrame:
    """Mean, std, and 95% normal-approximation CI of a metric per (dist, clf)."""
    g = df.groupby(["dist", "clf"])[metric].agg(["mean", "std", "count"]).reset_index()
    g["ci95"] = 1.96 * g["std"] / np.sqrt(g["count"])
    return g


def sweep(param: str, values: Iterable, **kwargs) -> pd.DataFrame:
    """Repeat `monte_carlo` while varying one parameter (e.g. 'separation',
    'n_train', 'std', 'n_features'). Adds a column named after `param`."""
    frames = []
    for v in values:
        df = monte_carlo(**{**kwargs, param: v})
        df[param] = v
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
