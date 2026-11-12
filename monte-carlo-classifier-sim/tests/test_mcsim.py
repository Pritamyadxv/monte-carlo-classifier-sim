import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mcsim import DISTRIBUTIONS, draw, make_dataset, monte_carlo, summarize
from mcsim import bayes_accuracy_gaussian


def test_moments_match_for_all_distributions():
    rng = np.random.default_rng(0)
    for name in DISTRIBUTIONS:
        x = draw(name, rng, 400_000, mean=3.0, std=2.0)
        assert abs(x.mean() - 3.0) < 0.05, name
        assert abs(x.std() - 2.0) < 0.1, name


def test_dataset_shape_and_balance():
    rng = np.random.default_rng(1)
    X, y = make_dataset("normal", rng, 101, n_features=3)
    assert X.shape == (101, 3) and y.shape == (101,)
    assert abs((y == 1).sum() - (y == 0).sum()) <= 1


def test_reproducible():
    a = monte_carlo(["normal"], ["lda"], n_trials=5, seed=7)
    b = monte_carlo(["normal"], ["lda"], n_trials=5, seed=7)
    assert a.equals(b)


def test_lda_close_to_bayes_on_gaussian_data():
    df = monte_carlo(["normal"], ["lda"], n_trials=100, n_train=500, n_test=2000,
                     separation=1.5, seed=3)
    acc = summarize(df)["mean"].iloc[0]
    assert abs(acc - bayes_accuracy_gaussian(1.5)) < 0.01


def test_zero_separation_is_chance():
    df = monte_carlo(["normal"], ["logistic"], n_trials=100, separation=0.0, seed=5)
    assert abs(df["accuracy"].mean() - 0.5) < 0.02


if __name__ == "__main__":          # run without pytest
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("PASS", k)
