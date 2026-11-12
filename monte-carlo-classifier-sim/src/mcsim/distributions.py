"""Random variable generation.

Every distribution is *standardised* (zero mean, unit variance) and then
rescaled to a requested mean / standard deviation. This means all families
have identical first and second moments, so any difference in classifier
behaviour is caused by distribution SHAPE (skewness, tails, bounded support,
discreteness) and not by variance.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Tuple

import numpy as np

# name -> (raw sampler(rng, size), theoretical mean, theoretical std)
_Raw = Tuple[Callable[[np.random.Generator, tuple], np.ndarray], float, float]


def _lognormal_moments(s: float) -> Tuple[float, float]:
    mean = math.exp(s * s / 2)
    var = (math.exp(s * s) - 1) * math.exp(s * s)
    return mean, math.sqrt(var)


_LN_MEAN, _LN_STD = _lognormal_moments(0.5)
_BETA_A, _BETA_B = 2.0, 5.0
_BETA_MEAN = _BETA_A / (_BETA_A + _BETA_B)
_BETA_STD = math.sqrt(
    _BETA_A * _BETA_B / ((_BETA_A + _BETA_B) ** 2 * (_BETA_A + _BETA_B + 1))
)

DISTRIBUTIONS: Dict[str, _Raw] = {
    "normal":      (lambda r, s: r.normal(0.0, 1.0, s), 0.0, 1.0),
    "uniform":     (lambda r, s: r.uniform(-math.sqrt(3), math.sqrt(3), s), 0.0, 1.0),
    "laplace":     (lambda r, s: r.laplace(0.0, 1 / math.sqrt(2), s), 0.0, 1.0),
    "student_t5":  (lambda r, s: r.standard_t(5, s), 0.0, math.sqrt(5 / 3)),
    "exponential": (lambda r, s: r.exponential(1.0, s), 1.0, 1.0),
    "gamma":       (lambda r, s: r.gamma(2.0, 1.0, s), 2.0, math.sqrt(2.0)),
    "lognormal":   (lambda r, s: r.lognormal(0.0, 0.5, s), _LN_MEAN, _LN_STD),
    "beta":        (lambda r, s: r.beta(_BETA_A, _BETA_B, s), _BETA_MEAN, _BETA_STD),
    "poisson":     (lambda r, s: r.poisson(4.0, s).astype(float), 4.0, 2.0),
}


def draw(name: str, rng: np.random.Generator, size, mean: float = 0.0,
         std: float = 1.0) -> np.ndarray:
    """Draw samples from distribution `name` with the given mean and std."""
    if name not in DISTRIBUTIONS:
        raise KeyError(f"Unknown distribution '{name}'. Options: {list(DISTRIBUTIONS)}")
    sampler, m, s = DISTRIBUTIONS[name]
    shape = (size,) if np.isscalar(size) else tuple(size)
    return mean + std * (sampler(rng, shape) - m) / s


def make_dataset(dist: str, rng: np.random.Generator, n: int, n_features: int = 2,
                 separation: float = 1.5, std: float = 1.0):
    """Two-class dataset with balanced classes.

    Class 0 features ~ dist(mean=0, std), class 1 features ~ dist(mean=separation, std).
    Features are independent. Returns X (n, n_features) and y (n,).
    """
    n1 = n // 2
    n0 = n - n1
    X0 = draw(dist, rng, (n0, n_features), 0.0, std)
    X1 = draw(dist, rng, (n1, n_features), separation, std)
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(n0, dtype=int), np.ones(n1, dtype=int)]
    idx = rng.permutation(n)
    return X[idx], y[idx]


def bayes_accuracy_gaussian(separation: float, n_features: int = 2,
                            std: float = 1.0) -> float:
    """Optimal accuracy for equal-prior, equal-covariance Gaussian classes.

    accuracy = Phi(delta / 2), delta = separation * sqrt(d) / std
    """
    delta = separation * math.sqrt(n_features) / std
    return 0.5 * (1 + math.erf(delta / 2 / math.sqrt(2)))
