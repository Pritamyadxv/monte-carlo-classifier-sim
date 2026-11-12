"""Monte Carlo simulation of random variables and classifier behaviour."""
from .distributions import DISTRIBUTIONS, draw, make_dataset, bayes_accuracy_gaussian
from .classifiers import CLASSIFIERS, get_classifier
from .simulation import monte_carlo, summarize, sweep

__all__ = [
    "DISTRIBUTIONS", "draw", "make_dataset", "bayes_accuracy_gaussian",
    "CLASSIFIERS", "get_classifier", "monte_carlo", "summarize", "sweep",
]
__version__ = "1.0.0"
