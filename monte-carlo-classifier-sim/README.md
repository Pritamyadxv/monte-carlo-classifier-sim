# Monte Carlo Random Variable and Classifier Simulation

Generate random variables from multiple probability distributions and study
their effect on classifier behaviour using Monte Carlo simulation.

## Idea

1. **Random variables** – 9 distributions (normal, uniform, Laplace, Student-t,
   exponential, gamma, log-normal, beta, Poisson), each *standardised* to the
   same mean and standard deviation. Only the **shape** (skewness, heavy tails,
   bounded support, discreteness) differs.
2. **Data** – two balanced classes; class 0 features ~ `dist(mean=0)`,
   class 1 features ~ `dist(mean=separation)`, independent features.
3. **Classifiers** – LDA, QDA, Logistic Regression, Gaussian Naive Bayes,
   k-NN (k=5), Decision Tree (depth 5).
4. **Monte Carlo** – repeat N times: draw a fresh train/test set, fit every
   classifier on the same data (paired comparison), record accuracy / F1 / AUC.
   Report mean, std and 95 % confidence intervals.
5. **Sweeps** – vary class separation or training-set size to see how
   sensitivity to distribution shape changes.

Reproducible: seeds come from `numpy.random.SeedSequence`; trial *t* uses the
same random stream for every distribution (common random numbers).

## Project structure

```
.
├── run_experiments.py        # CLI: runs everything, writes results/
├── requirements.txt
├── src/mcsim/
│   ├── distributions.py      # standardised random variables + dataset generator
│   ├── classifiers.py        # classifier registry
│   ├── simulation.py         # monte_carlo(), summarize(), sweep()
│   └── plots.py              # figures
├── tests/test_mcsim.py       # unit tests
└── results/                  # generated CSVs and PNGs
```

## Quick start

```bash
git clone https://github.com/<your-username>/monte-carlo-classifier-sim.git
cd monte-carlo-classifier-sim
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_experiments.py                 # full study (default 200 trials)
python run_experiments.py --trials 1000 --separation 2.0 --n-features 5
python run_experiments.py --no-sweeps     # only the main experiment
pytest -q                                 # run tests
```

Options: `--trials --sweep-trials --n-train --n-test --n-features --separation --std --seed --out`.

## Use as a library

```python
import sys; sys.path.insert(0, "src")
from mcsim import monte_carlo, summarize

df = monte_carlo(dists=["normal", "exponential"], clfs=["lda", "knn"],
                 n_trials=500, separation=1.5)
print(summarize(df))
```

## Outputs (`results/`)

| File | Content |
|------|---------|
| `distributions.png` | Histograms of all random variables (same mean/std) |
| `heatmap_accuracy.png` | Mean accuracy, distribution x classifier |
| `boxplot_accuracy.png` | Spread of accuracy across Monte Carlo trials |
| `convergence.png` | Running Monte Carlo mean vs number of trials |
| `sweep_separation.png` | Accuracy vs class separation (with Gaussian Bayes optimum) |
| `sweep_n_train.png` | Accuracy vs training-set size |
| `trials.csv`, `summary_accuracy.csv`, `sweep_*.csv` | Raw and aggregated numbers |

## Sample findings (200 trials, separation 1.5, 2 features, 100 training points)

* On **normal** data LDA/logistic reach ~0.852, matching the theoretical
  Bayes optimum `Phi(Delta/2) ~ 0.855` — a useful sanity check of the simulator.
* **Non-Gaussian shapes change behaviour even at equal mean and variance.**
  Skewed, lower-bounded distributions (exponential, gamma, log-normal) give
  higher accuracy for the non-linear learners (decision tree, k-NN) because the
  hard lower bound of the shifted class creates an easy boundary; linear
  models (LDA, logistic) can't exploit it.
* Decision trees overfit on small samples and are the most distribution-sensitive.
* With `separation = 0` every classifier collapses to ~0.5 (chance), as expected.

Exact numbers vary with the seed and settings; rerun to reproduce.

## Extending

* Add a distribution: add an entry to `DISTRIBUTIONS` with its theoretical mean and std.
* Add a classifier: add a factory to `CLASSIFIERS`.
* Other sweeps: `sweep("std", [...])`, `sweep("n_features", [...])`.

## License

MIT
