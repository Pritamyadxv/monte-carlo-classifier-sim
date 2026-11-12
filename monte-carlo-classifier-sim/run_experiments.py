#!/usr/bin/env python3
"""Run the full Monte Carlo study and save CSVs + figures to results/."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from mcsim import monte_carlo, summarize, sweep          # noqa: E402
from mcsim import plots                                  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=int, default=200, help="Monte Carlo trials")
    p.add_argument("--sweep-trials", type=int, default=50, help="trials per sweep point")
    p.add_argument("--n-train", type=int, default=100)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--n-features", type=int, default=2)
    p.add_argument("--separation", type=float, default=1.5)
    p.add_argument("--std", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results")
    p.add_argument("--no-sweeps", action="store_true")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    common = dict(n_train=a.n_train, n_test=a.n_test, n_features=a.n_features,
                  separation=a.separation, std=a.std, seed=a.seed)

    print(f"[1/4] Main experiment: {a.trials} trials ...")
    df = monte_carlo(n_trials=a.trials, **common)
    df.to_csv(f"{a.out}/trials.csv", index=False)
    summ = summarize(df)
    summ.to_csv(f"{a.out}/summary_accuracy.csv", index=False)
    print(summ.pivot(index="dist", columns="clf", values="mean").round(3).to_string())

    print("[2/4] Figures for main experiment ...")
    plots.plot_distributions(f"{a.out}/distributions.png", seed=a.seed)
    plots.plot_heatmap(summ, f"{a.out}/heatmap_accuracy.png")
    plots.plot_boxplot(df, f"{a.out}/boxplot_accuracy.png")
    plots.plot_convergence(df, f"{a.out}/convergence.png")

    if not a.no_sweeps:
        print("[3/4] Sweep: class separation ...")
        sw = sweep("separation", [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], n_trials=a.sweep_trials,
                   **{k: v for k, v in common.items() if k != "separation"})
        sw.to_csv(f"{a.out}/sweep_separation.csv", index=False)
        plots.plot_sweep(sw, "separation", f"{a.out}/sweep_separation.png",
                         n_features=a.n_features, std=a.std, show_bayes=True)

        print("[4/4] Sweep: training-set size ...")
        sw2 = sweep("n_train", [10, 20, 50, 100, 200, 500], n_trials=a.sweep_trials,
                    **{k: v for k, v in common.items() if k != "n_train"})
        sw2.to_csv(f"{a.out}/sweep_n_train.csv", index=False)
        plots.plot_sweep(sw2, "n_train", f"{a.out}/sweep_n_train.png")

    print(f"Done. Results in ./{a.out}/")


if __name__ == "__main__":
    main()
