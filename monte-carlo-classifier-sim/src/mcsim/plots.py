"""Matplotlib figures (headless-safe)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .distributions import DISTRIBUTIONS, bayes_accuracy_gaussian, draw


def plot_distributions(path, n=20000, seed=0):
    """Histogram of every distribution after standardisation (same mean/std)."""
    rng = np.random.default_rng(seed)
    names = list(DISTRIBUTIONS)
    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3 * rows))
    for ax, name in zip(axes.ravel(), names):
        x = draw(name, rng, n)
        ax.hist(x, bins=60, color="#3b6fb6", alpha=0.85, density=True)
        ax.set_title(f"{name}  (mean={x.mean():.2f}, std={x.std():.2f})", fontsize=9)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle("Random variables with identical mean (0) and std (1)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_heatmap(summary, path, metric_label="Mean accuracy"):
    pv = summary.pivot(index="dist", columns="clf", values="mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pv.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pv.columns)), pv.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pv.index)), pv.index)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            ax.text(j, i, f"{pv.values[i, j]:.3f}", ha="center", va="center",
                    color="white", fontsize=8)
    fig.colorbar(im, label=metric_label)
    ax.set_title(f"{metric_label}: distribution x classifier")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_boxplot(df, path, metric="accuracy"):
    clfs = sorted(df["clf"].unique())
    dists = sorted(df["dist"].unique())
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.8 / len(clfs)
    cmap = plt.get_cmap("tab10")
    for k, c in enumerate(clfs):
        data = [df[(df.dist == d) & (df.clf == c)][metric] for d in dists]
        pos = np.arange(len(dists)) + k * width - 0.4 + width / 2
        bp = ax.boxplot(data, positions=pos, widths=width * 0.9, patch_artist=True,
                        showfliers=False)
        for b in bp["boxes"]:
            b.set_facecolor(cmap(k))
        ax.plot([], [], color=cmap(k), lw=6, label=c)
    ax.set_xticks(range(len(dists)), dists, rotation=20)
    ax.set_ylabel(metric)
    ax.set_title(f"Monte Carlo distribution of {metric}")
    ax.legend(ncol=len(clfs), fontsize=8, loc="lower center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sweep(df, param, path, metric="accuracy", n_features=2, std=1.0,
               show_bayes=False):
    """One panel per classifier, one line per distribution."""
    clfs = sorted(df["clf"].unique())
    cols = 3
    rows = int(np.ceil(len(clfs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4 * rows), sharey=True)
    flat = np.atleast_1d(axes).ravel()
    m = df.groupby(["clf", "dist", param])[metric].mean().reset_index()
    for ax, c in zip(flat, clfs):
        for d, sub in m[m.clf == c].groupby("dist"):
            ax.plot(sub[param], sub[metric], marker="o", ms=3, label=d)
        if show_bayes and param == "separation":
            xs = np.array(sorted(df[param].unique()))
            ax.plot(xs, [bayes_accuracy_gaussian(x, n_features, std) for x in xs],
                    "k--", lw=1.5, label="Bayes optimum (Gaussian)")
        ax.set_title(c)
        ax.set_xlabel(param)
        ax.grid(alpha=0.3)
    flat[0].set_ylabel(metric)
    flat[0].legend(fontsize=6)
    for ax in flat[len(clfs):]:
        ax.axis("off")
    fig.suptitle(f"{metric} vs {param}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_convergence(df, path, clf="lda", metric="accuracy"):
    """Running Monte Carlo mean as trials accumulate."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for d, sub in df[df.clf == clf].groupby("dist"):
        x = sub.sort_values("trial")[metric].values
        n = np.arange(1, len(x) + 1)
        ax.plot(n, np.cumsum(x) / n, label=d, lw=1.2)
    ax.set_xlabel("number of Monte Carlo trials")
    ax.set_ylabel(f"running mean {metric}")
    ax.set_title(f"Convergence of the Monte Carlo estimate ({clf})")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
