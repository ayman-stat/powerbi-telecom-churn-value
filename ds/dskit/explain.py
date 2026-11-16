"""Explainability: SHAP (global, dependence, local), permutation importance and partial dependence."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .style import PALETTE, headline


def shap_explanation(model, X: pd.DataFrame, sample: int = 3000, seed: int = 42):
    """TreeExplainer SHAP values on a reproducible sample (log-odds for classifiers). Returns a shap.Explanation."""
    import shap
    Xs = X.sample(min(sample, len(X)), random_state=seed) if len(X) > sample else X
    ex = shap.TreeExplainer(model)
    e = ex(Xs)
    if e.values.ndim == 3:                         # some sklearn classifiers return one slice per class
        e = e[:, :, 1]
    return e


def plot_shap_beeswarm(e, max_display: int = 15, title: str = "What drives the prediction", subtitle: str | None = None):
    import matplotlib.pyplot as plt
    import shap
    plt.figure()                                   # shap draws on the current figure: start a fresh one
    shap.plots.beeswarm(e, max_display=max_display, show=False, plot_size=(10, 0.42 * max_display + 1.5))
    fig, ax = plt.gcf(), plt.gca()
    headline(ax, title, subtitle or "Each dot is one row; colour is the feature value; position is its push on the prediction (SHAP).")
    return fig


def shap_importance(e) -> pd.DataFrame:
    """Mean |SHAP| per feature, i.e. the average size of each feature's contribution."""
    v = np.abs(e.values).mean(0)
    t = pd.DataFrame({"feature": e.feature_names, "mean_abs_shap": v}).sort_values("mean_abs_shap", ascending=False)
    t["share"] = t["mean_abs_shap"] / t["mean_abs_shap"].sum()
    return t.set_index("feature")


def plot_shap_dependence(e, features, ncols: int = 3, title: str = "How each driver moves the prediction"):
    """SHAP dependence scatter for several features (non-linear shapes and thresholds)."""
    import matplotlib.pyplot as plt
    names = list(e.feature_names)
    n = len(features)
    rows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(rows, ncols, figsize=(4.6 * ncols, 3.6 * rows), squeeze=False)
    for ax, f in zip(axes.flat, features):
        j = names.index(f)
        x = e.data[:, j].astype(float)
        ax.scatter(x, e.values[:, j], s=5, alpha=0.35, color=PALETTE["navy"])
        ax.axhline(0, color=PALETTE["grey"], lw=0.8)
        ax.set_title(f, loc="left", fontsize=11)
        ax.set_ylabel("SHAP")
    for ax in list(axes.flat)[n:]:
        ax.set_axis_off()
    fig.suptitle(title, x=0.04, ha="left", fontsize=14, fontweight="bold", color=PALETTE["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_shap_waterfall(e, i: int, title: str, max_display: int = 10):
    """Why one customer/account got their score."""
    import matplotlib.pyplot as plt
    import shap
    plt.figure()
    shap.plots.waterfall(e[i], max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 0.45 * max_display + 1.8)
    fig.suptitle(title, x=0.02, ha="left", fontsize=13, fontweight="bold", color=PALETTE["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def permutation_table(model, X, y, scoring: str = "roc_auc", n_repeats: int = 5, seed: int = 42, labels: dict | None = None) -> pd.DataFrame:
    """Drop in score when each feature is shuffled on held-out data (model-agnostic, measured on the metric that matters)."""
    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, X, y, scoring=scoring, n_repeats=n_repeats, random_state=seed, n_jobs=1)
    t = pd.DataFrame({"feature": X.columns, "drop_mean": r.importances_mean, "drop_std": r.importances_std})
    if labels:
        t["label"] = t["feature"].map(labels).fillna(t["feature"])
    return t.sort_values("drop_mean", ascending=False).set_index("feature")


def plot_importance(t: pd.DataFrame, col: str = "drop_mean", err: str | None = "drop_std", top: int = 15, ax=None,
                    title: str = "Feature importance", xlabel: str = "Drop in AUC when shuffled"):
    import matplotlib.pyplot as plt
    t = t.head(top)[::-1]
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 0.38 * len(t) + 1.4))
    lab = t["label"] if "label" in t else t.index
    ax.barh(lab, t[col], xerr=t[err] if err and err in t else None, color=PALETTE["teal"], ecolor=PALETTE["grey"])
    ax.set_xlabel(xlabel)
    headline(ax, title)
    return ax


def plot_pdp(model, X: pd.DataFrame, features, title: str = "Partial dependence", ncols: int = 3, sample: int = 2000, seed: int = 42):
    """Partial dependence with ICE lines (sampled) for the main drivers."""
    import matplotlib.pyplot as plt
    from sklearn.inspection import PartialDependenceDisplay
    Xs = X.sample(min(sample, len(X)), random_state=seed)
    rows = int(np.ceil(len(features) / ncols))
    fig, ax = plt.subplots(rows, ncols, figsize=(4.6 * ncols, 3.5 * rows))
    PartialDependenceDisplay.from_estimator(model, Xs, features, kind="both", subsample=150, random_state=seed, ax=ax,
                                            ice_lines_kw={"color": PALETTE["light"], "alpha": 0.5, "lw": 0.6},
                                            pd_line_kw={"color": PALETTE["coral"], "lw": 2.4})
    fig.suptitle(title, x=0.04, ha="left", fontsize=14, fontweight="bold", color=PALETTE["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
