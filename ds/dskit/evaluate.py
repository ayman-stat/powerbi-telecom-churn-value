"""Model evaluation: discrimination, calibration, ranking value, thresholds, segments - each with uncertainty."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
)

from .style import CYCLE, PALETTE


# ------------------------------------------------------------------ classification
def ks_stat(y, p) -> float:
    """Kolmogorov-Smirnov separation: max gap between the cumulative score distributions of the two classes."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def binary_metrics(y, p) -> dict:
    y, p = np.asarray(y).astype(int), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    auc = roc_auc_score(y, p)
    return {"auc": auc, "gini": 2 * auc - 1, "pr_auc": average_precision_score(y, p), "ks": ks_stat(y, p),
            "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p), "base_rate": float(y.mean()), "mean_pred": float(p.mean())}


def bootstrap_ci(y, p, fn=roc_auc_score, n: int = 500, seed: int = 42, alpha: float = 0.05, stratify: bool = True):
    """Percentile bootstrap interval for any metric fn(y, p). Stratified so rare classes appear in every resample."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n):
        ix = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) if stratify else rng.integers(0, len(y), len(y))
        vals.append(fn(y[ix], p[ix]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(fn(y, p)), float(lo), float(hi)


def metrics_table(y, preds: dict, n_boot: int = 300, seed: int = 42) -> pd.DataFrame:
    """One row per model: AUC and PR AUC with 95% bootstrap intervals, plus KS, Brier and log loss."""
    rows = []
    for name, p in preds.items():
        m = binary_metrics(y, p)
        _, alo, ahi = bootstrap_ci(y, p, roc_auc_score, n_boot, seed)
        _, plo, phi = bootstrap_ci(y, p, average_precision_score, n_boot, seed)
        rows.append({"model": name, "AUC": m["auc"], "AUC 95% CI": f"{alo:.3f} - {ahi:.3f}", "PR AUC": m["pr_auc"],
                     "PR AUC 95% CI": f"{plo:.3f} - {phi:.3f}", "KS": m["ks"], "Brier": m["brier"], "Log loss": m["log_loss"]})
    return pd.DataFrame(rows).set_index("model").round(4)


def paired_auc_test(y, p_a, p_b, n: int = 1000, seed: int = 42) -> dict:
    """Paired bootstrap of AUC(a) - AUC(b) on the same rows: is the challenger really better?"""
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    d = []
    for _ in range(n):
        ix = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        d.append(roc_auc_score(y[ix], p_a[ix]) - roc_auc_score(y[ix], p_b[ix]))
    d = np.array(d)
    return {"delta_auc": float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b)), "ci_low": float(np.percentile(d, 2.5)),
            "ci_high": float(np.percentile(d, 97.5)), "p_value_one_sided": float((d <= 0).mean())}


def calibration_table(y, p, n: int = 10) -> pd.DataFrame:
    """Observed vs predicted rate per predicted-score decile."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n, labels=False)
    t = df.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), rows=("y", "size"))
    return t


def gains_table(y, p, n: int = 10, value=None) -> pd.DataFrame:
    """Decile table, highest score first: rate, lift and cumulative capture of events (and of value, if given)."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    if value is not None:
        df["v"] = np.asarray(value, float)
    df["decile"] = n - pd.qcut(df["p"].rank(method="first"), n, labels=False)
    agg = {"rows": ("y", "size"), "events": ("y", "sum"), "mean_score": ("p", "mean")}
    if value is not None:
        agg["value"] = ("v", "sum")
    t = df.groupby("decile").agg(**agg)
    t["rate"] = t["events"] / t["rows"]
    t["lift"] = t["rate"] / df["y"].mean()
    t["cum_capture"] = t["events"].cumsum() / t["events"].sum()
    if value is not None:
        t["cum_value_capture"] = t["value"].cumsum() / t["value"].sum()
    return t


def plot_binary_panel(y, preds: dict, suptitle: str | None = None, n_cal: int = 10):
    """2x2 panel: ROC, precision-recall, calibration (reliability) and cumulative gains for one or more models."""
    import matplotlib.pyplot as plt
    y = np.asarray(y).astype(int)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
    (a1, a2), (a3, a4) = axes
    for (name, p), c in zip(preds.items(), CYCLE):
        p = np.asarray(p)
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, color=c, label=f"{name}  AUC {roc_auc_score(y, p):.3f}")
        pr, rc, _ = precision_recall_curve(y, p)
        a2.plot(rc, pr, color=c, label=f"{name}  AP {average_precision_score(y, p):.3f}")
        cal = calibration_table(y, p, n_cal)
        a3.plot(cal["predicted"], cal["observed"], marker="o", color=c, label=name)
        order = np.argsort(-p)
        cum = np.concatenate([[0], np.cumsum(y[order]) / y.sum()])
        a4.plot(np.linspace(0, 1, len(cum)), cum, color=c, label=name)
    a1.plot([0, 1], [0, 1], ls="--", color=PALETTE["grey"], lw=1)
    a2.axhline(y.mean(), ls="--", color=PALETTE["grey"], lw=1, label=f"base rate {y.mean():.3f}")
    lim = min(1.0, max(a3.get_xlim()[1], a3.get_ylim()[1]))
    a3.plot([0, lim], [0, lim], ls="--", color=PALETTE["grey"], lw=1, label="perfect calibration")
    a4.plot([0, 1], [0, 1], ls="--", color=PALETTE["grey"], lw=1, label="random")
    for ax, t, xl, yl in ((a1, "ROC curve", "False positive rate", "True positive rate"),
                          (a2, "Precision-recall", "Recall", "Precision"),
                          (a3, "Calibration (deciles)", "Mean predicted probability", "Observed rate"),
                          (a4, "Cumulative gains", "Share of population targeted (highest score first)", "Share of events captured")):
        ax.set_title(t, loc="left", fontsize=12)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.legend(loc="lower right" if ax is not a2 else "upper right", fontsize=8.5)
    if suptitle:
        fig.suptitle(suptitle, x=0.06, ha="left", fontsize=14, fontweight="bold", color=PALETTE["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def threshold_table(y, p, shares=None, benefit_tp: float = 1.0, cost_fp: float = 0.0, cost_fn: float = 0.0, value=None) -> pd.DataFrame:
    """Act on the top-k% by score. For each k: alerts, precision, recall and net value
    = benefit_tp * TP (or the captured `value`) - cost_fp * FP - cost_fn * FN."""
    y, p = np.asarray(y).astype(int), np.asarray(p)
    shares = shares if shares is not None else [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
    order = np.argsort(-p)
    rows = []
    for s in shares:
        k = max(1, int(round(s * len(y))))
        sel = np.zeros(len(y), bool)
        sel[order[:k]] = True
        tp, fp = int((sel & (y == 1)).sum()), int((sel & (y == 0)).sum())
        fn = int((~sel & (y == 1)).sum())
        gain = float(np.asarray(value)[sel & (y == 1)].sum()) if value is not None else benefit_tp * tp
        rows.append({"target_share": s, "threshold": float(p[order[k - 1]]), "alerts": k, "tp": tp, "fp": fp,
                     "precision": tp / k, "recall": tp / max(1, y.sum()), "net_value": gain - cost_fp * fp - cost_fn * fn})
    return pd.DataFrame(rows)


def segment_performance(df: pd.DataFrame, segment: str, y: str, p: str, min_n: int = 200) -> pd.DataFrame:
    """Per-segment discrimination and calibration: does the model work equally well for every group?"""
    rows = []
    for seg, g in df.groupby(segment, observed=True):
        if len(g) < min_n or g[y].nunique() < 2:
            continue
        rows.append({segment: seg, "rows": len(g), "base_rate": g[y].mean(), "mean_pred": g[p].mean(),
                     "calibration_ratio": g[p].mean() / max(1e-9, g[y].mean()), "auc": roc_auc_score(g[y], g[p])})
    return pd.DataFrame(rows).set_index(segment).round(4)


# ------------------------------------------------------------------ regression / forecasting
def regression_metrics(y, yhat) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    nz = y != 0
    return {"mae": mean_absolute_error(y, yhat), "rmse": float(np.sqrt(mean_squared_error(y, yhat))),
            "mape": float(np.mean(np.abs((y[nz] - yhat[nz]) / y[nz]))), "wape": float(np.abs(y - yhat).sum() / np.abs(y).sum()),
            "rmspe": float(np.sqrt(np.mean(((y[nz] - yhat[nz]) / y[nz]) ** 2))), "bias_pct": float((yhat.sum() - y.sum()) / y.sum()),
            "r2": r2_score(y, yhat)}


def plot_residuals(y, yhat, title: str = "Residual diagnostics", sample: int = 20000, seed: int = 42):
    """Predicted vs actual, residual distribution and residual vs prediction (heteroscedasticity check)."""
    import matplotlib.pyplot as plt
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ix = np.random.default_rng(seed).choice(len(y), min(sample, len(y)), replace=False)
    r = y - yhat
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.4))
    a1.scatter(yhat[ix], y[ix], s=4, alpha=0.25, color=PALETTE["navy"])
    lim = [min(y.min(), yhat.min()), max(y.max(), yhat.max())]
    a1.plot(lim, lim, ls="--", color=PALETTE["coral"], lw=1)
    a1.set(xlabel="Predicted", ylabel="Actual", title="Predicted vs actual")
    a2.hist(r, bins=60, color=PALETTE["teal"], alpha=0.8)
    a2.axvline(0, color=PALETTE["coral"], lw=1)
    a2.set(xlabel="Residual (actual - predicted)", ylabel="Rows", title=f"Residuals: mean {r.mean():,.1f}")
    a3.scatter(yhat[ix], r[ix], s=4, alpha=0.25, color=PALETTE["purple"])
    a3.axhline(0, color=PALETTE["coral"], lw=1)
    a3.set(xlabel="Predicted", ylabel="Residual", title="Residual vs predicted")
    fig.suptitle(title, x=0.04, ha="left", fontsize=14, fontweight="bold", color=PALETTE["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig

