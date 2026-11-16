"""Exploratory analysis: target rates with confidence intervals, distributions, association, WoE/IV and drift (PSI)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .style import PALETTE, headline


def wilson(k, n, z: float = 1.96):
    """Wilson score interval for a proportion (works for small n and rates near 0 or 1)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def binned(s: pd.Series, q: int | None = None, bins=None, labels=None) -> pd.Series:
    """Quantile or fixed bins with readable labels; missing values get their own 'Missing' level."""
    if bins is not None:
        b = pd.cut(s, bins, labels=labels, include_lowest=True)
    else:
        b = pd.qcut(s, q or 10, duplicates="drop")
    b = b.astype(str)
    return b.where(s.notna(), "Missing")


def rate_by(df: pd.DataFrame, col: str, target: str, ax=None, order=None, min_n: int = 30, title: str | None = None,
            subtitle: str | None = None, xlabel: str | None = None, rate_label: str = "Target rate", rotate: int = 0):
    """Target rate per level with a 95% Wilson interval (line) over the share of rows (bars).
    Returns the table so the notebook can quote exact numbers."""
    import matplotlib.pyplot as plt
    g = df.groupby(col, observed=True)[target].agg(["sum", "count"])
    if order is not None:
        g = g.reindex([o for o in order if o in g.index])
    g = g[g["count"] >= min_n]
    lo, hi = wilson(g["sum"], g["count"])
    t = pd.DataFrame({"rows": g["count"], "share": g["count"] / g["count"].sum(), "rate": g["sum"] / g["count"], "ci_low": lo, "ci_high": hi})
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(t))
    ax2 = ax.twinx()
    ax2.bar(x, t["share"] * 100, color=PALETTE["light"], width=0.7, zorder=0)
    ax2.set_ylabel("Share of rows, %", color=PALETTE["grey"])
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.fill_between(x, t["ci_low"] * 100, t["ci_high"] * 100, color=PALETTE["teal"], alpha=0.18, lw=0)
    ax.plot(x, t["rate"] * 100, marker="o", color=PALETTE["teal"], lw=2.2)
    overall = df[target].mean() * 100
    ax.axhline(overall, ls="--", lw=1, color=PALETTE["coral"])
    ax.text(len(t) - 0.5, overall, f" overall {overall:.1f}%", va="bottom", ha="right", fontsize=9, color=PALETTE["coral"])
    ax.set_xticks(x, [str(i) for i in t.index], rotation=rotate, ha="right" if rotate else "center")
    ax.set_ylabel(f"{rate_label}, %")
    ax.set_xlabel(xlabel or col)
    ax.set_ylim(bottom=0)
    headline(ax, title or f"{rate_label} by {col}", subtitle or "Line: rate with 95% Wilson interval. Bars: share of rows.")
    return t


def dist_by_target(df: pd.DataFrame, col: str, target: str, ax=None, clip: float = 0.99, log: bool = False, bins: int = 40,
                   labels=("Retained", "Churned"), title: str | None = None):
    """Overlaid density histograms of a numeric feature for each class, clipped at a quantile so tails don't flatten the view."""
    import matplotlib.pyplot as plt
    s = df[col].astype(float)
    cap = s.quantile(clip)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    edges = np.histogram_bin_edges(np.log1p(s.clip(0, cap).dropna()) if log else s.clip(upper=cap).dropna(), bins=bins)
    for v, lab, colr in ((0, labels[0], PALETTE["navy"]), (1, labels[1], PALETTE["coral"])):
        x = s[df[target] == v].clip(upper=cap).dropna()
        ax.hist(np.log1p(x.clip(lower=0)) if log else x, bins=edges, density=True, alpha=0.55, color=colr, label=f"{lab} (median {x.median():,.1f})")
    ax.set_xlabel(f"log(1 + {col})" if log else col)
    ax.set_ylabel("Density")
    ax.legend()
    headline(ax, title or f"{col} by outcome")
    return ax


def corr_heatmap(df: pd.DataFrame, cols, ax=None, method: str = "spearman", title: str | None = None, annotate: bool = True):
    """Lower-triangle rank-correlation heatmap; returns the matrix."""
    import matplotlib.pyplot as plt
    c = df[cols].corr(method=method)
    mask = np.triu(np.ones_like(c, dtype=bool), k=1)
    if ax is None:
        n = len(cols)
        _, ax = plt.subplots(figsize=(min(1 + 0.6 * n, 14), min(0.8 + 0.5 * n, 12)))
    im = ax.imshow(np.ma.masked_where(mask, c.values), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    ax.grid(False)
    if annotate and len(cols) <= 18:
        for i in range(len(cols)):
            for j in range(i + 1):
                v = c.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color="white" if abs(v) > 0.6 else PALETTE["ink"])
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    headline(ax, title or f"{method.title()} correlation between features")
    return c


def high_correlation_pairs(df: pd.DataFrame, cols, threshold: float = 0.8, method: str = "spearman") -> pd.DataFrame:
    c = df[cols].corr(method=method)
    pairs = [(a, b, c.loc[a, b]) for i, a in enumerate(cols) for b in cols[i + 1:] if abs(c.loc[a, b]) >= threshold]
    return pd.DataFrame(pairs, columns=["feature_a", "feature_b", "rho"]).sort_values("rho", key=abs, ascending=False)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramér's V between two categoricals."""
    t = pd.crosstab(x, y).values
    n = t.sum()
    if n == 0 or min(t.shape) < 2:
        return 0.0
    exp = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / n
    chi2 = ((t - exp) ** 2 / exp).sum()
    r, k = t.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / max(1e-12, min(kc - 1, rc - 1))))


def woe_iv(df: pd.DataFrame, col: str, target: str, q: int = 10, bins=None, eps: float = 0.5) -> tuple[pd.DataFrame, float]:
    """Weight of evidence and information value of one feature (numeric -> quantile bins, categorical -> levels).
    WoE = ln(%goods / %bads) with goods = target 0; eps smooths empty cells."""
    s = df[col]
    b = binned(s.astype(float), q=q, bins=bins) if pd.api.types.is_numeric_dtype(s) and s.nunique() > q else s.astype(str).fillna("Missing")
    g = df.groupby(b, observed=True)[target].agg(["count", "sum"]).rename(columns={"sum": "bads"})
    g["goods"] = g["count"] - g["bads"]
    pg, pb = (g["goods"] + eps) / (g["goods"].sum() + eps), (g["bads"] + eps) / (g["bads"].sum() + eps)
    g["bad_rate"] = g["bads"] / g["count"]
    g["woe"] = np.log(pg / pb)
    g["iv_part"] = (pg - pb) * g["woe"]
    return g, float(g["iv_part"].sum())


def iv_table(df: pd.DataFrame, features, target: str, q: int = 10) -> pd.DataFrame:
    """Information value ranking with the usual credit-scoring strength labels."""
    rows = [(f, woe_iv(df, f, target, q=q)[1]) for f in features]
    t = pd.DataFrame(rows, columns=["feature", "iv"]).sort_values("iv", ascending=False).set_index("feature")
    labels = ["useless", "weak", "medium", "strong", "suspicious (check leakage)"]
    t["strength"] = pd.cut(t["iv"], [-np.inf, 0.02, 0.1, 0.3, 0.5, np.inf], labels=labels)
    return t


def psi(expected, actual, bins: int = 10) -> float:
    """Population stability index of `actual` against `expected`, on the expected sample's quantile bins.
    Rule of thumb: < 0.10 stable, 0.10-0.25 monitor, > 0.25 shifted."""
    e, a = np.asarray(expected, float), np.asarray(actual, float)
    e, a = e[~np.isnan(e)], a[~np.isnan(a)]
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                                   # (near-)constant feature: compare level shares instead
        levels = np.unique(np.concatenate([e, a]))
        pe = np.array([(e == v).mean() for v in levels]) + 1e-6
        pa = np.array([(a == v).mean() for v in levels]) + 1e-6
        return float(np.sum((pa - pe) * np.log(pa / pe)))
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e) + 1e-6
    pa = np.histogram(a, edges)[0] / len(a) + 1e-6
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def stability(reference: pd.DataFrame, current: pd.DataFrame, features, bins: int = 10) -> pd.DataFrame:
    """PSI per feature between two samples (e.g. training snapshot vs the out-of-time snapshot)."""
    rows = []
    for f in features:
        if pd.api.types.is_numeric_dtype(reference[f]):
            v = psi(reference[f], current[f], bins)
        else:
            levels = pd.concat([reference[f], current[f]]).astype(str).unique()
            pe = reference[f].astype(str).value_counts(normalize=True).reindex(levels, fill_value=0).values + 1e-6
            pa = current[f].astype(str).value_counts(normalize=True).reindex(levels, fill_value=0).values + 1e-6
            v = float(np.sum((pa - pe) * np.log(pa / pe)))
        rows.append((f, round(v, 4)))
    t = pd.DataFrame(rows, columns=["feature", "psi"]).sort_values("psi", ascending=False).set_index("feature")
    t["status"] = pd.cut(t["psi"], [-np.inf, 0.1, 0.25, np.inf], labels=["stable", "monitor", "shifted"])
    return t
