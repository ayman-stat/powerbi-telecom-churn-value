"""Data audit: schema profile, missingness, duplicates and a univariate leakage screen."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .style import PALETTE, headline


def profile(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: type, completeness, cardinality, range and an example value."""
    rows = []
    for c in df.columns:
        s = df[c]
        num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        nn = s.dropna()
        rows.append({
            "column": c, "dtype": str(s.dtype), "non_null": int(s.notna().sum()), "missing_pct": round(float(s.isna().mean()) * 100, 2),
            "unique": int(s.nunique(dropna=True)),
            "min": nn.min() if num and len(nn) else (nn.min() if pd.api.types.is_datetime64_any_dtype(s) and len(nn) else None),
            "median": float(nn.median()) if num and len(nn) else None,
            "max": nn.max() if (num or pd.api.types.is_datetime64_any_dtype(s)) and len(nn) else None,
            "top": None if num else (nn.astype(str).value_counts().index[0] if len(nn) else None),
            "zeros_pct": round(float((s == 0).mean()) * 100, 2) if num else None,
        })
    return pd.DataFrame(rows).set_index("column")


def duplicates(df: pd.DataFrame, key=None) -> dict:
    """Exact duplicate rows and duplicate keys."""
    out = {"rows": len(df), "exact_duplicate_rows": int(df.duplicated().sum())}
    if key is not None:
        out["duplicate_keys"] = int(df.duplicated(subset=key).sum())
    return out


def plot_missing(df: pd.DataFrame, top: int = 25, ax=None, title: str | None = None):
    """Horizontal bars of missing share for the columns that have any."""
    import matplotlib.pyplot as plt
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0].head(top)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, max(2.2, 0.32 * max(len(miss), 1) + 1)))
    if miss.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=12, color=PALETTE["teal"], transform=ax.transAxes)
        ax.set_axis_off()
    else:
        ax.barh(miss.index[::-1], miss.values[::-1] * 100, color=PALETTE["amber"])
        for i, v in enumerate(miss.values[::-1] * 100):
            ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)
        ax.set_xlabel("Missing, % of rows")
    headline(ax, title or f"{len(miss)} of {df.shape[1]} columns have missing values")
    return ax


def leakage_screen(df: pd.DataFrame, features, target: str, flag_at: float = 0.85) -> pd.DataFrame:
    """Single-feature AUC for every candidate. A lone feature that almost separates the classes is usually
    information from after the outcome (or the outcome itself), so it is flagged for review before modelling.
    Categoricals are scored by their target mean (in-sample, so this is an upper bound)."""
    y = df[target].astype(int).values
    rows = []
    for f in features:
        s = df[f]
        if pd.api.types.is_numeric_dtype(s):
            x = s.astype(float).fillna(s.astype(float).median()).values
        else:
            x = s.astype(str).map(df.groupby(s.astype(str))[target].mean()).values
        try:
            auc = roc_auc_score(y, x)
        except ValueError:
            continue
        rows.append({"feature": f, "single_feature_auc": round(max(auc, 1 - auc), 4), "direction": "+" if auc >= 0.5 else "-"})
    t = pd.DataFrame(rows).sort_values("single_feature_auc", ascending=False).set_index("feature")
    t["review"] = np.where(t["single_feature_auc"] >= flag_at, "CHECK FOR LEAKAGE", "")
    return t
