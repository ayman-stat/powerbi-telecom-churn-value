"""Model selection: cross-validated comparison, time-ordered splits and budgeted Optuna tuning."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import get_scorer

from .style import PALETTE, headline


def cv_compare(models: dict, X, y, cv, scoring=("roc_auc", "average_precision", "neg_brier_score"), groups=None) -> pd.DataFrame:
    """Fit every candidate on the same folds; mean and spread per metric, plus fit time. Folds come from `cv`
    (use a time-ordered splitter for anything that will score the future)."""
    splits = list(cv.split(X, y, groups))
    rows = []
    for name, m in models.items():
        scores = {s: [] for s in scoring}
        t0 = time.perf_counter()
        for tr, te in splits:
            est = clone(m).fit(X.iloc[tr], np.asarray(y)[tr])
            for s in scoring:
                scores[s].append(get_scorer(s)(est, X.iloc[te], np.asarray(y)[te]))
        row = {"model": name, "fit_seconds": round((time.perf_counter() - t0) / len(splits), 2)}
        for s in scoring:
            v = np.array(scores[s]) * (-1 if s.startswith("neg_") else 1)
            row[f"{s.replace('neg_', '')}_mean"], row[f"{s.replace('neg_', '')}_std"] = v.mean(), v.std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("model").round(4)


def plot_cv_compare(t: pd.DataFrame, metric: str = "roc_auc", title: str | None = None, ax=None):
    import matplotlib.pyplot as plt
    t = t.sort_values(f"{metric}_mean")
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 0.55 * len(t) + 1.5))
    ax.barh(t.index, t[f"{metric}_mean"], xerr=t[f"{metric}_std"], color=PALETTE["navy"], ecolor=PALETTE["grey"])
    lo = (t[f"{metric}_mean"] - 3 * t[f"{metric}_std"]).min()
    ax.set_xlim(max(0, lo - 0.02), min(1, t[f"{metric}_mean"].max() + 0.03) if t[f"{metric}_mean"].max() <= 1 else None)
    for i, v in enumerate(t[f"{metric}_mean"]):
        ax.text(v, i, f"  {v:.3f}", va="center", fontsize=9)
    ax.set_xlabel(f"{metric} (mean of folds, bar = 1 sd)")
    headline(ax, title or f"Candidate models, cross-validated {metric}")
    return ax


def expanding_time_splits(time_index: pd.Series, cutoffs, gap=None):
    """Expanding-window folds on a time column: train on t < cutoff (minus an optional gap), test on the next period.
    `cutoffs` is a list of (train_end, test_end) timestamps. Yields (train_idx, test_idx)."""
    t = pd.to_datetime(pd.Series(time_index)).reset_index(drop=True)
    for train_end, test_end in cutoffs:
        train_end, test_end = pd.Timestamp(train_end), pd.Timestamp(test_end)
        tr_end = train_end - (pd.Timedelta(gap) if gap else pd.Timedelta(0))
        yield np.where(t < tr_end)[0], np.where((t >= train_end) & (t < test_end))[0]


class PredefinedSplits:
    """Wrap a list of (train_idx, test_idx) so it can be passed as `cv` to sklearn and cv_compare."""

    def __init__(self, splits):
        self.splits = list(splits)

    def split(self, X=None, y=None, groups=None):
        yield from self.splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.splits)


def tune_lgbm(X, y, cv, task: str = "binary", n_trials: int = 30, timeout: int = 420, scoring: str | None = None,
              seed: int = 42, fixed: dict | None = None):
    """Budgeted Optuna search (TPE) over the LightGBM parameters that matter, scored on the given folds.
    Returns (best_params, study). Kept small on purpose: the goal is a defensible challenger, not leaderboard tuning."""
    import lightgbm as lgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    scoring = scoring or ("roc_auc" if task == "binary" else "neg_root_mean_squared_error")
    Est = lgb.LGBMClassifier if task == "binary" else lgb.LGBMRegressor
    splits = list(cv.split(X, y))
    y = np.asarray(y)
    fixed = dict(fixed or {})
    base = {"subsample_freq": 1, "random_state": seed, "verbose": -1, "n_jobs": 4}
    space = {                                      # a parameter named in `fixed` is held at that value, not searched
        "n_estimators": lambda t: t.suggest_int("n_estimators", 150, 900, step=50),
        "learning_rate": lambda t: t.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "num_leaves": lambda t: t.suggest_int("num_leaves", 8, 96, log=True),
        "min_child_samples": lambda t: t.suggest_int("min_child_samples", 20, 400, log=True),
        "subsample": lambda t: t.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": lambda t: t.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": lambda t: t.suggest_float("reg_lambda", 1e-3, 20, log=True),
    }

    def objective(trial):
        params = {**base, **{k: f(trial) for k, f in space.items() if k not in fixed}, **fixed}
        vals = []
        for k, (tr, te) in enumerate(splits):
            m = Est(**params).fit(X.iloc[tr], y[tr])
            vals.append(get_scorer(scoring)(m, X.iloc[te], y[te]))
            trial.report(float(np.mean(vals)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(vals))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    best = {**base, **study.best_params, **fixed}
    return best, study


def plot_optuna(study, title: str = "Hyperparameter search"):
    """Trial scores and running best, plus each parameter's rank correlation with the score."""
    import matplotlib.pyplot as plt
    df = study.trials_dataframe()
    df = df[df["state"] == "COMPLETE"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    a1.scatter(df["number"], df["value"], s=18, color=PALETTE["grey"], label="trial")
    a1.plot(df["number"], df["value"].cummax(), color=PALETTE["coral"], label="best so far")
    a1.set(xlabel="Trial", ylabel="CV score", title="Search history")
    a1.legend()
    pcols = [c for c in df.columns if c.startswith("params_")]
    rho = df[pcols + ["value"]].corr(method="spearman")["value"].drop("value").sort_values()
    a2.barh([c.replace("params_", "") for c in rho.index], rho.values, color=[PALETTE["teal"] if v > 0 else PALETTE["coral"] for v in rho.values])
    a2.set(xlabel="Spearman rho with CV score", title="Which settings mattered")
    fig.suptitle(title, x=0.04, ha="left", fontsize=14, fontweight="bold", color=PALETTE["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def learning_curve_table(model, X, y, cv, scoring: str = "roc_auc", sizes=(0.1, 0.25, 0.5, 0.75, 1.0)) -> pd.DataFrame:
    """Train vs validation score as the training set grows: is the model data-limited or capacity-limited?"""
    from sklearn.model_selection import learning_curve
    n, tr, va = learning_curve(model, X, y, cv=cv, scoring=scoring, train_sizes=list(sizes), n_jobs=1, shuffle=True, random_state=42)
    return pd.DataFrame({"train_rows": n, "train_score": tr.mean(1), "valid_score": va.mean(1), "valid_std": va.std(1)})
