"""Unit tests for the shared dskit helpers (synthetic data only, so CI needs no raw files)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ds"))
from dskit import audit, eda, evaluate, report  # noqa: E402

RNG = np.random.default_rng(0)


def test_psi_is_zero_for_same_distribution_and_large_for_a_shift():
    a = RNG.normal(0, 1, 20000)
    assert eda.psi(a, RNG.normal(0, 1, 20000)) < 0.01
    assert eda.psi(a, RNG.normal(1.0, 1, 20000)) > 0.25


def test_wilson_interval_contains_the_rate_and_stays_in_bounds():
    lo, hi = eda.wilson(np.array([0, 5, 50]), np.array([10, 10, 50]))
    assert np.all(lo >= 0) and np.all(hi <= 1)
    assert lo[1] < 0.5 < hi[1]


def test_ks_and_auc_for_perfect_and_random_scores():
    y = np.r_[np.zeros(500), np.ones(500)]
    assert evaluate.ks_stat(y, y) == pytest.approx(1.0)
    m = evaluate.binary_metrics(y, RNG.uniform(size=1000))
    assert 0.44 < m["auc"] < 0.56


def test_gains_table_captures_every_event_once():
    y = RNG.integers(0, 2, 1000)
    t = evaluate.gains_table(y, RNG.uniform(size=1000), n=10)
    assert t["events"].sum() == y.sum()
    assert t["cum_capture"].iloc[-1] == pytest.approx(1.0)


def test_bootstrap_interval_brackets_the_estimate():
    y = RNG.integers(0, 2, 2000)
    p = y * 0.3 + RNG.uniform(size=2000)
    est, lo, hi = evaluate.bootstrap_ci(y, p, n=200)
    assert lo < est < hi


def test_threshold_table_precision_recall_are_consistent():
    y = np.r_[np.ones(100), np.zeros(900)]
    t = evaluate.threshold_table(y, y + RNG.uniform(0, 0.1, 1000), shares=[0.1, 1.0])
    assert t.loc[0, "precision"] == pytest.approx(1.0) and t.loc[0, "recall"] == pytest.approx(1.0)
    assert t.loc[1, "precision"] == pytest.approx(0.1)


def test_information_value_is_near_zero_for_noise_and_high_for_signal():
    n = 5000
    x = RNG.normal(size=n)
    df = pd.DataFrame({"noise": RNG.normal(size=n), "signal": x, "y": (x + RNG.normal(0, 0.5, n) > 1).astype(int)})
    iv = eda.iv_table(df, ["noise", "signal"], "y")
    assert iv.loc["noise", "iv"] < 0.05 and iv.loc["signal", "iv"] > 0.5


def test_leakage_screen_flags_a_copy_of_the_target():
    y = RNG.integers(0, 2, 1000)
    df = pd.DataFrame({"y": y, "leak": y + RNG.normal(0, 0.01, 1000), "x": RNG.normal(size=1000)})
    t = audit.leakage_screen(df, ["leak", "x"], "y")
    assert t.loc["leak", "review"] == "CHECK FOR LEAKAGE" and t.loc["x", "review"] == ""


def test_tune_lgbm_holds_fixed_parameters_out_of_the_search():
    pytest.importorskip("lightgbm")
    pytest.importorskip("optuna")
    from dskit import select
    X = pd.DataFrame(RNG.normal(size=(300, 3)), columns=list("abc"))
    y = (X["a"] + RNG.normal(0, 1, 300) > 0).astype(int)
    cv = select.PredefinedSplits([(np.arange(200), np.arange(200, 300))])
    best, study = select.tune_lgbm(X, y, cv, n_trials=2, timeout=60, fixed={"n_estimators": 20, "n_jobs": 1})
    assert best["n_estimators"] == 20 and best["n_jobs"] == 1
    assert "n_estimators" not in study.best_params


def test_regression_metrics_are_zero_for_a_perfect_forecast():
    y = RNG.uniform(10, 100, 100)
    m = evaluate.regression_metrics(y, y)
    assert m["mae"] == 0 and m["mape"] == 0 and m["r2"] == pytest.approx(1.0)


def test_update_metrics_writes_strict_json_for_non_finite_values(tmp_path):
    import json
    path = str(tmp_path / "metrics.json")
    report.update_metrics(path, "s", {"inf": float("inf"), "nan": np.nan, "x": np.float64(0.1234567)})
    data = json.loads(open(path, encoding="utf-8").read(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    assert data["s"] == {"inf": None, "nan": None, "x": 0.123457}
