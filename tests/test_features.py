"""Project tests on small synthetic frames (the Cell2Cell file is not redistributed, so CI never needs it):
the ETL design matrix, engineered ratios, bands, the split, the prior correction and the retention simulator."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ds"))
import features as F  # noqa: E402

RNG = np.random.default_rng(0)


def synthetic(n=200):
    d = pd.DataFrame({c: RNG.uniform(0, 100, n) for c in F.NUM + F.EXTRA_NUM})
    d["HandsetPrice"] = np.where(RNG.random(n) < 0.5, np.nan, 150.0)
    d["PercChangeMinutes"] = RNG.normal(0, 50, n)
    d.loc[:4, "PercChangeMinutes"] = np.nan
    d["HandsetWebCapable"] = RNG.choice(["Yes", "No"], n)
    d["HandsetRefurbished"] = RNG.choice(["Yes", "No"], n)
    d["MadeCallToRetentionTeam"] = RNG.choice(["Yes", "No"], n)
    d["CreditRating"] = RNG.choice(["1-Highest", "2-High", "5-Low", "7-Lowest"], n)
    d["Churned"] = (RNG.random(n) < 0.3).astype(int)
    return d


def test_etl_matrix_codes_unknown_price_and_flags():
    d = synthetic()
    X = F.etl_matrix(d)
    assert list(X.columns) == F.ETL_FEATS
    assert (X.loc[d["HandsetPrice"].isna(), "HandsetPrice"] == -1).all()
    assert set(X["credit"]) <= {1, 2, 5, 7}
    assert (X["retention_call"] == (d["MadeCallToRetentionTeam"] == "Yes")).all()


def test_engineered_ratios_are_row_wise_so_they_cannot_leak_across_the_split():
    d = synthetic()
    full = F.engineer(d)
    part = F.engineer(d.iloc[:50])
    pd.testing.assert_frame_equal(full.iloc[:50], part)
    assert full["usage_missing"].iloc[:5].eq(1).all() and full["usage_missing"].iloc[5:].eq(0).all()
    zero = d.iloc[10:11].copy()                              # a row with usage history
    zero[["MonthlyMinutes", "OutboundCalls", "InboundCalls", "MonthlyRevenue"]] = 0
    assert np.isfinite(F.engineer(zero).drop(columns=["revenue_change_rel", "recurring_share", "revenue_per_minute"]).values).all()


def test_split_is_stratified_and_reproducible():
    d = synthetic(1000)
    tr, te = F.split(d)
    tr2, te2 = F.split(d)
    assert (te == te2).all() and len(te) == 300 and len(set(tr) & set(te)) == 0
    assert abs(d["Churned"].iloc[te].mean() - d["Churned"].mean()) < 0.01


def test_bands_use_the_report_edges():
    d = pd.DataFrame({"MonthsInService": [6, 12, 17, 18, 40, np.nan], "CurrentEquipmentDays": [10, 200, 400, 600, 800, np.nan],
                      "MonthlyRevenue": [10, 30, 50, 70, 90, np.nan], "CustomerCareCalls": [0, 1, 3, 6, 0, np.nan],
                      "CreditRating": ["1-Highest"] * 5 + [None], "MadeCallToRetentionTeam": ["No"] * 6})
    b = F.bands(d)
    assert b["TenureBand"].tolist() == ["Under 12 months", "12-17 months", "12-17 months", "18-23 months", "36+ months", "Unknown"]
    assert b["HandsetAgeBand"].tolist()[:5] == ["Under 6 months", "6-12 months", "12-18 months", "18-24 months", "24+ months"]
    assert b["CreditClass"].tolist()[-1] == "Unknown"


def test_prior_correction_rescales_the_odds_and_keeps_the_ranking():
    p = np.array([0.1, 0.2882, 0.5, 0.9])
    q = F.prior_correct(p, 0.2882, 0.02)
    assert (np.argsort(p) == np.argsort(q)).all()
    assert q[1] == pytest.approx(0.02)                      # the sample's base rate maps to the true base rate
    odds = lambda x: x / (1 - x)
    assert np.allclose(odds(q) / odds(p), (0.02 / 0.98) / (0.2882 / 0.7118))


def test_population_weights_recreate_the_true_churn_rate_with_the_same_head_count():
    y = np.r_[np.ones(2882), np.zeros(7118)]
    w = F.population_weights(y, 0.2882, 0.02)
    assert np.average(y, weights=w) == pytest.approx(0.02)
    assert w.sum() == pytest.approx(len(y))


def test_simulator_matches_a_hand_calculation():
    test = pd.DataFrame({"Churned": [1, 0, 1, 0], "MonthlyRevenue": [50.0, 80.0, 20.0, np.nan], "Rank": [1, 2, 6, 10]})
    r = F.simulate(test, "Rank", 20, offer_cost=25, save_rate=0.2, months=12)
    assert r["targeted"] == 2 and r["churners_reached"] == 1
    assert r["retained"] == pytest.approx(50 * 0.2 * 12) and r["net"] == pytest.approx(120 - 50)
    assert r["roi"] == pytest.approx(70 / 50)
