"""Data loading and feature engineering for the telecom notebooks (imported by the notebooks and by tests/test_features.py).

Mirrors etl/build_data.py, the production ETL behind the Power BI report: the same file, the same 19 numeric fields plus
four flags, the same stratified 70/30 split (seed 42) and the same gradient-boosting champion, so the notebooks reproduce
its published metrics. On top of that: engineered ratios, a leakage-aware feature list, the prior correction for the
over-sampled churn rate and the retention-economics backtest used by the report's simulator.

Cell2Cell is distributed for academic use and is not redistributed here; notebooks show profiles and aggregates only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / ".raw"
SEED = 42

# the ETL's model inputs (no age, marital status, occupation or household data)
NUM = ["MonthlyRevenue", "MonthlyMinutes", "TotalRecurringCharge", "OverageMinutes", "RoamingCalls", "PercChangeMinutes", "PercChangeRevenues",
       "DroppedCalls", "BlockedCalls", "UnansweredCalls", "CustomerCareCalls", "MonthsInService", "CurrentEquipmentDays", "HandsetPrice",
       "RetentionCalls", "RetentionOffersAccepted", "ActiveSubs", "Handsets", "DroppedBlockedCalls"]
FLAGS = ["web_capable", "refurbished", "retention_call", "credit"]
ETL_FEATS = NUM + FLAGS
GBM_PARAMS = dict(max_iter=400, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=2.0, random_state=SEED)
# extra behavioural fields the ETL leaves out (still no demographics)
EXTRA_NUM = ["DirectorAssistedCalls", "ThreewayCalls", "ReceivedCalls", "OutboundCalls", "InboundCalls", "PeakCallsInOut", "OffPeakCallsInOut",
             "CallForwardingCalls", "CallWaitingCalls", "UniqueSubs", "HandsetModels", "AdjustmentsToCreditRating", "ReferralsMadeBySubscriber"]
ENGINEERED = ["minutes_change_rel", "revenue_change_rel", "overage_share", "recurring_share", "revenue_per_minute", "drop_rate", "care_per_100_calls",
              "unanswered_share", "handset_age_years", "handset_price_missing", "usage_missing", "outbound_share"]
LABEL = {"MonthlyRevenue": "Monthly revenue", "MonthlyMinutes": "Monthly minutes", "TotalRecurringCharge": "Recurring charge",
         "OverageMinutes": "Overage minutes", "RoamingCalls": "Roaming calls", "PercChangeMinutes": "Change in minutes",
         "PercChangeRevenues": "Change in revenue", "DroppedCalls": "Dropped calls", "BlockedCalls": "Blocked calls",
         "UnansweredCalls": "Unanswered calls", "CustomerCareCalls": "Customer-care calls", "MonthsInService": "Tenure, months",
         "CurrentEquipmentDays": "Handset age, days", "HandsetPrice": "Handset price", "RetentionCalls": "Retention calls",
         "RetentionOffersAccepted": "Retention offers accepted", "ActiveSubs": "Active lines", "Handsets": "Handsets issued",
         "DroppedBlockedCalls": "Dropped or blocked calls", "web_capable": "Web-capable handset", "refurbished": "Refurbished handset",
         "retention_call": "Called retention team", "credit": "Credit rating (1 best)", "minutes_change_rel": "Minutes change vs usage",
         "revenue_change_rel": "Revenue change vs revenue", "overage_share": "Overage share of minutes",
         "recurring_share": "Recurring share of revenue",
         "revenue_per_minute": "Revenue per minute", "drop_rate": "Dropped/blocked per call", "care_per_100_calls": "Care calls per 100 calls",
         "unanswered_share": "Unanswered share of calls", "handset_age_years": "Handset age, years", "handset_price_missing": "Handset price unknown",
         "usage_missing": "No usage history", "outbound_share": "Outbound share of calls"}

SAMPLE_CHURN = 0.2882          # churn rate of this modelling sample (churners over-represented)
ASSUMED_MONTHLY_CHURN = 0.02   # assumption for a real base (the original tournament's representative sample is usually quoted near 2%)


def load(raw: Path = RAW) -> pd.DataFrame:
    """The raw file with the ETL's type fixes: numeric fields coerced and the 0/1 target."""
    d = pd.read_csv(Path(raw) / "cell2celltrain.csv")
    d["Churned"] = (d["Churn"] == "Yes").astype(int)
    for c in NUM + EXTRA_NUM:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def etl_matrix(d: pd.DataFrame) -> pd.DataFrame:
    """The ETL's design matrix: NaN kept for the gradient-boosting model (handled natively), handset price unknown -> -1."""
    X = d[NUM].copy()
    X["HandsetPrice"] = X["HandsetPrice"].fillna(-1)
    X["web_capable"] = (d["HandsetWebCapable"] == "Yes").astype(int)
    X["refurbished"] = (d["HandsetRefurbished"] == "Yes").astype(int)
    X["retention_call"] = (d["MadeCallToRetentionTeam"] == "Yes").astype(int)
    X["credit"] = d["CreditRating"].fillna("0").str[0].astype(int)
    return X


def engineer(d: pd.DataFrame) -> pd.DataFrame:
    """Ratios that put usage, cost and service quality on a per-customer scale. Each is a row-wise transform (nothing is
    fitted), so it can be built before the split without leaking. `PercChangeMinutes` is, despite its name, an absolute
    change in minutes (range -3,875 to +5,192), so it is divided by the customer's own usage."""
    calls = (d["OutboundCalls"] + d["InboundCalls"]).clip(lower=1)
    mins = d["MonthlyMinutes"].clip(lower=1)
    rev = d["MonthlyRevenue"].where(d["MonthlyRevenue"] > 0)
    return pd.DataFrame({
        "minutes_change_rel": (d["PercChangeMinutes"] / mins).clip(-5, 5),
        "revenue_change_rel": (d["PercChangeRevenues"] / rev).clip(-5, 5),
        "overage_share": (d["OverageMinutes"] / mins).clip(0, 5),
        "recurring_share": (d["TotalRecurringCharge"] / rev).clip(-1, 5),
        "revenue_per_minute": (rev / mins).clip(upper=50),
        "drop_rate": (d["DroppedBlockedCalls"] / calls).clip(upper=5),
        "care_per_100_calls": (100 * d["CustomerCareCalls"] / calls).clip(upper=500),
        "unanswered_share": (d["UnansweredCalls"] / calls).clip(upper=10),
        "handset_age_years": d["CurrentEquipmentDays"].clip(lower=0) / 365.25,
        "handset_price_missing": d["HandsetPrice"].isna().astype(int),
        "usage_missing": d["PercChangeMinutes"].isna().astype(int),
        "outbound_share": (d["OutboundCalls"] / calls).clip(upper=5),
    }, index=d.index)


def full_matrix(d: pd.DataFrame) -> pd.DataFrame:
    """Challenger inputs: the ETL's features, the extra behavioural fields and the engineered ratios."""
    X = etl_matrix(d)
    return pd.concat([X, d[EXTRA_NUM], engineer(d)], axis=1)


def split(d: pd.DataFrame, test_size: float = 0.3, seed: int = SEED):
    """The ETL's stratified 70/30 split; returns (train_idx, test_idx) as positions."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(d))
    tr, te = train_test_split(idx, test_size=test_size, stratify=d["Churned"].values, random_state=seed)
    return tr, te


def bands(d: pd.DataFrame) -> pd.DataFrame:
    """The report's bands (same edges and labels as the ETL), missing -> 'Unknown'."""
    def band(s, edges, labels):
        b = pd.cut(s, edges, labels=labels, right=False).astype(object)
        return b.where(b.notna(), "Unknown").astype(str)
    return pd.DataFrame({
        "TenureBand": band(d["MonthsInService"], [0, 12, 18, 24, 36, 999],
                           ["Under 12 months", "12-17 months", "18-23 months", "24-35 months", "36+ months"]),
        "HandsetAgeBand": band(d["CurrentEquipmentDays"], [-999, 180, 365, 540, 730, 99999],
                               ["Under 6 months", "6-12 months", "12-18 months", "18-24 months", "24+ months"]),
        "ARPUBand": band(d["MonthlyRevenue"], [-999, 30, 45, 60, 80, 99999], ["Under $30", "$30-45", "$45-60", "$60-80", "$80+"]),
        "CareBand": band(d["CustomerCareCalls"], [-1, 0.01, 2, 5, 9999], ["None", "Up to 2 a month", "2-5 a month", "5+ a month"]),
        "CreditClass": d["CreditRating"].fillna("Unknown").str.replace(r"^\d-", "", regex=True),
        "RetentionContact": np.where(d["MadeCallToRetentionTeam"] == "Yes", "Called retention team", "No retention call"),
    }, index=d.index)


BAND_ORDER = {"TenureBand": ["Under 12 months", "12-17 months", "18-23 months", "24-35 months", "36+ months"],
              "HandsetAgeBand": ["Under 6 months", "6-12 months", "12-18 months", "18-24 months", "24+ months"],
              "ARPUBand": ["Under $30", "$30-45", "$45-60", "$60-80", "$80+"], "CareBand": ["None", "Up to 2 a month", "2-5 a month", "5+ a month"],
              "CreditClass": ["Highest", "High", "Good", "Medium", "Low", "VeryLow", "Lowest"]}


def signed_log1p(x):
    """log(1 + |x|) with the sign kept: tames heavy-tailed usage and change fields that can be negative."""
    x = np.asarray(x, float)
    return np.sign(x) * np.log1p(np.abs(x))


def prior_correct(p, sample_rate: float = SAMPLE_CHURN, true_rate: float = ASSUMED_MONTHLY_CHURN) -> np.ndarray:
    """Bayes prior correction for a model trained on a sample with churners over-represented: rescale the odds by
    (true odds / sample odds). Ranking is unchanged; probabilities move to the population scale."""
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    r = (true_rate / (1 - true_rate)) / (sample_rate / (1 - sample_rate))
    odds = p / (1 - p) * r
    return odds / (1 + odds)


def population_weights(y, sample_rate: float = SAMPLE_CHURN, true_rate: float = ASSUMED_MONTHLY_CHURN) -> np.ndarray:
    """Case weights that turn the over-sampled backtest into a base with the true churn rate (same total head count)."""
    y = np.asarray(y)
    return np.where(y == 1, true_rate / sample_rate, (1 - true_rate) / (1 - sample_rate))


def deciles(score) -> np.ndarray:
    """1 = highest score, as in the ETL (ties broken by order of appearance)."""
    s = pd.Series(np.asarray(score, float))
    return (10 - pd.qcut(s.rank(method="first"), 10, labels=False)).values


def simulate(test: pd.DataFrame, rank_col: str, share_pct: int, offer_cost: float = 25.0, save_rate: float = 0.2, months: int = 12,
             margin: float = 1.0, weights=None) -> dict:
    """The report's retention simulator on the hold-out base: offer a deal to the top `share_pct`% (whole deciles of
    `rank_col`, 1 = first); revenue retained = saved churners' monthly revenue x months x margin; net = retained - offer
    spend. `weights` reweights subscribers (e.g. to the true churn rate)."""
    w = np.ones(len(test)) if weights is None else np.asarray(weights, float)
    sel = test[rank_col].values <= share_pct / 10
    y, rev = test["Churned"].values, test["MonthlyRevenue"].fillna(0).values
    retained = float(np.sum(w * sel * y * rev) * save_rate * months * margin)
    targeted = float(np.sum(w * sel))
    spend = targeted * offer_cost
    return {"targeted": targeted, "churners_reached": float(np.sum(w * sel * y)), "retained": retained, "spend": spend,
            "net": retained - spend, "roi": (retained - spend) / spend if spend else np.nan}
