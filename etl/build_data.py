"""ETL + churn model for the Telecom Churn & Subscriber Value report.

Run with the project venv:  ..\\.venv\\Scripts\\python etl\\build_data.py

Source: Cell2Cell (Teradata Center for CRM at Duke University; distributed via Kaggle "Telecom churn (cell2cell)").
51,047 real, anonymised subscribers of a US wireless carrier: revenue, minutes, overage, roaming, dropped/blocked calls,
customer-care and retention-team calls, handset age and price, tenure, credit rating and churn.
Note: this is a modelling sample with churners over-represented, so churn rates are read as relative risk, not market churn.

Get the source file (not redistributed here): .raw/cell2celltrain.csv <- https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom
"""
import os, json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW, OUT = os.path.join(ROOT, ".raw"), os.path.join(ROOT, "data")
os.makedirs(OUT, exist_ok=True)
SEED = 42
metrics = {}


def save(df, name, fmt="%.6g"):
    df.to_csv(os.path.join(OUT, name), index=False, float_format=fmt)
    print(f"  {name:28s} {len(df):>9,} rows")


d = pd.read_csv(os.path.join(RAW, "cell2celltrain.csv"))
assert len(d) == 51047
d["Churned"] = (d["Churn"] == "Yes").astype(int)
num = ["MonthlyRevenue", "MonthlyMinutes", "TotalRecurringCharge", "OverageMinutes", "RoamingCalls", "PercChangeMinutes", "PercChangeRevenues",
       "DroppedCalls", "BlockedCalls", "UnansweredCalls", "CustomerCareCalls", "MonthsInService", "CurrentEquipmentDays", "HandsetPrice",
       "DroppedBlockedCalls", "RetentionCalls", "RetentionOffersAccepted", "ActiveSubs", "Handsets"]
for c in num:
    d[c] = pd.to_numeric(d[c], errors="coerce")
metrics["subscribers"] = len(d)
metrics["churn_rate_sample"] = round(d["Churned"].mean(), 4)


def band(s, edges, labels):
    b_ = pd.cut(s, edges, labels=labels, right=False).astype(object)
    return b_.where(b_.notna(), "Unknown").astype(str)


d["TenureBand"] = band(d["MonthsInService"], [0, 12, 18, 24, 36, 999], ["Under 12 months", "12-17 months", "18-23 months", "24-35 months", "36+ months"])
d["HandsetAgeBand"] = band(d["CurrentEquipmentDays"], [-999, 180, 365, 540, 730, 99999], ["Under 6 months", "6-12 months", "12-18 months", "18-24 months", "24+ months"])
d["ARPUBand"] = band(d["MonthlyRevenue"], [-999, 30, 45, 60, 80, 99999], ["Under $30", "$30-45", "$45-60", "$60-80", "$80+"])
d["CareBand"] = band(d["CustomerCareCalls"], [-1, 0.01, 2, 5, 9999], ["None", "Up to 2 a month", "2-5 a month", "5+ a month"])
d["DropBand"] = band(d["DroppedBlockedCalls"], [-1, 2, 5, 10, 20, 99999], ["Under 2 a month", "2-5", "5-10", "10-20", "20+"])
d["UsageTrend"] = np.select([d["PercChangeMinutes"] <= -100, d["PercChangeMinutes"] < -20, d["PercChangeMinutes"] <= 20],
                            ["Falling fast", "Falling", "Stable"], "Rising")
d.loc[d["PercChangeMinutes"].isna(), "UsageTrend"] = "No history"
d["Market"] = d["ServiceArea"].fillna("UNK").str[:3]
d["RetentionContact"] = np.where(d["MadeCallToRetentionTeam"] == "Yes", "Called retention team", "No retention call")
d["CreditClass"] = d["CreditRating"].fillna("Unknown").str.replace(r"^\d-", "", regex=True)

# ---- churn model: behaviour, handset, tenure, plan and credit (no age, marital status, occupation or household data)
feat_num = ["MonthlyRevenue", "MonthlyMinutes", "TotalRecurringCharge", "OverageMinutes", "RoamingCalls", "PercChangeMinutes", "PercChangeRevenues",
            "DroppedCalls", "BlockedCalls", "UnansweredCalls", "CustomerCareCalls", "MonthsInService", "CurrentEquipmentDays", "HandsetPrice",
            "RetentionCalls", "RetentionOffersAccepted", "ActiveSubs", "Handsets", "DroppedBlockedCalls"]
X = d[feat_num].copy()
X["HandsetPrice"] = X["HandsetPrice"].fillna(-1)
X["web_capable"] = (d["HandsetWebCapable"] == "Yes").astype(int)
X["refurbished"] = (d["HandsetRefurbished"] == "Yes").astype(int)
X["retention_call"] = (d["MadeCallToRetentionTeam"] == "Yes").astype(int)
X["credit"] = d["CreditRating"].fillna("0").str[0].astype(int)
y = d["Churned"].values
Xtr, Xte, ytr, yte, itr, ite = train_test_split(X, y, np.arange(len(X)), test_size=0.3, stratify=y, random_state=SEED)
gbm = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=2.0, random_state=SEED).fit(Xtr, ytr)
p_all = gbm.predict_proba(X)[:, 1]
p_te = p_all[ite]
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.5)).fit(Xtr.fillna(0), ytr)
metrics.update(churn_auc_test=round(roc_auc_score(yte, p_te), 4), churn_auc_train=round(roc_auc_score(ytr, p_all[itr]), 4),
               churn_auc_logit=round(roc_auc_score(yte, lr.predict_proba(Xte.fillna(0))[:, 1]), 4))
d["ChurnScore"] = np.round(p_all, 4)
d["Sample"] = "Train"
d.loc[ite, "Sample"] = "Test"
test = d.loc[ite].copy()
test["RiskDecile"] = 10 - pd.qcut(test["ChurnScore"].rank(method="first"), 10, labels=False)
test["ValueAtRisk"] = test["ChurnScore"] * test["MonthlyRevenue"].fillna(0)
test["ValueDecile"] = 10 - pd.qcut(test["ValueAtRisk"].rank(method="first"), 10, labels=False)
dec = test.groupby("RiskDecile")["Churned"].mean()
metrics["churn_top_decile_rate"] = round(dec.loc[1], 4)
metrics["churn_top_decile_lift"] = round(dec.loc[1] / test["Churned"].mean(), 2)
metrics["churn_top_30_capture"] = round(test.loc[test["RiskDecile"] <= 3, "Churned"].sum() / test["Churned"].sum(), 4)
rev = test["MonthlyRevenue"].fillna(0) * test["Churned"]
metrics["revenue_top_30_capture_by_value"] = round(rev[test["ValueDecile"] <= 3].sum() / rev.sum(), 4)
metrics["revenue_top_30_capture_by_risk"] = round(rev[test["RiskDecile"] <= 3].sum() / rev.sum(), 4)
print({k: v for k, v in metrics.items()})
d["RiskDecile"] = np.nan
d.loc[ite, "RiskDecile"] = test["RiskDecile"].values
d["ValueDecile"] = np.nan
d.loc[ite, "ValueDecile"] = test["ValueDecile"].values
pi = permutation_importance(gbm, Xte, yte, scoring="roc_auc", n_repeats=4, random_state=SEED)
LAB = {"MonthlyRevenue": "Monthly revenue", "MonthlyMinutes": "Monthly minutes", "TotalRecurringCharge": "Recurring charge", "OverageMinutes": "Overage minutes",
       "RoamingCalls": "Roaming calls", "PercChangeMinutes": "Change in minutes", "PercChangeRevenues": "Change in revenue", "DroppedCalls": "Dropped calls",
       "BlockedCalls": "Blocked calls", "UnansweredCalls": "Unanswered calls", "CustomerCareCalls": "Customer-care calls", "MonthsInService": "Tenure",
       "CurrentEquipmentDays": "Handset age", "HandsetPrice": "Handset price", "RetentionCalls": "Retention calls", "RetentionOffersAccepted": "Retention offers accepted",
       "ActiveSubs": "Active lines", "Handsets": "Handsets issued", "DroppedBlockedCalls": "Dropped or blocked calls", "web_capable": "Web-capable handset",
       "refurbished": "Refurbished handset", "retention_call": "Called retention team", "credit": "Credit rating"}
imp = pd.DataFrame({"Driver": [LAB[c] for c in X.columns], "AUCDrop": pi.importances_mean.round(4)}).sort_values("AUCDrop", ascending=False).head(12)

subs = d[["CustomerID", "Churned", "MonthlyRevenue", "MonthlyMinutes", "OverageMinutes", "CustomerCareCalls", "DroppedBlockedCalls", "MonthsInService",
          "CurrentEquipmentDays", "HandsetPrice", "TenureBand", "HandsetAgeBand", "ARPUBand", "CareBand", "DropBand", "UsageTrend", "Market",
          "RetentionContact", "CreditClass", "ChurnScore", "Sample", "RiskDecile", "ValueDecile"]].rename(columns={"CustomerID": "SubscriberKey"})
subs["MonthlyRevenue"] = subs["MonthlyRevenue"].round(2)
ORDERS = {"TenureBand": ["Under 12 months", "12-17 months", "18-23 months", "24-35 months", "36+ months", "Unknown"],
          "HandsetAgeBand": ["Under 6 months", "6-12 months", "12-18 months", "18-24 months", "24+ months", "Unknown"],
          "ARPUBand": ["Under $30", "$30-45", "$45-60", "$60-80", "$80+", "Unknown"], "CareBand": ["None", "Up to 2 a month", "2-5 a month", "5+ a month", "Unknown"],
          "DropBand": ["Under 2 a month", "2-5", "5-10", "10-20", "20+", "Unknown"], "UsageTrend": ["Falling fast", "Falling", "Stable", "Rising", "No history"]}
for c, order in ORDERS.items():
    subs[c + "Order"] = subs[c].map({v: k for k, v in enumerate(order, 1)})
save(subs, "fact_subscriber.csv")
save(imp, "churn_drivers.csv")
save(pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"]), "model_metrics.csv", fmt="%.10g")
print(json.dumps(metrics, indent=2))
