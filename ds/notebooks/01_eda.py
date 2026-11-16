# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 01 · Subscriber churn: what the data supports before any model
#
# **Summary.** Cell2Cell has 51,047 subscribers, no duplicates and few true blanks. Missing values hide behind codes instead: handset
# price is "Unknown" for 57%, and age and income are zero for a quarter. One field is mislabelled: `PercChangeMinutes`
# is an absolute change in minutes, not a percentage. The README's driver claims reproduce exactly. Churn peaks at 12-17
# months in service (33.6% vs 24.0% under 12 months); handsets 18+ months old churn at 34.0% vs 22.9% for handsets under
# six months; retention-team callers churn at 45.0% vs 28.2%. Usage falling by half or more is the strongest behavioural
# warning (40% vs 26% for stable usage). Two folk beliefs fail: more care calls go with *less* churn, and low credit
# ratings churn less than high ones. Every signal is weak on its own (best single-feature AUC 0.57, best IV 0.14), so a
# modest model AUC is a property of this data, not a modelling failure. Retention-team contact stays in the model,
# because it is known before the churn window, but it is treated as a symptom. The sample over-represents churners (28.8%
# vs an assumed 2% a month). Ranking metrics carry over to a real base; probabilities and money need a prior correction
# first.

# %%
import os, sys, warnings
from pathlib import Path
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "ds" / "dskit").exists())
sys.path.insert(0, str(ROOT / "ds"))
warnings.simplefilter("ignore")              # library warnings print local paths into the outputs
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from dskit import set_style, PALETTE, headline, savefig, audit, eda, evaluate, explain, select, report
import features as F
set_style(); pd.set_option("display.float_format", lambda v: f"{v:,.4f}"); pd.set_option("display.width", 160)
FIG, METRICS, SEED = ROOT / "ds" / "figures", ROOT / "ds" / "reports" / "metrics.json", 42

# %% [markdown]
# ## 1. Business understanding
#
# | | |
# |---|---|
# | Decision | Which subscribers get a retention offer this month, and how many offers to fund |
# | Unit | Subscriber (one row each; 51,047 subscribers of a US wireless carrier, anonymised) |
# | Target | Churned in the observation window that follows the snapshot of usage, billing and care data |
# | Action | Offer a retention deal (e.g. a handset upgrade or a bill credit) to the top *k*% of a ranked list |
# | Success metric | Net retained revenue = saved churners x monthly revenue x months kept - offer spend |
# | Money at stake | Churners in this sample bill $58 a month on average, so each save is worth about $700 a year in revenue |
# | Constraints | Offer budget; no randomised retention test in the data (save rate unknown); the sample over-represents churners; the report's model uses no age, marital status, occupation or household data |
#
# The data is Cell2Cell (Teradata Center for CRM, Duke University), distributed for academic use. It is not
# redistributed here, and this notebook shows profiles and aggregates only, no subscriber rows.

# %%
d = F.load()
B = F.bands(d)
print(f"{len(d):,} subscribers, {d.shape[1] - 1} source columns; churn rate {d['Churned'].mean():.2%}; "
      f"mean monthly revenue ${d['MonthlyRevenue'].mean():.2f} (churners ${d.loc[d['Churned'] == 1, 'MonthlyRevenue'].mean():.2f})")
audit.duplicates(d.drop(columns=["Churned"]), key="CustomerID")

# %% [markdown]
# ## 2. Data audit
#
# Profile of every source column (types, completeness, ranges).

# %%
prof = audit.profile(d.drop(columns=["CustomerID", "Churned"]))
prof

# %%
ax = audit.plot_missing(d.drop(columns=["CustomerID", "Churned"]), title="Few true blanks: 14 of 57 columns, all under 2%")
plt.show()

# %% [markdown]
# True blanks are rare, but **missing values hide behind codes**. Size every sentinel and odd value, with the churn rate
# of the rows that carry it.

# %%
sentinels = {
    "HandsetPrice = 'Unknown'": d["HandsetPrice"].isna(),
    "AgeHH1 = 0 (age not known)": d["AgeHH1"] == 0,
    "IncomeGroup = 0 (income not known)": d["IncomeGroup"] == 0,
    "Usage block blank (revenue, minutes, overage...)": d["MonthlyRevenue"].isna(),
    "Usage change blank (PercChangeMinutes)": d["PercChangeMinutes"].isna(),
    "CurrentEquipmentDays < 0": d["CurrentEquipmentDays"] < 0,
    "MonthlyRevenue <= 0": d["MonthlyRevenue"] <= 0,
    "TotalRecurringCharge < 0": d["TotalRecurringCharge"] < 0,
    "ServiceArea blank": d["ServiceArea"].isna(),
}
sent = pd.DataFrame({k: {"rows": int(m.sum()), "share": m.mean(), "churn_rate_flagged": d.loc[m, "Churned"].mean() if m.any() else np.nan,
                         "churn_rate_rest": d.loc[~m, "Churned"].mean()} for k, m in sentinels.items()}).T
sent["rows"] = sent["rows"].astype(int)
print(f"Months in service: min {d['MonthsInService'].min():.0f}, so every subscriber has at least 6 months of tenure (a left-truncated sample)")
print(f"'PercChangeMinutes' ranges {d['PercChangeMinutes'].min():,.0f} to {d['PercChangeMinutes'].max():,.0f}: an absolute change in minutes, not a percentage")
sent

# %% [markdown]
# The codes matter more than the blanks. Unknown handset price covers 57% of subscribers and churns slightly more (30.2%
# vs 27.0%). The age and income zeros cover a quarter each. A blank usage block (156 subscribers) or usage change (367)
# goes with 45-57% churn. Every subscriber has at least six months of tenure, so the data says nothing about churn in the
# first six months.


# %% [markdown]
# Usage fields are heavy-tailed. Percentiles show where a few extreme accounts sit.

# %%
heavy = ["MonthlyRevenue", "MonthlyMinutes", "OverageMinutes", "RoamingCalls", "CustomerCareCalls", "DroppedBlockedCalls", "CurrentEquipmentDays"]
q = d[heavy].quantile([0.5, 0.9, 0.99, 0.999, 1.0]).T
q.columns = ["median", "p90", "p99", "p99.9", "max"]
q["max / p99"] = q["max"] / q["p99"]
q

# %% [markdown]
# **Audit decisions.**
#
# * **Handset price "Unknown" (57%)** is a missing code. The ETL keeps it as -1 for the tree model; the challenger adds an
#   explicit missing flag; the logistic pipeline imputes and flags it.
# * **Age and income zeros** are missing codes too, but these demographics stay out of the model, as in the report
#   (fairness, and they add little).
# * **The usage block is blank for 156 subscribers, and usage change for 367; those subscribers churn at 57%.** A blank
#   recent-usage field can be a *consequence* of leaving (an account winding down) rather than a cause. See the leakage
#   section.
# * **Negative handset ages (76) and non-positive revenue (9)** are data errors. Handset age is clipped at 0 in the
#   engineered features; there are too few to matter.
# * **Extremes stay in.** The max/p99 ratios of 3-15 are real heavy users (roaming, overage) rather than unit errors.
#   Trees are insensitive to them; the logistic pipeline uses a log transform.
# * No duplicate subscribers or rows.

# %% [markdown]
# ## 3. The target and the over-sampling
#
# 28.8% of this sample churned. A real carrier loses about 2% of subscribers a month (that is an assumption here; the
# original Duke tournament's representative sample is usually quoted near that level). Churners were over-sampled to
# give modellers enough of them. Three consequences:
#
# 1. **Rates in this notebook are relative risk.** "45% of retention callers churn" means they churn at 1.6x the base
#    rate, not that 45% of them leave next month.
# 2. **Ranking metrics survive.** AUC, lift and capture depend only on the ordering within each class, which
#    over-sampling does not change.
# 3. **Probabilities and money do not.** A model trained here predicts on the sample's scale. Before any expected value
#    (P(churn) x value x save rate - cost) is computed, the probabilities need a prior correction (notebook 02) and the
#    backtest needs reweighting to the real churn rate (notebook 03).

# %%
odds_ratio = (F.ASSUMED_MONTHLY_CHURN / (1 - F.ASSUMED_MONTHLY_CHURN)) / (F.SAMPLE_CHURN / (1 - F.SAMPLE_CHURN))
print(f"Each churner in the sample stands for {F.ASSUMED_MONTHLY_CHURN / F.SAMPLE_CHURN:.3f} real churners and each stayer for "
      f"{(1 - F.ASSUMED_MONTHLY_CHURN) / (1 - F.SAMPLE_CHURN):.3f} real stayers; odds scale by {odds_ratio:.4f}")

# %% [markdown]
# ## 4. Who churns? Rates with 95% Wilson intervals
#
# The drivers the report and README quote (tenure, handset age, care and retention calls), plus usage change, handset
# price and credit rating.

# %%
v = d.assign(**B)
v["UsageChange"] = eda.binned(F.engineer(d)["minutes_change_rel"], bins=[-5.01, -0.5, -0.1, 0.1, 0.5, 5.01],
                              labels=["Down 50%+", "Down 10-50%", "Stable", "Up 10-50%", "Up 50%+"])
v["UsageChange"] = v["UsageChange"].replace("Missing", "No history")
v["HandsetPriceBand"] = eda.binned(d["HandsetPrice"], bins=[0, 40, 100, 200, 500], labels=["Under $40", "$40-99", "$100-199", "$200+"]).replace("Missing", "Unknown")
panels = [("TenureBand", "Months in service", F.BAND_ORDER["TenureBand"]),
          ("HandsetAgeBand", "Age of current handset", F.BAND_ORDER["HandsetAgeBand"]),
          ("CareBand", "Customer-care calls per month", F.BAND_ORDER["CareBand"]),
          ("RetentionContact", "Retention team contact", ["No retention call", "Called retention team"]),
          ("UsageChange", "Minutes this month vs own average", ["Down 50%+", "Down 10-50%", "Stable", "Up 10-50%", "Up 50%+", "No history"]),
          ("CreditClass", "Credit rating", F.BAND_ORDER["CreditClass"])]
fig, axes = plt.subplots(2, 3, figsize=(17, 9.2))
tabs = {}
for ax, (col, xl, order) in zip(axes.flat, panels):
    r = v.groupby(col)["Churned"].mean()
    hi_lab, lo_lab = r.reindex(order).dropna().idxmax(), r.reindex(order).dropna().idxmin()
    tabs[col] = eda.rate_by(v, col, "Churned", ax=ax, order=order, rate_label="Churn rate", xlabel=xl, rotate=20,
                            title=f"{hi_lab}: {r[hi_lab]:.0%} vs {lo_lab}: {r[lo_lab]:.0%}", subtitle="Rate with 95% Wilson interval; bars = share of subscribers")
    ax.set_ylim(0, 66)
fig.suptitle("Handset age, retention calls and falling usage separate churners best; tenure peaks at 12-17 months",
             x=0.02, ha="left", fontsize=15, fontweight="bold", color=PALETTE["ink"])
fig.tight_layout(rect=(0, 0, 1, 0.95))
savefig(fig, "01-churn-drivers", FIG); plt.show()
pd.concat(tabs)[["rows", "rate", "ci_low", "ci_high"]]

# %% [markdown]
# * **Tenure:** churn rises from 24.0% under 12 months to a 33.6% peak at 12-17 months, then eases to 27.4% after three
#   years. That is the end of a one-year contract, when switching becomes free.
# * **Handset age:** a steady climb from 22.9% (under 6 months) to 36.1% (24+ months). An upgrade offer is the obvious
#   retention lever.
# * **Usage change:** a fall of 50% or more means 40.2% churn, against 26.4% for stable usage. A blank history means 56.7%.
# * **Retention-team callers:** 45.0% vs 28.2%. That is 1.6 times the base rate even after the team has spoken to them.
# * **Counter-intuitive:** care calls go with lower churn (31% with none, 25% with 2-5 a month), and the lowest churn is in
#   the low credit class (22%), plausibly because low-credit customers sit on contracts or deposits that make leaving
#   costly.
#
# All intervals are narrow (bands of 1,000+ subscribers), so these differences are not noise. They are associations,
# though, not effects of the variable.


# %%
old = d["CurrentEquipmentDays"] >= 540
new = d["CurrentEquipmentDays"] < 180
ret = d["MadeCallToRetentionTeam"] == "Yes"
readme = pd.DataFrame({
    "README claim": ["Churn peaks at 12-17 months in service", "Handsets 18+ months old vs under 6 months", "Retention-team callers vs everyone else"],
    "reproduced": [f"12-17 months: {tabs['TenureBand'].loc['12-17 months', 'rate']:.1%} (highest band)",
                   f"{d.loc[old, 'Churned'].mean():.1%} vs {d.loc[new, 'Churned'].mean():.1%}",
                   f"{d.loc[ret, 'Churned'].mean():.1%} vs {d.loc[~ret, 'Churned'].mean():.1%}"]})
assert tabs["TenureBand"]["rate"].idxmax() == "12-17 months"
readme

# %%
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2))
eda.dist_by_target(d, "CurrentEquipmentDays", "Churned", ax=a1, title="Churners carry older handsets")
eda.dist_by_target(d, "MonthlyMinutes", "Churned", ax=a2, log=True, title="Churners use slightly fewer minutes")
fig.tight_layout(); plt.show()

# %% [markdown]
# ## 5. Correlation structure
#
# Many usage fields count the same calls in different ways. The Spearman matrix over the numeric behaviour fields, and
# the pairs above |rho| 0.8:

# %%
usage = F.NUM + F.EXTRA_NUM
corr = eda.corr_heatmap(d, usage, title="Usage fields form one large block: minutes, calls and peak/off-peak counts move together")
savefig(plt.gcf(), "01-correlation", FIG); plt.show()
pairs = eda.high_correlation_pairs(d, usage, threshold=0.8)
pairs

# %% [markdown]
# Thirteen pairs exceed 0.8. Handsets issued and handset models are near-duplicates (0.91). Dropped and dropped-or-blocked
# calls overlap by construction (0.88). Received, outbound, peak and off-peak call counts form one block (0.80-0.87). The
# 19 ETL fields carry perhaps half a dozen independent signals.


# %% [markdown]
# Information value of each candidate, the ETL's features plus the engineered ratios, as a single-feature view:

# %%
cand = pd.concat([F.etl_matrix(d), F.engineer(d)], axis=1).assign(Churned=d["Churned"])
iv = eda.iv_table(cand, [c for c in cand.columns if c != "Churned"], "Churned")
iv.head(15)

# %% [markdown]
# No feature is even "strong" by the credit-scoring scale. Tenure (IV 0.14) and handset age (0.11) lead, and the
# engineered usage change relative to the customer's own minutes (0.07) is the best behavioural signal, more than twice
# the raw minute change (0.03).


# %% [markdown]
# ## 6. Leakage audit: when is each field known?
#
# Every field is measured over the months *before* the churn window, according to the source documentation, so none is
# a post-outcome field by construction. Two families still need judgement: fields that record the customer's own *intent*
# to leave, and blanks that may reflect an account already winding down.

# %%
screen = audit.leakage_screen(cand, [c for c in cand.columns if c != "Churned"], "Churned")
screen.head(10)

# %%
fam = {"Retention-team contact (MadeCallToRetentionTeam = Yes)": ret, "Retention offer accepted before": d["RetentionOffersAccepted"] > 0,
       "No usage-change history": d["PercChangeMinutes"].isna(), "Care calls 5+ a month": d["CustomerCareCalls"] >= 5}
leak = pd.DataFrame({k: {"subscribers": int(m.sum()), "share": m.mean(), "churn_if_yes": d.loc[m, "Churned"].mean(), "churn_if_no": d.loc[~m, "Churned"].mean(),
                         "share_of_all_churners": d.loc[m, "Churned"].sum() / d["Churned"].sum()} for k, m in fam.items()}).T
leak

# %% [markdown]
# **Leakage decisions.**
#
# * **The screen finds no leak.** No feature exceeds a single-feature AUC of 0.58, so there is no near-perfect predictor
#   and no sign of an outcome field. It also means the model must combine many weak signals.
# * **Retention-team contact is kept, as a symptom.** The call is recorded before the churn window, so it is known when
#   the list is scored and is a legitimate predictor. It is also the customer announcing an intent to leave, and these
#   3.4% of subscribers (5.3% of churners) are already in the retention team's hands. Notebook 02 measures how much of the
#   model's AUC depends on it; a model that mostly re-finds people who already called adds little. In the campaign, these
#   subscribers belong to the retention team's own process and should not be counted as model wins.
# * **A blank usage history is a possible consequence of leaving** (an account suspended or ported before the snapshot
#   closed): 367 subscribers churning at 57%. It is only 1.4% of churners, so it cannot carry much of the AUC. It is kept as
#   a flag and its effect is measured in notebook 02.
# * **Care calls are not a leak, and they point the other way from the folklore.** Subscribers with 5+ care calls a month
#   churn at 25%, those with none at 31%. Callers are engaged customers; the silent ones are the risk.
#
# ## 7. What EDA changed in the plan
#
# * **Expect a modest AUC and judge the model on money.** The strongest single feature reaches AUC 0.57 and IV 0.14. Lift at
#   the top of the list, and value captured, are the metrics that decide whether the model is worth running.
# * **Handle the codes explicitly.** Handset price "Unknown" gets a flag, the blank usage block is kept as missing for the
#   trees and imputed with a flag for the logistic model, and negative handset ages are clipped.
# * **Engineer ratios that normalise behaviour by the customer's own level.** Usage change relative to the customer's own
#   minutes (falling 50%+ means 40% churn), overage share, revenue per minute, dropped calls per call and care calls per
#   100 calls.
# * **Collinearity.** Thirteen pairs exceed |rho| 0.8 (handsets and handset models at 0.91, the call-count block). Trees cope,
#   the logistic model needs regularisation, and SHAP credit should be read by family.
# * **Correct the prior before any money.** Rescale probabilities from the 28.8% sample rate to an assumed 2% monthly churn,
#   and reweight the economics backtest the same way.
# * **Keep the ETL's split.** The data is a single snapshot with no dates, so out-of-time validation is impossible. A
#   stratified random 70/30 split is the honest best here, and the model card states the limitation.
# * **Fairness.** Age, marital status, occupation and household fields stay out, as in the report. Segment performance is
#   checked by credit class, tenure and revenue band.

# %%
report.update_metrics(METRICS, "01_eda", {
    "subscribers": len(d), "churn_rate_sample": float(d["Churned"].mean()), "assumed_monthly_churn": F.ASSUMED_MONTHLY_CHURN,
    "churn_by_tenure": {k: float(v_) for k, v_ in tabs["TenureBand"]["rate"].items()},
    "churn_old_vs_new_handset": [float(d.loc[old, "Churned"].mean()), float(d.loc[new, "Churned"].mean())],
    "churn_retention_callers_vs_rest": [float(d.loc[ret, "Churned"].mean()), float(d.loc[~ret, "Churned"].mean())],
    "churn_usage_missing": float(d.loc[d["PercChangeMinutes"].isna(), "Churned"].mean()),
    "high_correlation_pairs": int(len(pairs)), "max_single_feature_auc": float(screen["single_feature_auc"].max()),
})
print("metrics written")
