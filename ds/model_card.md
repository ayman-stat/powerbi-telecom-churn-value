# Model card: subscriber churn score and retention targeting

The churn score feeds the Power BI report's risk pages and retention simulator. It is built in `etl/build_data.py` and
reproduced, challenged and extended in `ds/notebooks/`. Numbers below come from `ds/reports/metrics.json`, written by the
notebooks.

| | |
|---|---|
| **Intended use** | Rank subscribers each month for a retention offer, and size the offer budget with prior-corrected expected value. Not for credit, pricing or any decision about an individual without human review. |
| **Owner** | Mohamed Ayman (CVM data science) |
| **Data** | Cell2Cell (Teradata Center for CRM, Duke University; academic use, not redistributed): 51,047 anonymised subscribers of a US wireless carrier, one snapshot. Churners are over-sampled (28.8% vs an assumed 2% a month in a real base). All subscribers have at least 6 months of tenure. |
| **Target** | Churned in the window after the usage, billing and care snapshot. |
| **Features** | Production: 23 fields covering usage and revenue, usage change, network quality (dropped, blocked and unanswered calls), care and retention contacts, tenure, handset age, price and type, active lines and credit rating. Challenger: those plus 13 behavioural fields and 12 engineered ratios (usage change relative to own usage, overage share, revenue per minute, dropped calls per call, care calls per 100 calls). No age, marital status, occupation or household data. |
| **Design** | The ETL's stratified random 70/30 split (seed 42): 35,732 training and 15,315 hold-out subscribers. With no dates in the data, out-of-time validation is impossible. Model choice uses 3-fold CV inside the training data; the hold-out is used once. |
| **Production model** | `HistGradientBoostingClassifier` (400 iterations, learning rate 0.04, 31 leaves, 100 rows per leaf, L2 2.0; sklearn's automatic early stopping is active at this size). |

## Performance on the hold-out (15,315 subscribers, 28.8% churn)

| Model | AUC | 95% CI | PR AUC | Top-decile churn (lift) | Churners in top 30% |
|---|---|---|---|---|---|
| Rule: handset age | 0.572 | 0.563 - 0.582 | 0.333 | 34.7% (1.20) | 34.5% |
| Logistic, as published (blanks to 0) | 0.601 | 0.592 - 0.611 | 0.367 | 42.2% (1.47) | 38.6% |
| Logistic pipeline (impute + flags + log + L2) | 0.621 | 0.611 - 0.630 | 0.393 | 46.4% (1.61) | 41.0% |
| **Production champion (GBM)** | **0.672** | 0.664 - 0.681 | 0.452 | 53.9% (1.87) | 45.9% |
| Tuned LightGBM challenger (48 inputs) | 0.682 | 0.674 - 0.692 | 0.468 | 56.0% (1.94) | 46.8% |

Paired bootstrap on the same subscribers: challenger vs champion +0.010 (95% CI +0.006 to +0.014); champion vs logistic
pipeline +0.051 (+0.043 to +0.060); logistic pipeline vs published logistic +0.020 (+0.013 to +0.027). Every published
model metric (`data/model_metrics.csv`) is reproduced exactly and asserted in notebook 02.

**Champion/challenger finding.** The tuned LightGBM is better and is recommended after one month of shadow scoring. The
published numbers stay as the production record until then. Linear models are not competitive on this data.

**AUC in context.** An AUC around 0.67 is in line with what is usually reported for Cell2Cell. No single field separates
churners (best single-feature AUC 0.57), and the learning curve adds only 0.007 over the last 40% of training rows.

## Calibration and the prior correction

On the sample's scale, the score is calibrated (mean prediction 28.8% vs 28.8% observed, decile by decile). For a real
base, probabilities are rescaled by the odds ratio (0.02/0.98) / (0.288/0.712) = 0.050. On the hold-out reweighted to a
2% churn rate, the riskiest decile is predicted at 5.7% and churns at 5.6%. That is a consistency check of the correction,
not a validation of the 2% assumption, which only billing data can confirm.

## Segment performance (champion, hold-out)

| Segment | AUC range | Calibration ratio range | Note |
|---|---|---|---|
| Tenure band | 0.63 - 0.72 | 0.97 - 1.07 | Best for subscribers under 12 months |
| Handset age | 0.60 - 0.71 | 0.96 - 1.04 | Weakest for handsets 24+ months old, where almost everyone is at risk |
| Revenue band | 0.65 - 0.69 | 0.93 - 1.04 | |
| Credit class | 0.65 - 0.68 | 0.88 - 1.09 | Churn under-predicted for the Lowest class (30.8% vs 27.2%); monitor |

## Explainability

Handset age, tenure, usage change and monthly minutes lead in both SHAP and permutation importance (rank correlation
0.93), which matches the published driver chart. Old handsets and falling or low usage raise risk; long tenure, a high
recurring charge and a worse credit rating lower it. Leakage sensitivity: removing the retention-contact fields costs
0.002 AUC (CI -0.004 to +0.000), and removing the blank-usage signal costs 0.001. Neither carries the model.

## Decision use (notebook 03)

* **Rank by revenue at risk** (P(churn) x monthly revenue). The top 30% then hold 58.0% of churned revenue, vs 44.1%
  when ranking by risk.
* **The report's default simulation** ($25 offer, 20% save rate, 12 months, top 30%) nets $234K (ROI 204%) on the
  over-sampled hold-out. On a base reweighted to 2% monthly churn, the same campaign loses about $80K. Even the top
  decile pays back only below about $9 per offer.
* **Use expected-value targeting** with prior-corrected probabilities. A $5 offer then goes to 15% of the base and nets
  $6.2K per 15,315 subscribers, against $6.5K expected.
* **Measure the save rate first.** Across 2,000 Monte Carlo draws, a $5 offer to the top 10% still loses money 38% of the
  time. An 80/20 randomised hold-out on the top 10% needs about 10,700 subscribers per arm (50/50) to detect a 20% save
  rate.

## Limitations and ethics

* Random rather than out-of-time validation, and early-2000s data: expect lower performance on a current base.
* The over-sampling correction rests on an assumed 2% monthly churn and on churners being sampled at random among
  churners.
* Field timing (retention calls in particular) is not fully documented; the sensitivity refit shows the model does not
  rely on it.
* Credit rating is used and can proxy for protected characteristics; calibration for the Lowest credit class is
  monitored. Demographic fields are excluded by design.
* The economics are a backtest with assumed save rate, offer cost, months kept and margin, not a causal estimate.

## Monitoring and retraining

Feature and score PSI at every monthly scoring run (watch > 0.10, act > 0.25). Real monthly churn from billing (update
the prior correction if it is outside 1.5%-2.5%). AUC < 0.65 or top-decile lift < 1.6 on the matured list triggers a
refit. Keep a 10-20% random hold-out from every offer list. Retrain quarterly; every refit must beat the live model on a
paired test.
