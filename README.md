# Telecom Churn & Subscriber Value

How a mobile operator's CVM team decides who gets a retention offer: churn drivers, a churn model and a retention-economics simulator on 51K real subscribers.

**Domain:** Telecom / CVM  ·  **Built with:** Power BI Desktop (PBIP, PBIR, TMDL), DAX, Python  ·  **Author:** [Mohamed Ayman](https://mohamedaymanstat.vercel.app)

![Subscriber base](docs/img/01-subscriber-base.png)

## What the data says

- Churn peaks at 12-17 months in service, when contracts end, and on handsets 18+ months old (34% vs 23% on new handsets).
- Customers who call the retention team churn at 45% vs 28%: a symptom, not a cure.
- Ranking by revenue at risk finds 58% of lost revenue in the top 30% of the base, vs 44% when ranking by churn risk.
- At default assumptions, value-ranked offers return $234K net, 55% more than ranking by risk.

## Pages

| Page | Preview |
|---|---|
| 1. Subscriber base | <img src="docs/img/01-subscriber-base.png" width="420"> |
| 2. Network & care | <img src="docs/img/02-network-care.png" width="420"> |
| 3. Churn model | <img src="docs/img/03-churn-model.png" width="420"> |
| 4. Retention economics | <img src="docs/img/04-retention-economics.png" width="420"> |
| 5. About & method | <img src="docs/img/05-about-method.png" width="420"> |

## Model and DAX

- Retention simulator with four what-if inputs (ranking method, share targeted, offer cost, save rate), backtested on the hold-out base.
- Net-value curve for both ranking methods and a measure that finds the best share under the chosen assumptions.

## Machine learning

- Gradient boosting on usage, network, care, handset, tenure, plan and credit features; no age, marital status or occupation.
- Hold-out AUC 0.67 vs 0.60 for logistic regression, in line with published results on this dataset.

## Design

- Identity: **Telco**: a deep-teal hero band carries the action title and KPIs in white, amber marks the selected page.
- Every page states its finding as the title, names its source in the footer and uses readable field names.
- Value-driven colour (gradients) on bars, conditional formatting in tables, numbers right-aligned under right-aligned headers.
- Each report in this portfolio has its own visual identity; see the other repos linked below.

## Data

- [Cell2Cell (Teradata Center for CRM, Duke University)](https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom): 51,047 anonymised subscribers of a US wireless carrier; churners are over-represented (modelling sample). Licence: Academic use.

The source licence does not allow redistribution, so `data/` is not committed. To rebuild it, download the source files into `.raw/` as described at the top of `etl/build_data.py`, then run the ETL.

## How to run

1. Open `TelecomChurnValue.pbip` in Power BI Desktop (October 2026 or later; PBIR and DAX user-defined functions).
2. Build the data (step below), set the `DataSource` parameter to `Local` and `LocalDataFolder` to this repo's `data\` folder, then refresh.
3. To rebuild everything from source:

```bash
pip install pandas numpy scikit-learn scipy openpyxl xlrd pyarrow
python etl/build_data.py      # source data -> data/*.csv (and models, where relevant)
python build/model.py         # writes the TMDL semantic model
python build/report.py        # writes the PBIR report
```

The report and model are generated from code (`build/`, using a shared design toolkit), so every visual, measure and page is reviewable in a diff.

## Limitations

- Churn rates are relative risk, not market churn, because the sample over-represents churners.
- Retention contact is observational: there is no randomised retention test in the data.

## More in this portfolio

- [Olist Marketplace Performance](https://github.com/ayman-stat/powerbi-olist-marketplace): E-commerce / marketplace
- [Retail Bank Campaign Strategy](https://github.com/ayman-stat/powerbi-bank-campaign-strategy): Banking / direct marketing
- [CVM Command Centre: retention, churn and offer experiments](https://github.com/ayman-stat/powerbi-cvm-experiments): CVM / retention / experimentation
- [Retail Network Performance](https://github.com/ayman-stat/powerbi-retail-network): Retail / store operations
- [Card Portfolio & Risk](https://github.com/ayman-stat/powerbi-card-portfolio-risk): Payments / cards / credit risk

---
Mohamed Ayman, Senior Data Scientist  ·  [Portfolio](https://mohamedaymanstat.vercel.app)  ·  [GitHub](https://github.com/ayman-stat)  ·  [LinkedIn](https://www.linkedin.com/in/mohamed-ghareb/)
