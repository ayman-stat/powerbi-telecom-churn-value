"""Report definition (PBIR) for the Telecom Churn & Subscriber Value report.

Identity: TELCO - a deep-teal hero band carries the action title and KPIs in white, amber marks the selected page and the
single highlight, body cards sit on a cool grey page. Modelled on how operators run churn and retention (CVM) teams;
not affiliated with any operator.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "toolkit"))  # vendored copy of the shared design toolkit
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_tools"))  # master copy, when present
from pbip import Report, write_pbip, C, M, proj, b, i, n, s, props, idprops, color  # noqa: E402
import design as D  # noqa: E402
from design import S, GUT, use, theme, chrome, kpi_band, bar, combo, line, matrix, table, P, _title, fill_for, about_page  # noqa: E402

use(D.TELCO)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NAME = "TelecomChurnValue"
TITLE = "Telecom Churn & Subscriber Value"
SRC = "Cell2Cell (Teradata Center for CRM, Duke University), via Kaggle: 51,047 anonymised subscribers of a US wireless carrier. Churners are over-represented in the sample."
K = lambda m: M("KPI", m)  # noqa: E731
SUBS = "Subscriber"
NP = 5


def buttons(p, key, x, y, w, h, field, literal, cols, title):
    ent, prop = field["Column"]["Expression"]["SourceRef"]["Entity"], field["Column"]["Property"]
    return p.add(key, "advancedSlicerVisual", x, y, w, h, query={"Values": [proj(field)]},
                 objects={"layout": props(columnCount=i(cols), rowCount=i(1)),
                          "value": idprops(fontSize=n(12), fontFamily=s(S.font_b), fontColor=color(S.ink)),
                          "fillCustom": [{"properties": {"show": b(True), "fillColor": color(S.panel)}, "selector": {"id": "default"}},
                                         {"properties": {"show": b(True), "fillColor": color(S.sel_fill)}, "selector": {"id": "selected"}}],
                          "outline": [{"properties": {"show": b(True), "lineColor": color(S.line), "weight": n(1)}, "selector": {"id": "default"}},
                                      {"properties": {"show": b(True), "lineColor": color(S.primary), "weight": n(2)}, "selector": {"id": "selected"}}],
                          "shapeCustomRectangle": props(tileShape=s("rectangleRounded"), rectangleRoundedCurve=i(8)),
                          "selection": props(singleSelect=b(True)),
                          "general": [{"properties": {"filter": {"filter": {"Version": 2, "From": [{"Name": "f", "Entity": ent, "Type": 0}],
                              "Where": [{"Condition": {"In": {"Expressions": [{"Column": {"Expression": {"SourceRef": {"Source": "f"}}, "Property": prop}}],
                                                               "Values": [[{"Literal": {"Value": literal}}]]}}}]}}}}]},
                 vco={"title": props(show=b(True), text=s(title), fontSize=n(10), fontColor=color(S.muted), fontFamily=s(S.font)),
                      "background": props(show=b(False)), "border": props(show=b(False)), "dropShadow": props(show=b(False)),
                      "padding": props(top=n(0), bottom=n(0), left=n(0), right=n(0))}, alt=title)


VALID = {"TenureBand": ["Under 12 months", "12-17 months", "18-23 months", "24-35 months", "36+ months"],
         "HandsetAgeBand": ["Under 6 months", "6-12 months", "12-18 months", "18-24 months", "24+ months"],
         "ARPUBand": ["Under $30", "$30-45", "$45-60", "$60-80", "$80+"], "CareBand": ["None", "Up to 2 a month", "2-5 a month", "5+ a month"],
         "DropBand": ["Under 2 a month", "2-5", "5-10", "10-20", "20+"]}


def churn_bar(p, key, x, y, w, h, field, label, title, sub, horizontal=False, order=None):
    flt = [D.in_filter(SUBS, field, VALID[field], key)] if field in VALID else None
    return bar(p, key, x, y, w, h, (C(SUBS, field), label), (K("Churn Rate"), "Churn rate"), title, sub, horizontal=horizontal,
               sort_desc=order is None, sort_cat=C(SUBS, order) if order else None, gradient=(S.light, S.primary), filters=flt)


def build():
    r = Report(os.path.join(ROOT, f"{NAME}.Report"), NAME, f"{NAME}.SemanticModel", theme())
    r.n_pages = NP
    x0, cw, by, bh = S.x0, S.cw, S.body_y, S.body_h
    w3 = (cw - 2 * GUT) // 3
    th = 330

    # ------------------------------------------------------------------ 1 Subscriber base
    p = r.page("base", "Subscriber base", bg=S.page)
    chrome(p, TITLE, "Churn peaks at 12-17 months, when contracts end, and on handsets 18+ months old", "51,047 subscribers, ARPU $58.83  ·  churners are over-represented in this modelling sample, so read rates as relative risk", f"1 / {NP}", SRC)
    kpi_band(p, "kpi", [(K("Subscribers"), "Subscribers"), (K("ARPU"), "ARPU, monthly"), (K("Minutes per Subscriber"), "Minutes per subscriber"),
                        (K("Churn Rate"), "Churn rate (sample)"), (K("Revenue at Risk"), "Monthly revenue at risk"), (K("Avg Tenure"), "Average tenure, months")])
    churn_bar(p, "ten", x0, by, w3, th, "TenureBand", "Tenure", "Churn by tenure", "Share of each tenure band that churned", order="TenureBandOrder")
    churn_bar(p, "hs", x0 + w3 + GUT, by, w3, th, "HandsetAgeBand", "Handset age", "Churn by handset age", "Old handsets are the strongest single signal", order="HandsetAgeBandOrder")
    churn_bar(p, "arpu", x0 + 2 * (w3 + GUT), by, cw - 2 * (w3 + GUT), th, "ARPUBand", "Monthly revenue", "Churn by ARPU band", "Monthly revenue per subscriber", order="ARPUBandOrder")
    mw = 1150
    matrix(p, "mkt", x0, by + th + GUT, mw, bh - th - GUT, [(C(SUBS, "Market"), "Market")],
           [(K("Subscribers"), "Subscribers"), (K("ARPU"), "ARPU"), (K("Churn Rate"), "Churn rate"), (K("Churn Index"), "Churn index"), (K("Revenue at Risk"), "Revenue at risk")],
           "Market scorecard", "Service-area codes  ·  churn index: 100 = base average  ·  data bars: revenue at risk",
           gradients=[(K("Churn Index"), S.soft_pos, S.soft_neg, S.panel, 100)], databars=[K("Revenue at Risk")], sort=[(K("Subscribers"), "Descending")], font=11)
    churn_bar(p, "trend", x0 + mw + GUT, by + th + GUT, cw - mw - GUT, bh - th - GUT, "UsageTrend", "Usage trend", "Falling usage comes before churn",
              "Churn by change in minutes of use", horizontal=True, order="UsageTrendOrder")

    # ------------------------------------------------------------------ 2 Network & care experience
    p = r.page("experience", "Network & care", bg=S.page)
    chrome(p, TITLE, "An old handset raises churn by half; a retention call is a symptom, not a cure", "Handset 18+ months: 34% churn vs 23% under 6 months  ·  heavy network and care users churn less: those counts track usage, not pain", f"2 / {NP}", SRC)
    kpi_band(p, "kpi", [(K("Avg Dropped Blocked"), "Dropped or blocked calls / month"), (K("Avg Care Calls"), "Care calls / month"),
                        (K("Churn Heavy Care"), "Churn, 5+ care calls"), (K("Churn Old Handset"), "Churn, handset 18+ months"),
                        (K("Churn New Handset"), "Churn, handset under 6 months"), (K("Churn Retention Callers"), "Churn, called retention")])
    churn_bar(p, "drop", x0, by, w3, th, "DropBand", "Dropped or blocked calls", "Churn by dropped and blocked calls", "Calls per month", order="DropBandOrder")
    churn_bar(p, "care", x0 + w3 + GUT, by, w3, th, "CareBand", "Care calls", "Churn by customer-care contact", "Calls to customer care per month", order="CareBandOrder")
    churn_bar(p, "credit", x0 + 2 * (w3 + GUT), by, cw - 2 * (w3 + GUT), th, "CreditClass", "Credit rating", "Churn by credit rating", "Higher-risk credit classes churn more",
              horizontal=True)
    mw = 1150
    matrix(p, "heat", x0, by + th + GUT, mw, bh - th - GUT, [(C(SUBS, "HandsetAgeBand"), "Handset age")], [(K("Churn Rate"), "Churn rate")],
           "Handset age x tenure: where the upgrade offer pays", "Churn rate  ·  darker = higher  ·  long-tenure customers on old handsets are the prime upgrade target",
           columns=[(C(SUBS, "TenureBand"), "Tenure")], gradients=[(K("Churn Rate"), S.panel, S.primary)], font=11,
           filters=[D.in_filter(SUBS, "HandsetAgeBand", VALID["HandsetAgeBand"], "heat-h"), D.in_filter(SUBS, "TenureBand", VALID["TenureBand"], "heat-t")])
    churn_bar(p, "ret", x0 + mw + GUT, by + th + GUT, cw - mw - GUT, bh - th - GUT, "RetentionContact", "Retention contact", "Retention-team callers",
              "Association, not effect: callers were already at risk", horizontal=True)

    # ------------------------------------------------------------------ 3 Churn model (ML)
    p = r.page("model", "Churn model", bg=S.page)
    chrome(p, TITLE, "Ranking by revenue at risk finds 58% of lost revenue in the top 30%, vs 44% by risk", "Gradient boosting, hold-out AUC 0.67 vs 0.60 for logistic regression  ·  the top decile churns at 54%, 1.9x the base", f"3 / {NP}", SRC)
    kpi_band(p, "kpi", [(K("Hold-out AUC"), "Hold-out AUC"), (K("Logistic AUC"), "Logistic benchmark"), (K("Top Decile Churn"), "Churn, top risk decile"),
                        (K("Top Decile Lift"), "Top-decile lift"), (K("Top 30 Churner Capture"), "Churners in top 30%"),
                        (K("Revenue Capture by Value"), "Churned revenue in top 30% (value)")])
    L = 1150
    R = cw - L - GUT
    combo(p, "dec", x0, by, L, th, (C(SUBS, "RiskDecile"), "Risk decile (1 = highest)"), [(K("Test Subscribers"), "Subscribers")], [(K("Test Churn Rate"), "Churn rate")],
          "Churn by predicted risk decile", "Hold-out subscribers  ·  columns: subscribers  ·  line: actual churn rate", sort_cat=C(SUBS, "RiskDecile"),
          col_colors=("#BFDDE1",), line_colors=(S.accent,))
    bar(p, "drv", x0 + L + GUT, by, R, bh, (C("Churn Drivers", "Driver"), "Driver"), (K("Driver Importance"), "AUC drop"), "What drives the score",
        "Permutation importance on hold-out data  ·  no age, marital status, occupation or household data")
    hw = (L - GUT) // 2
    p.add("calib", "clusteredColumnChart", x0, by + th + GUT, hw, bh - th - GUT,
          query={"Category": [P((C(SUBS, "RiskDecile"), "Risk decile"))], "Y": [P((K("Test Churn Rate"), "Actual")), P((K("Test Predicted Churn"), "Predicted"))]},
          objects={"dataPoint": [fill_for(K("Test Churn Rate"), S.primary), fill_for(K("Test Predicted Churn"), S.accent)],
                   "legend": props(show=b(True), position=s("Top")), "labels": props(show=b(False)),
                   "valueAxis": props(show=b(True), gridlineShow=b(True), labelColor=color(S.muted))},
          vco=_title("Calibration by decile", "Predicted vs actual churn, one axis"), sort=[(C(SUBS, "RiskDecile"), "Ascending")], alt="Calibration by decile")
    vx, vw, vy, vh = x0 + hw + GUT, L - hw - GUT, by + th + GUT, bh - th - GUT
    D.panel(p, "valrisk-bg", vx, vy, vw, vh)
    D.textbox(p, "valrisk-h", vx + 20, vy + 16, vw - 40, 56, [[("Target value, not just risk", 17, S.ink, True)],
                                                            [("Share of churned revenue found in the top 30% of the hold-out base", 13, S.muted, False)]])
    p.add("valrisk", "cardVisual", vx + 8, vy + 80, vw - 16, 150,
          query={"Data": [P((K("Revenue Capture by Value"), "Ranked by revenue at risk")), P((K("Revenue Capture by Risk"), "Ranked by churn risk"))]},
          objects={"value": idprops(fontSize=n(40), fontColor=color(S.primary), fontFamily=s(S.font_num), horizontalAlignment=s("left"), labelDisplayUnits=s("0")),
                   "label": idprops(show=b(True), fontSize=n(12), fontColor=color(S.muted), position=s("aboveValue"), horizontalAlignment=s("left")),
                   "outline": idprops(show=b(False)), "fillCustom": [{"properties": {"show": b(False)}}],
                   "layout": idprops(style=s("Table"), customizeLines=b(True), gridlineWidth=n(1), gridlineColor=color(S.line), gridlineStyle=s("solid")),
                   "padding": idprops(leftMargin=i(14), rightMargin=i(8), topMargin=i(0), bottomMargin=i(0))},
          vco=dict(D.NOBOX), alt="Revenue captured by ranking method")
    D.textbox(p, "valrisk-note", vx + 20, vy + 240, vw - 40, vh - 256,
              [[("Same budget, same model: ordering by churn probability x ARPU reaches 14 more points of the revenue that walks out "
                 "the door. High-value customers who might leave beat low-value customers who surely will.", 15, S.ink, False)]])

    # ------------------------------------------------------------------ 4 Retention economics (what-if)
    p = r.page("economics", "Retention economics", bg=S.page)
    chrome(p, TITLE, "Value-ranked offers return $234K net, 55% more than ranking by churn risk", "Defaults: top 30% offered a $25 deal, 20% of would-be churners saved, 12 months of revenue  ·  change any assumption above", f"4 / {NP}", SRC)
    kpi_band(p, "kpi", [(K("Targeted"), "Subscribers offered"), (K("Churners Reached"), "Would-be churners reached"), (K("Saves"), "Saves"),
                        (K("Revenue Retained"), "Revenue retained, 12 months"), (K("Offer Spend"), "Offer spend"), (K("Net Value"), "Net value")])
    D.panel(p, "ctrl", x0, by, cw, 92)
    buttons(p, "rank", x0 + 20, by + 8, 380, 76, C("Ranking", "Ranking"), "'By revenue at risk'", 2, "Rank subscribers")
    buttons(p, "share", x0 + 424, by + 8, 800, 76, C("Target Share", "Target Share"), "30L", 10, "Share of base offered a deal")
    buttons(p, "cost", x0 + 1248, by + 8, 290, 76, C("Offer Cost", "Offer Cost"), "25L", 4, "Cost per offer")
    buttons(p, "save", x0 + 1562, by + 8, cw - 1582, 76, C("Save Rate", "Save Rate"), "0.2D", 4, "Save rate")
    cy, ch = by + 92 + GUT, bh - 92 - GUT
    L = 1150
    line(p, "curve", x0, cy, L, ch, (C("Share Axis", "Share Axis"), "Share of base offered a deal, %"),
         [(K("Net by Value Ranking"), "Rank by revenue at risk"), (K("Net by Risk Ranking"), "Rank by churn risk")],
         "Net value of the programme by share targeted", "Backtest on 15,315 hold-out subscribers  ·  12 months of retained revenue minus offer spend",
         colors=(S.primary, S.grey), sort_cat=C("Share Axis", "Share Axis"), markers=True)
    D.panel(p, "best-bg", x0 + L + GUT, cy, cw - L - GUT, 250)
    p.add("best", "cardVisual", x0 + L + GUT + 8, cy + 14, cw - L - GUT - 16, 222,
          query={"Data": [P((K("Best Share"), "Best share at these assumptions")), P((K("Retention ROI"), "Return on offer spend"))]},
          objects={"value": idprops(fontSize=n(34), fontColor=color(S.primary), fontFamily=s(S.font_num), horizontalAlignment=s("left"), labelDisplayUnits=s("0")),
                   "label": idprops(show=b(True), fontSize=n(12), fontColor=color(S.muted), position=s("aboveValue"), horizontalAlignment=s("left")),
                   "outline": idprops(show=b(False)), "fillCustom": [{"properties": {"show": b(False)}}],
                   "layout": idprops(style=s("Table"), customizeLines=b(True), gridlineWidth=n(1), gridlineColor=color(S.line), gridlineStyle=s("solid")),
                   "padding": idprops(leftMargin=i(16), rightMargin=i(8), topMargin=i(0), bottomMargin=i(0))},
          vco=dict(D.NOBOX), alt="Best share and ROI")
    bar(p, "vdec", x0 + L + GUT, cy + 250 + GUT, cw - L - GUT, ch - 250 - GUT, (C(SUBS, "ValueDecile"), "Revenue-at-risk decile"), (K("Churned Revenue"), "Churned revenue"),
        "Where the churned revenue sits", "Hold-out monthly revenue lost, by revenue-at-risk decile", horizontal=False, sort_desc=False,
        sort_cat=C(SUBS, "ValueDecile"), filters=[D.in_filter(SUBS, "Sample", ["Test"], "vdec")])

    # ------------------------------------------------------------------ 5 About
    p = r.page("about", "About & method", bg=S.page)
    chrome(p, TITLE, "How this report is built", "Questions, data, model and definitions behind every number", f"5 / {NP}", SRC)
    about_page(p, [
        ("Business questions", ["Who churns, and how early can we see it?", "Which network and care experiences push customers out?",
                                "Who should get a retention offer, and how many offers pay back?"]),
        ("Data & ML", ["Cell2Cell: 51,047 subscribers, 58 fields; churners are over-represented (modelling sample), so rates are relative.",
                       "Gradient boosting on usage, network, care, handset, tenure, plan and credit; no age, marital status or occupation.",
                       "Hold-out AUC 0.67 vs 0.60 for logistic regression, in line with published results on this dataset."]),
        ("Model & DAX", ["One subscriber fact with banded attributes; what-if tables for share targeted, offer cost, save rate and ranking.",
                         "Simulator backtested on the 30% hold-out using actual churn outcomes.",
                         "Retention-team contact shown as association only: no randomised retention test exists in the data."]),
    ], [("ARPU", "average monthly revenue per subscriber."), ("Revenue at risk", "churn probability x monthly revenue."),
        ("Lift", "churn rate in a decile / churn rate of the base."), ("Save rate", "share of would-be churners an offer keeps (assumption)."),
        ("Net value", "12 months of retained revenue minus the cost of every offer made.")],
        "Built by Mohamed Ayman, Senior Data Scientist  ·  github.com/ayman-stat  ·  mohamedaymanstat.vercel.app")

    r.write()
    write_pbip(os.path.join(ROOT, f"{NAME}.pbip"), f"{NAME}.Report")


if __name__ == "__main__":
    build()
    print("report written")
