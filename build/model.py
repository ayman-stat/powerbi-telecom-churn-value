"""Semantic model for the Telecom Churn & Subscriber Value report (writes TMDL)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "toolkit"))  # vendored copy of the shared design toolkit
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_tools"))  # master copy, when present
from pbip import col, meas, m_table, calc_table, measure_table, write_model, csv_expressions  # noqa: E402

REPO = "powerbi-telecom-churn-value"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NAME = "TelecomChurnValue"
SUB = "'Subscriber'"


def banded(name, desc=None):
    return [col(name, sort=name + "Order", desc=desc), col(name + "Order", "int64", "0", hidden=True)]


TABLES = [
    m_table("Subscriber", "fact_subscriber.csv", [
        col("SubscriberKey", "int64", "0", hidden=True, key=True), col("Churned", "int64", "0", hidden=True),
        col("MonthlyRevenue", "double", "$#,0.00", hidden=True), col("MonthlyMinutes", "double", "#,0", hidden=True), col("OverageMinutes", "double", "#,0", hidden=True),
        col("CustomerCareCalls", "double", "0.0", hidden=True), col("DroppedBlockedCalls", "double", "0.0", hidden=True), col("MonthsInService", "int64", "0", hidden=True),
        col("CurrentEquipmentDays", "double", "#,0", hidden=True), col("HandsetPrice", "double", "$#,0", hidden=True),
        *banded("TenureBand", "Months in service"), *banded("HandsetAgeBand", "Age of the current handset"), *banded("ARPUBand", "Monthly revenue"),
        *banded("CareBand", "Customer-care calls per month"), *banded("DropBand", "Dropped or blocked calls per month"), *banded("UsageTrend", "Change in minutes of use"),
        col("Market", desc="Service-area code (first three letters)"), col("RetentionContact"), col("CreditClass", desc="Credit rating"),
        col("ChurnScore", "double", "0.0%", hidden=True), col("Sample", desc="Train (70%) or hold-out Test (30%)"),
        col("RiskDecile", "double", "0", desc="Hold-out only: 1 = highest churn risk"), col("ValueDecile", "double", "0", desc="Hold-out only: 1 = highest revenue at risk")],
        desc="Fact and dimension in one: 51,047 subscribers of a US wireless carrier (Cell2Cell, anonymised)."),
    m_table("Churn Drivers", "churn_drivers.csv", [col("Driver"), col("AUCDrop", "double", "0.000", hidden=True)]),
    m_table("Model Metrics", "model_metrics.csv", [col("Metric"), col("Value", "double", "0.####", hidden=True)], hidden=True),
    calc_table("Target Share", "GENERATESERIES(10, 100, 10)", [col("Target Share", "int64", "0\"%\"", src="Value", desc="Share of the hold-out base offered a retention deal")]),
    calc_table("Share Axis", "GENERATESERIES(10, 100, 10)", [col("Share Axis", "int64", "0", src="Value", desc="Share of the base targeted, % (chart axis)")]),
    calc_table("Offer Cost", 'DATATABLE("Offer Cost", INTEGER, {{10}, {25}, {50}, {100}})', [col("Offer Cost", "int64", "$0", desc="Cost of one retention offer, USD")]),
    calc_table("Save Rate", 'DATATABLE("Save Rate", DOUBLE, {{0.1}, {0.2}, {0.3}, {0.4}})', [col("Save Rate", "double", "0%", desc="Share of would-be churners an offer keeps")]),
    calc_table("Ranking", 'DATATABLE("Ranking", STRING, "Ranking Order", INTEGER, {{"By revenue at risk", 1}, {"By churn risk", 2}})',
               [col("Ranking", sort="Ranking Order", desc="How subscribers are ordered for targeting"), col("Ranking Order", "int64", "0", hidden=True)]),
]

FB, FE, FM, FR, FX = "1 Base", "2 Experience", "3 Model", "4 Retention economics", "9 Model metrics"


def metric(name, key, fmt, desc=None):
    return meas(name, f'LOOKUPVALUE(\'Model Metrics\'[Value], \'Model Metrics\'[Metric], "{key}")', fmt, FX, desc)


def sim(name, body, fmt, desc=None, axis=False):
    """Simulator measure on hold-out subscribers. body uses _in (targeted table)."""
    share = "SELECTEDVALUE('Share Axis'[Share Axis])" if axis else "[Share Value]"
    return f'''VAR _k = {share} / 10
VAR _byValue = {'"By revenue at risk"' if axis == "value" else ('"By churn risk"' if axis == "risk" else "[Ranking Value]")} = "By revenue at risk"
VAR _in = FILTER({SUB}, {SUB}[Sample] = "Test" && IF(_byValue, {SUB}[ValueDecile], {SUB}[RiskDecile]) <= _k)
RETURN {body}'''


RET = "SUMX(FILTER(_in, 'Subscriber'[Churned] = 1), 'Subscriber'[MonthlyRevenue]) * [Save Rate Value] * 12"
NET = f"{RET} - COUNTROWS(_in) * [Offer Cost Value]"
MEASURES = [
    meas("Subscribers", f"COUNTROWS({SUB})", "#,0", FB),
    meas("Churn Rate", f"AVERAGE({SUB}[Churned])", "0.0%", FB, "Share of subscribers who churned. The source over-represents churners: read as relative risk."),
    meas("ARPU", f"AVERAGE({SUB}[MonthlyRevenue])", "$0.00", FB, "Average monthly revenue per subscriber."),
    meas("Monthly Revenue", f"SUM({SUB}[MonthlyRevenue])", "$#,0", FB),
    meas("Minutes per Subscriber", f"AVERAGE({SUB}[MonthlyMinutes])", "#,0", FB),
    meas("Avg Tenure", f"AVERAGE({SUB}[MonthsInService])", "0.0", FB, "Months in service."),
    meas("Revenue at Risk", f"SUMX({SUB}, {SUB}[ChurnScore] * {SUB}[MonthlyRevenue])", "$#,0", FB, "Churn probability x monthly revenue: the monthly revenue in play."),
    meas("Churned Revenue", f"CALCULATE(SUM({SUB}[MonthlyRevenue]), {SUB}[Churned] = 1)", "$#,0", FB, "Monthly revenue of subscribers who left."),
    meas("Churn Index", f"DIVIDE([Churn Rate], CALCULATE([Churn Rate], ALL({SUB}))) * 100", "0", FB, "Churn rate vs the whole base (100 = average)."),
    meas("Avg Dropped Blocked", f"AVERAGE({SUB}[DroppedBlockedCalls])", "0.0", FE, "Dropped or blocked calls per subscriber per month."),
    meas("Avg Care Calls", f"AVERAGE({SUB}[CustomerCareCalls])", "0.0", FE),
    meas("Avg Overage", f"AVERAGE({SUB}[OverageMinutes])", "#,0", FE),
    meas("Churn Heavy Care", f'CALCULATE([Churn Rate], {SUB}[CareBand] = "5+ a month")', "0.0%", FE),
    meas("Churn Old Handset", f'CALCULATE([Churn Rate], {SUB}[HandsetAgeBand] IN {{"18-24 months", "24+ months"}})', "0.0%", FE),
    meas("Churn New Handset", f'CALCULATE([Churn Rate], {SUB}[HandsetAgeBand] = "Under 6 months")', "0.0%", FE),
    meas("Churn Retention Callers", f'CALCULATE([Churn Rate], {SUB}[RetentionContact] = "Called retention team")', "0.0%", FE,
         "Association only: subscribers who call the retention team are already at risk."),
    meas("Test Subscribers", f'CALCULATE([Subscribers], {SUB}[Sample] = "Test")', "#,0", FM),
    meas("Test Churn Rate", f'CALCULATE([Churn Rate], {SUB}[Sample] = "Test")', "0.0%", FM),
    meas("Test Predicted Churn", f'CALCULATE(AVERAGE({SUB}[ChurnScore]), {SUB}[Sample] = "Test")', "0.0%", FM),
    meas("Driver Importance", "SUM('Churn Drivers'[AUCDrop])", "0.000", FM),
    metric("Hold-out AUC", "churn_auc_test", "0.00"), metric("Logistic AUC", "churn_auc_logit", "0.00"),
    metric("Top Decile Churn", "churn_top_decile_rate", "0%"), metric("Top Decile Lift", "churn_top_decile_lift", "0.0\"x\""),
    metric("Top 30 Churner Capture", "churn_top_30_capture", "0%", "Share of hold-out churners in the riskiest 30%."),
    metric("Revenue Capture by Value", "revenue_top_30_capture_by_value", "0%", "Share of churned revenue in the top 30% ranked by revenue at risk."),
    metric("Revenue Capture by Risk", "revenue_top_30_capture_by_risk", "0%", "Share of churned revenue in the top 30% ranked by churn risk."),
    meas("Share Value", "SELECTEDVALUE('Target Share'[Target Share], 30)", "0\"%\"", FR),
    meas("Offer Cost Value", "SELECTEDVALUE('Offer Cost'[Offer Cost], 25)", "$0", FR),
    meas("Save Rate Value", "SELECTEDVALUE('Save Rate'[Save Rate], 0.2)", "0%", FR),
    meas("Ranking Value", "SELECTEDVALUE('Ranking'[Ranking], \"By revenue at risk\")", None, FR),
    meas("Targeted", sim("Targeted", "COUNTROWS(_in)", "#,0"), "#,0", FR, "Hold-out subscribers offered a deal."),
    meas("Churners Reached", sim("", "SUMX(_in, 'Subscriber'[Churned])", "#,0"), "#,0", FR, "Targeted subscribers who went on to churn (backtest)."),
    meas("Saves", "[Churners Reached] * [Save Rate Value]", "#,0", FR),
    meas("Revenue Retained", sim("", RET, ""), "$#,0", FR, "12 months of revenue from saved churners."),
    meas("Offer Spend", "[Targeted] * [Offer Cost Value]", "$#,0", FR),
    meas("Net Value", "[Revenue Retained] - [Offer Spend]", "$#,0;-$#,0", FR, "Revenue retained minus offer spend, hold-out base."),
    meas("Retention ROI", "DIVIDE([Net Value], [Offer Spend])", "0%", FR),
    meas("Net by Value Ranking", sim("", NET, "", axis="value"), "$#,0;-$#,0", FR),
    meas("Net by Risk Ranking", sim("", NET, "", axis="risk"), "$#,0;-$#,0", FR),
    meas("Best Share", '''VAR _t = ADDCOLUMNS(ALL('Share Axis'[Share Axis]), "@n", IF([Ranking Value] = "By revenue at risk", [Net by Value Ranking], [Net by Risk Ranking]))
RETURN MAXX(TOPN(1, _t, [@n], DESC), 'Share Axis'[Share Axis])''', "0\"%\"", FR, "Target share with the highest net value under the chosen assumptions."),
]


def build(local=True):
    tables = TABLES + [measure_table("KPI", MEASURES, "All report measures, organised in display folders.")]
    exprs = csv_expressions(REPO, os.path.join(ROOT, "data") + "\\", default="Local" if local else "GitHub")
    write_model(os.path.join(ROOT, f"{NAME}.SemanticModel"), NAME, tables, [], exprs,
                desc="Telecom Churn & Subscriber Value: 51,047 subscribers of a US wireless carrier (Cell2Cell), churn model scores and a retention simulator.")


if __name__ == "__main__":
    build(local="--github" not in sys.argv)
    print("model written")
