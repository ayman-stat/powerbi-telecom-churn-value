"""Design systems for the portfolio reports (1920 x 1080 canvas).

Every report gets its own identity: a `Style` sets palette, typography, navigation pattern, KPI treatment,
corner radius and surfaces. Call `use(STYLE)` once at the top of a report script; every helper below reads
the active style at call time. Shared conventions (action titles, source footnote, page number, readable
field names, right-aligned numbers) stay the same across identities.

Identities
    CONSULTING    navy + teal, navy top bar, one KPI band with dividers        (Olist marketplace)
    PRIVATE_BANK  deep green + brass on ivory, Georgia titles, left rail nav    (Bank campaign)
    COMMAND       dark mode, violet treatment vs slate control, pill tabs       (CVM experiments)
    RETAIL        warm white, orange + charcoal, underline tabs, soft shadows   (Retail network)
    EDITORIAL     paper + forest green, amber/teal energy palette, thin rules   (Energy transition)
"""
from dataclasses import dataclass, field, asdict
from pbip import C, M, b, color, i, idprops, lit, n, proj, props, qref, s, hexid

W, H, MG, GUT = 1920, 1080, 32, 24

# Legacy constants (consulting palette) kept for the Olist report script.
NAVY, TEAL, GREY, INK, MUTED, FAINT = "#0B2545", "#13A89E", "#9AA9BC", "#0B2545", "#55657D", "#8593A8"
LINE, PAGE, WHITE, RED, GREEN, AMBER, SKY = "#E1E7EF", "#F4F6F9", "#FFFFFF", "#C8102E", "#13865A", "#D98324", "#4A6FA5"
LIGHT, SOFT_RED, SOFT_TEAL = "#C5D2E3", "#F2C4C9", "#BFE6E1"
FONT, FONT_B = "Segoe UI", "Segoe UI Semibold"
BODY_Y, FOOT_Y = 344, 1040


@dataclass
class Style:
    key: str
    name: str
    primary: str          # the data in focus
    accent: str           # one highlight
    grey: str             # context series
    light: str            # low end of value gradients
    ink: str
    muted: str
    faint: str
    page: str
    panel: str
    line: str
    grid: str
    head: str             # table header fill
    band: str             # table banding
    neg: str
    pos: str
    warn: str
    soft_neg: str
    soft_pos: str
    sel_fill: str         # selected slicer button
    data_colors: list
    header_bg: str
    header_ink: str
    header_muted: str
    nav: str = "topbar"   # topbar | rail | pills | underline
    kpi: str = "band"     # band | tiles
    font: str = "Segoe UI"
    font_b: str = "Segoe UI Semibold"
    font_title: str = "Segoe UI Semibold"
    font_num: str = "Segoe UI Semibold"
    title_size: int = 30
    radius: int = 6
    border: bool = True
    shadow: bool = False
    dark: bool = False
    map_style: str = "grayscale_light"
    accent_bar: str = "Left"   # KPI tile accent position
    # layout (filled by use())
    x0: int = MG
    cw: int = W - 2 * MG
    title_y: int = 90
    kpi_y: int = 192
    kpi_h: int = 128
    body_y: int = 344
    foot_y: int = 1040
    slicer_x: int = 1408       # where the title band's slicer zone starts
    bx: int = MG               # body area (differs from x0/cw when KPIs sit in a left column)
    bw: int = W - 2 * MG
    kpi_w: int = 300

    @property
    def body_h(self):
        return self.foot_y - 16 - self.body_y


CONSULTING = Style("consulting", "MAConsulting", NAVY, TEAL, GREY, LIGHT, INK, MUTED, FAINT, PAGE, WHITE, LINE, "#EDF1F6",
                   "#EEF2F7", "#FAFBFD", RED, GREEN, AMBER, SOFT_RED, SOFT_TEAL, "#DCE6F2",
                   [NAVY, TEAL, GREY, AMBER, SKY, "#B8C4D4", "#1B7F79", RED, "#6C5B9A", "#2E86AB"],
                   NAVY, WHITE, "#AFC1D6", nav="topbar", kpi="band")

PRIVATE_BANK = Style("bank", "MAPrivateBank", "#0F4C45", "#B8894B", "#AEB8B5", "#D3E2DE", "#1C2826", "#5C6967", "#8E9997",
                     "#F5F2EC", "#FFFFFF", "#E7E0D4", "#EFEBE4", "#F1ECE3", "#FBF9F5", "#A63D32", "#0F4C45", "#B8894B",
                     "#EFD2CC", "#CFE5DE", "#DCEBE7",
                     ["#0F4C45", "#B8894B", "#AEB8B5", "#4F8A80", "#D8BE91", "#1F6B60", "#7D8A88", "#A63D32"],
                     "#0E2B28", "#F5F2EC", "#9FB5B0", nav="rail", kpi="tiles", font_title="Georgia", title_size=30,
                     radius=2, border=False, accent_bar="Top")

COMMAND = Style("cvm", "MACommandCentre", "#8B7BFF", "#2EE6B6", "#5D6B86", "#2E2A5C", "#E8ECF4", "#9AA5BC", "#66728A",
                "#0A101E", "#111A2E", "#1E2A44", "#1A2440", "#16213A", "#131D33", "#FF6B81", "#2EE6B6", "#FFB547",
                "#4A1F2E", "#0F3B35", "#2E2A5C",
                ["#8B7BFF", "#2EE6B6", "#5D6B86", "#FFB547", "#FF6B81", "#4CC9F0", "#B4A9FF", "#7FE7CF"],
                "#0A101E", "#E8ECF4", "#9AA5BC", nav="pills", kpi="tiles", font_num="Segoe UI Light", radius=12, border=True,
                dark=True, map_style="grayscale_dark", accent_bar="Left")

RETAIL = Style("retail", "MARetail", "#E8590C", "#1F2937", "#CFCAC4", "#FBD9C4", "#1F2328", "#6A6F78", "#9BA0A8",
               "#F6F5F2", "#FFFFFF", "#ECE9E4", "#F1EEE9", "#FAF6F1", "#FCFBF9", "#D64545", "#2F9E6E", "#F59F00",
               "#F8D4D4", "#D2EEDF", "#FDE3D3",
               ["#E8590C", "#1F2937", "#CFCAC4", "#F59F00", "#2F9E6E", "#7A8290", "#FFA94D", "#495057"],
               "#FFFFFF", "#1F2328", "#6A6F78", nav="underline", kpi="tiles", font_num="Trebuchet MS", radius=14, border=False,
               shadow=True, accent_bar="Bottom")

EDITORIAL = Style("energy", "MAEditorial", "#1B4332", "#E9A23B", "#B7BDB4", "#D8E6DC", "#1B2621", "#58645D", "#8C968F",
                  "#F4F2EA", "#FFFFFF", "#E2DED2", "#ECE9DF", "#EFEDE4", "#FAF9F4", "#B5523B", "#2D6A4F", "#E9A23B",
                  "#F1D5CC", "#D3E8DA", "#E3EEE6",
                  ["#1B4332", "#E9A23B", "#2A9D8F", "#B7BDB4", "#E76F51", "#52796F", "#F4D58D", "#264653"],
                  "#1B4332", "#F4F2EA", "#A9C2B5", nav="topbar", kpi="band", font_title="Segoe UI Light", title_size=34,
                  radius=0, border=True)

CARD = Style("card", "MACardRisk", "#2F5BEA", "#F0533A", "#B9C0CC", "#D3DDFB", "#151A23", "#5B6474", "#8F97A6",
             "#F3F4F7", "#FFFFFF", "#E3E6EC", "#ECEEF2", "#F1F3F7", "#FAFBFC", "#E5484D", "#1F9D6B", "#F5A524",
             "#FBD5D6", "#CDEFE0", "#DCE5FD",
             ["#2F5BEA", "#F0533A", "#B9C0CC", "#F5A524", "#13A3A3", "#4B3FB8", "#8AA4F5", "#151A23"],
             "#151A23", "#FFFFFF", "#A6AEBD", nav="segmented", kpi="column", radius=10, border=True, accent_bar="Left")

TELCO = Style("telco", "MATelco", "#0A7C8C", "#FFB020", "#A9B8BE", "#CBE7EA", "#0E2328", "#55707A", "#8AA0A8",
              "#F2F5F6", "#FFFFFF", "#DDE6E9", "#E9EFF1", "#EEF4F5", "#F9FBFB", "#E04E39", "#2E9D63", "#FFB020",
              "#F8D3CD", "#CFEBDD", "#D9EEF0",
              ["#0A7C8C", "#FFB020", "#A9B8BE", "#E04E39", "#073B45", "#5FB7C1", "#2E9D63", "#7A8C92"],
              "#073B45", "#FFFFFF", "#9CC3C9", nav="hero", kpi="hero", radius=12, border=True, title_size=32)

S = Style(**asdict(CONSULTING))


def use(style):
    """Activate a style (mutates the shared S so `from design import S` stays valid) and derive layout."""
    for k, v in asdict(style).items():
        setattr(S, k, v)
    if S.nav == "rail":
        rail = 256
        S.x0, S.cw, S.title_y, S.kpi_y, S.kpi_h = rail + MG, W - rail - 2 * MG, 36, 150, 120
    elif S.nav == "pills":
        S.x0, S.cw, S.title_y, S.kpi_y, S.kpi_h = MG, W - 2 * MG, 92, 196, 124
    elif S.nav == "underline":
        S.x0, S.cw, S.title_y, S.kpi_y, S.kpi_h = MG, W - 2 * MG, 100, 200, 124
    elif S.nav == "segmented":
        S.x0, S.cw, S.title_y, S.kpi_y, S.kpi_h = MG, W - 2 * MG, 86, 188, 0
    elif S.nav == "hero":
        S.x0, S.cw, S.title_y, S.kpi_y, S.kpi_h = MG, W - 2 * MG, 76, 172, 108
    else:
        S.x0, S.cw, S.title_y, S.kpi_y, S.kpi_h = MG, W - 2 * MG, 90, 192, 128
    S.foot_y = 1040
    S.slicer_x = S.x0 + S.cw - 480
    S.bx, S.bw = S.x0, S.cw
    if S.kpi == "column":
        S.body_y = S.kpi_y
        S.bx, S.bw = S.x0 + S.kpi_w + GUT, S.cw - S.kpi_w - GUT
    elif S.nav == "hero":
        S.body_y = 304 + GUT
    else:
        S.body_y = S.kpi_y + S.kpi_h + GUT
    return S


def theme(name=None):
    ink, muted = S.ink, S.muted
    axis = {"show": True, "fontSize": 10, "labelColor": {"solid": {"color": muted}}, "showAxisTitle": False, "gridlineShow": False}
    vaxis = {"show": False, "gridlineShow": False, "showAxisTitle": False}
    legend = {"show": True, "position": "Top", "fontSize": 10, "labelColor": {"solid": {"color": muted}}}
    bars = {"labels": [{"show": True, "fontSize": 10, "color": {"solid": {"color": ink}}}], "valueAxis": [vaxis],
            "categoryAxis": [dict(axis, innerPadding=24)], "dataPoint": [{"borderShow": False}], "legend": [{"show": False}]}
    yaxis = {"show": True, "fontSize": 10, "labelColor": {"solid": {"color": muted}}, "showAxisTitle": False,
             "gridlineShow": True, "gridlineStyle": "dotted", "gridlineColor": {"solid": {"color": S.grid}}}
    table_style = {
        "columnHeaders": [{"autoSizeColumnWidth": True, "columnAdjustment": "growToFit", "backColor": {"solid": {"color": S.head}},
                           "fontColor": {"solid": {"color": ink}}, "fontFamily": S.font_b, "fontSize": 10, "bold": False}],
        "values": [{"backColorPrimary": {"solid": {"color": S.panel}}, "backColorSecondary": {"solid": {"color": S.band}},
                    "fontColorPrimary": {"solid": {"color": ink}}, "fontColorSecondary": {"solid": {"color": ink}}, "fontSize": 10}],
        "grid": [{"gridHorizontal": True, "gridHorizontalColor": {"solid": {"color": S.grid}}, "gridVertical": False, "rowPadding": 6,
                  "outlineColor": {"solid": {"color": S.line}}}],
        "rowHeaders": [{"fontColor": {"solid": {"color": ink}}, "fontSize": 10}]}
    shadow = [{"show": True, "preset": "Bottom", "position": "Outer", "color": {"solid": {"color": "#1F2328"}}, "transparency": 93,
               "shadowBlur": 14, "shadowDistance": 2, "shadowSpread": 0}] if S.shadow else [{"show": False}]
    flat = {"padding": [{"top": 0, "bottom": 0, "left": 0, "right": 0}], "background": [{"show": False}], "border": [{"show": False}],
            "dropShadow": [{"show": False}]}
    return {
        "name": name or S.name,
        "dataColors": list(S.data_colors),
        "good": S.pos, "neutral": S.grey, "bad": S.neg, "maximum": S.primary, "center": S.light, "minimum": S.panel, "null": S.grid,
        "background": S.panel, "foreground": ink, "tableAccent": S.accent,
        "textClasses": {
            "callout": {"fontSize": 30, "fontFace": S.font_num, "color": ink},
            "title": {"fontSize": 13, "fontFace": S.font_b, "color": ink},
            "header": {"fontSize": 12, "fontFace": S.font_b, "color": ink},
            "label": {"fontSize": 10, "fontFace": S.font, "color": muted}},
        "visualStyles": {
            "*": {"*": {
                "border": [{"show": S.border, "color": {"solid": {"color": S.line}}, "radius": S.radius}],
                "background": [{"show": True, "color": {"solid": {"color": S.panel}}, "transparency": 0}],
                "dropShadow": shadow,
                "padding": [{"top": 14, "bottom": 12, "left": 18, "right": 18}],
                "title": [{"show": True, "fontSize": 13, "fontFamily": S.font_b, "fontColor": {"solid": {"color": ink}}, "titleWrap": True}],
                "subTitle": [{"fontSize": 10, "fontFamily": S.font, "fontColor": {"solid": {"color": muted}}}],
                "visualHeader": [{"show": False}],
                "categoryAxis": [{"gridlineStyle": "dotted", "gridlineColor": {"solid": {"color": S.grid}}}],
                "valueAxis": [{"gridlineStyle": "dotted", "gridlineColor": {"solid": {"color": S.grid}}}]}},
            "page": {"*": {"background": [{"color": {"solid": {"color": S.page}}, "transparency": 0}],
                           "outspacePane": [{"backgroundColor": {"solid": {"color": S.panel}}, "foregroundColor": {"solid": {"color": ink}},
                                             "border": True, "borderColor": {"solid": {"color": S.line}},
                                             "checkboxAndApplyColor": {"solid": {"color": S.primary}}, "inputBoxColor": {"solid": {"color": S.page}},
                                             "fontFamily": S.font, "titleSize": 12, "headerSize": 11}],
                           "filterCard": [{"$id": "Applied", "backgroundColor": {"solid": {"color": S.sel_fill}}, "foregroundColor": {"solid": {"color": ink}},
                                           "borderColor": {"solid": {"color": S.line}}, "inputBoxColor": {"solid": {"color": S.page}}},
                                          {"$id": "Available", "backgroundColor": {"solid": {"color": S.panel}}, "foregroundColor": {"solid": {"color": ink}},
                                           "borderColor": {"solid": {"color": S.line}}, "inputBoxColor": {"solid": {"color": S.page}}}]}},
            "textbox": {"*": flat}, "shape": {"*": dict(flat, title=[{"show": False}])},
            "actionButton": {"*": {"dropShadow": [{"show": False}]}}, "pageNavigator": {"*": {"dropShadow": [{"show": False}]}},
            "advancedSlicerVisual": {"*": {"dropShadow": [{"show": False}]}},
            "cardVisual": {"*": {"value": [{"bold": False, "$id": "default"}], "label": [{"show": True, "$id": "default"}],
                                 "cardCalloutArea": [{"paddingUniform": 0}], "title": [{"show": False}],
                                 "border": [{"show": False}], "background": [{"show": False}], "dropShadow": [{"show": False}]}},
            "barChart": {"*": bars}, "clusteredBarChart": {"*": bars},
            "columnChart": {"*": dict(bars, legend=[legend])}, "clusteredColumnChart": {"*": bars},
            "lineChart": {"*": {"lineStyles": [{"strokeWidth": 2.5, "lineChartType": "linear", "areaShow": False}],
                                "labels": [{"show": False}], "categoryAxis": [axis], "valueAxis": [yaxis], "legend": [legend]}},
            "lineClusteredColumnComboChart": {"*": {"lineStyles": [{"strokeWidth": 2.5, "lineChartType": "linear"}],
                                                    "labels": [{"show": False}], "categoryAxis": [axis],
                                                    "valueAxis": [dict(yaxis, secShow=True, secLabelColor={"solid": {"color": muted}})],
                                                    "legend": [legend]}},
            "scatterChart": {"*": {"categoryAxis": [dict(axis, showAxisTitle=True, titleColor={"solid": {"color": muted}})],
                                   "valueAxis": [dict(yaxis, showAxisTitle=True, titleColor={"solid": {"color": muted}})],
                                   "legend": [dict(legend, show=False)]}},
            "donutChart": {"*": {"legend": [dict(legend, position="Right")],
                                 "labels": [{"show": True, "color": {"solid": {"color": ink}}, "fontSize": 10}]}},
            "tableEx": {"*": {k: v for k, v in table_style.items() if k != "rowHeaders"}}, "pivotTable": {"*": table_style},
            "slicer": {"*": {"header": [{"fontFamily": S.font_b, "textSize": 10, "fontColor": {"solid": {"color": ink}}, "outlineStyle": 0}],
                             "items": [{"fontFamily": S.font, "textSize": 10, "fontColor": {"solid": {"color": ink}}, "outlineStyle": 0,
                                        "padding": 2, "background": {"solid": {"color": S.panel}}}]}},
        },
    }


# ----------------------------------------------------------------------------------------------------------- primitives
def L(v):
    """Bare literal (FillRule colour stops must not carry an expr wrapper)."""
    return {"Literal": {"Value": v}}


NOBOX = {"background": props(show=b(False)), "border": props(show=b(False)), "dropShadow": props(show=b(False)),
         "padding": props(top=n(0), bottom=n(0), left=n(0), right=n(0)), "title": props(show=b(False))}


def textbox(page, key, x, y, w, h, paragraphs, alt=None, parent=None, hidden=False):
    """paragraphs: [[(text, size_px, color, bold[, font])], ...]; a leading "left"/"right"/"center" sets alignment."""
    paras = []
    for runs in paragraphs:
        align = "left"
        if runs and isinstance(runs[0], str):
            align, runs = runs[0], runs[1:]
        out = []
        for r in runs:
            t, sz, c, bold = r[:4]
            face = r[4] if len(r) > 4 else (S.font_b if bold else S.font)
            out.append({"value": t, "textStyle": {"fontFamily": face, "fontSize": f"{sz}px", "color": c}})
        paras.append({"textRuns": out, "horizontalTextAlignment": align})
    return page.add(key, "textbox", x, y, w, h, objects={"general": props(paragraphs=paras)}, vco=dict(NOBOX), parent=parent, hidden=hidden)


def shape(page, key, x, y, w, h, fill, radius=0, line=None, parent=None, hidden=False, transparency=0, shadow=False):
    objs = {"shape": idprops(tileShape=s("rectangle")),
            "fill": idprops(fillColor=color(fill), transparency=n(transparency)),
            "outline": idprops(show=b(bool(line)), **({"lineColor": color(line), "weight": n(1)} if line else
                                                       {"lineColor": color(fill), "weight": n(0), "transparency": n(100)}))}
    if radius:
        objs["shape"] = idprops(tileShape=s("rectangleRounded"), rectangleRoundedCurve=i(radius))
    vco = dict(NOBOX)
    if shadow and S.shadow:
        vco["dropShadow"] = props(show=b(True), preset=s("Bottom"), position=s("Outer"), color=color("#1F2328"), transparency=n(92),
                                  shadowBlur=n(14), shadowDistance=n(2), shadowSpread=n(0))
    return page.add(key, "shape", x, y, w, h, objects=objs, vco=vco, parent=parent, hidden=hidden)


def panel(page, key, x, y, w, h, parent=None, hidden=False, fill=None):
    """A surface matching the style's chart cards: an empty container with background, rounded border and shadow
    (visual-container borders round reliably; the shape visual's corner settings do not)."""
    vco = {"background": props(show=b(True), color=color(fill or S.panel), transparency=n(0)),
           "border": props(show=b(S.border), color=color(S.line), radius=n(S.radius), width=n(1)),
           "padding": props(top=n(0), bottom=n(0), left=n(0), right=n(0)), "title": props(show=b(False))}
    if S.shadow:
        vco["dropShadow"] = props(show=b(True), preset=s("Bottom"), position=s("Outer"), color=color("#1F2328"), transparency=n(92),
                                  shadowBlur=n(14), shadowDistance=n(2), shadowSpread=n(0))
    else:
        vco["dropShadow"] = props(show=b(False))
    if S.radius and not S.border:
        vco["border"] = props(show=b(True), color=color(fill or S.panel), radius=n(S.radius), width=n(1))
    return page.add(key, "textbox", x, y, w, h, objects={"general": props(paragraphs=[{"textRuns": [{"value": " "}]}])},
                    vco=vco, parent=parent, hidden=hidden, alt="Panel")


def _nav_text(default, selected, size=11, align=None):
    base = {"fontFamily": s(S.font_b), "fontSize": n(size), "fontColor": color(default)}
    if align:
        base.update(horizontalAlignment=s(align), leftMargin=i(22))
    sel = dict(base, fontColor=color(selected))
    return [{"properties": base, "selector": {"id": "default"}},
            {"properties": sel, "selector": {"id": "selected"}},
            {"properties": dict(sel), "selector": {"id": "hover"}}]


def _fill3(default, selected, hover, show=True):
    return [{"properties": {"show": b(show), "fillColor": color(default), "transparency": n(0)}, "selector": {"id": "default"}},
            {"properties": {"show": b(True), "fillColor": color(selected), "transparency": n(0)}, "selector": {"id": "selected"}},
            {"properties": {"show": b(True), "fillColor": color(hover), "transparency": n(0)}, "selector": {"id": "hover"}}]


def chrome(page, report_title, title, subtitle, page_no, source, slicers=(), title_w=None):
    """Navigation + action-title band + footer, drawn in the active style's pattern."""
    tw = title_w or (S.slicer_x - S.x0 - GUT)
    if S.nav == "topbar":
        shape(page, "bar", 0, 0, W, 64, S.header_bg)
        shape(page, "bar-accent", 0, 64, W, 4, S.accent)
        textbox(page, "brand", MG, 17, 760, 32, [[(report_title.upper(), 15, S.header_ink, True), ("   |   Portfolio report by Mohamed Ayman", 12, S.header_muted, False)]])
        page.add("nav", "pageNavigator", 900, 12, W - 900 - MG, 42,
                 objects={"layout": props(orientation=lit("0D"), cellPadding=i(6), rowCount=i(1), columnCount=i(page.report.n_pages)),
                          "text": _nav_text(S.header_muted, S.header_ink),
                          "fill": _fill3(S.header_bg, _mix(S.header_bg, "#FFFFFF", .10), _mix(S.header_bg, "#FFFFFF", .06)),
                          "outline": [{"properties": {"show": b(False)}, "selector": {"id": "default"}},
                                      {"properties": {"show": b(True), "lineColor": color(S.accent), "weight": n(2)}, "selector": {"id": "selected"}}]},
                 vco=dict(NOBOX), alt="Page navigation")
    elif S.nav == "rail":
        rail = S.x0 - MG
        shape(page, "rail", 0, 0, rail, H, S.header_bg)
        textbox(page, "brand-eyebrow", 28, 36, rail - 48, 22, [[("PORTFOLIO  ·  MOHAMED AYMAN", 10, S.header_muted, True)]])
        textbox(page, "brand", 28, 64, rail - 48, 96, [[(report_title, 22, S.header_ink, False, S.font_title)]])
        shape(page, "brand-rule", 28, 168, 48, 3, S.accent)
        page.add("nav", "pageNavigator", 12, 196, rail - 24, 58 * page.report.n_pages,
                 objects={"layout": props(orientation=lit("1D"), cellPadding=i(4), columnCount=i(1), rowCount=i(page.report.n_pages)),
                          "text": _nav_text(S.header_muted, S.header_ink, 13, "left"),
                          "fill": _fill3(S.header_bg, _mix(S.header_bg, "#FFFFFF", .08), _mix(S.header_bg, "#FFFFFF", .04)),
                          "accentBar": [{"properties": {"show": b(False)}, "selector": {"id": "default"}},
                                        {"properties": {"show": b(True), "position": s("Left"), "color": color(S.accent), "width": n(4)},
                                         "selector": {"id": "selected"}}],
                          "outline": [{"properties": {"show": b(False)}}]},
                 vco=dict(NOBOX), alt="Page navigation")
        textbox(page, "rail-foot", 28, H - 96, rail - 48, 60, [[(f"Page {page_no}", 11, S.header_muted, True)],
                                                                [("Built in Power BI (PBIR, TMDL)", 10, S.header_muted, False)]])
    elif S.nav == "pills":
        shape(page, "brand-dot", MG, 30, 12, 12, S.accent, radius=50)
        textbox(page, "brand", MG + 22, 22, 640, 28, [[(report_title.upper(), 13, S.ink, True), ("     Mohamed Ayman  ·  portfolio", 12, S.muted, False)]])
        n_pages = page.report.n_pages
        nav_w = (166 if n_pages > 5 else 196) * n_pages
        page.add("nav", "pageNavigator", W - MG - nav_w, 16, nav_w, 44,
                 objects={"layout": props(orientation=lit("0D"), cellPadding=i(8), rowCount=i(1), columnCount=i(n_pages)),
                          "shape": props(tileShape=s("pill")),
                          "text": _nav_text(S.muted, S.page, 11),
                          "fill": _fill3(S.panel, S.primary, _mix(S.panel, "#FFFFFF", .06)),
                          "outline": [{"properties": {"show": b(True), "lineColor": color(S.line), "weight": n(1)}, "selector": {"id": "default"}},
                                      {"properties": {"show": b(False)}, "selector": {"id": "selected"}}]},
                 vco=dict(NOBOX), alt="Page navigation")
        shape(page, "head-rule", 0, 72, W, 1, S.line)
    elif S.nav == "underline":
        shape(page, "head", 0, 0, W, 76, S.header_bg)
        shape(page, "head-rule", 0, 76, W, 1, S.line)
        shape(page, "logo", MG, 20, 36, 36, S.primary, radius=10)
        textbox(page, "logo-t", MG, 26, 36, 26, [["center", ("MA", 13, "#FFFFFF", True)]])
        textbox(page, "brand", MG + 50, 17, 700, 44, [[(report_title, 16, S.header_ink, True)], [("Retail analytics portfolio  ·  Mohamed Ayman", 11, S.header_muted, False)]])
        n_pages = page.report.n_pages
        nav_w = 180 * n_pages
        page.add("nav", "pageNavigator", W - MG - nav_w, 14, nav_w, 62,
                 objects={"layout": props(orientation=lit("0D"), cellPadding=i(2), rowCount=i(1), columnCount=i(n_pages)),
                          "text": _nav_text(S.header_muted, S.header_ink, 12),
                          "fill": _fill3(S.header_bg, S.header_bg, S.page),
                          "accentBar": [{"properties": {"show": b(False)}, "selector": {"id": "default"}},
                                        {"properties": {"show": b(True), "position": s("Bottom"), "color": color(S.primary), "width": n(4)},
                                         "selector": {"id": "selected"}}],
                          "outline": [{"properties": {"show": b(False)}}],
                          "shadow": [{"properties": {"show": b(False)}, "selector": {"id": k}} for k in ("default", "hover", "selected")]},
                 vco=dict(NOBOX), alt="Page navigation")
    elif S.nav == "segmented":
        shape(page, "head", 0, 0, W, 64, S.header_bg)
        textbox(page, "brand", MG, 12, 600, 44, [[(report_title, 15, S.header_ink, True)], [("Issuer advisory portfolio  ·  Mohamed Ayman", 11, S.header_muted, False)]])
        n_pages = page.report.n_pages
        nav_w = 172 * n_pages
        nx = W - MG - nav_w
        track = _mix(S.header_bg, "#FFFFFF", .12)
        page.add("nav-track", "textbox", nx - 6, 10, nav_w + 12, 44, objects={"general": props(paragraphs=[{"textRuns": [{"value": " "}]}])},
                 vco={"background": props(show=b(True), color=color(track), transparency=n(0)),
                      "border": props(show=b(True), color=color(track), radius=n(22), width=n(1)),
                      "padding": props(top=n(0), bottom=n(0), left=n(0), right=n(0)), "title": props(show=b(False)),
                      "dropShadow": props(show=b(False))}, alt="Navigation track")
        page.add("nav", "pageNavigator", nx, 14, nav_w, 36,
                 objects={"layout": props(orientation=lit("0D"), cellPadding=i(4), rowCount=i(1), columnCount=i(n_pages)),
                          "shape": props(tileShape=s("pill")),
                          "text": _nav_text(S.header_muted, S.header_bg, 11),
                          "fill": [{"properties": {"show": b(True)}},
                                   {"properties": {"fillColor": color(track), "transparency": n(0)}, "selector": {"id": "default"}},
                                   {"properties": {"fillColor": color("#FFFFFF"), "transparency": n(0)}, "selector": {"id": "selected"}},
                                   {"properties": {"fillColor": color(_mix(track, "#FFFFFF", .15)), "transparency": n(0)}, "selector": {"id": "hover"}}],
                          "outline": [{"properties": {"show": b(False)}}]},
                 vco=dict(NOBOX), alt="Page navigation")
    elif S.nav == "hero":
        shape(page, "hero", 0, 0, W, 304, S.header_bg)
        shape(page, "hero-rule", 0, 304, W, 4, S.accent)
        textbox(page, "brand", MG, 18, 640, 28, [[(report_title.upper(), 13, S.header_ink, True), ("     Mohamed Ayman  ·  portfolio", 12, S.header_muted, False)]])
        n_pages = page.report.n_pages
        nav_w = 190 * n_pages
        page.add("nav", "pageNavigator", W - MG - nav_w, 8, nav_w, 48,
                 objects={"layout": props(orientation=lit("0D"), cellPadding=i(2), rowCount=i(1), columnCount=i(n_pages)),
                          "text": _nav_text(S.header_muted, S.header_ink, 12),
                          "fill": [{"properties": {"show": b(False)}}],
                          "accentBar": [{"properties": {"show": b(False)}, "selector": {"id": "default"}},
                                        {"properties": {"show": b(True), "position": s("Bottom"), "color": color(S.accent), "width": n(3)},
                                         "selector": {"id": "selected"}}],
                          "outline": [{"properties": {"show": b(False)}}]},
                 vco=dict(NOBOX), alt="Page navigation")
    # title band
    ty = S.title_y
    if S.nav == "pills":
        textbox(page, "eyebrow", S.x0, ty - 6, tw, 22, [[(f"PAGE {page_no}", 11, S.accent, True)]])
        ty += 14
    t_ink, s_ink = (S.header_ink, S.header_muted) if S.nav == "hero" else (S.ink, S.muted)
    textbox(page, "title", S.x0, ty, tw, 52, [[(title, S.title_size, t_ink, S.font_title != "Georgia", S.font_title)]])
    textbox(page, "subtitle", S.x0, ty + 56, tw, 26, [[(subtitle, 14, s_ink, False)]])
    if S.nav == "rail":
        shape(page, "title-rule", S.x0, ty + 92, 64, 2, S.accent)
    # footer
    shape(page, "foot-line", S.x0, S.foot_y - 10, S.cw, 1, S.line)
    textbox(page, "source", S.x0, S.foot_y, S.cw - 380, 22, [[("Source: " + source, 11, S.faint, False)]])
    if S.nav != "rail":
        textbox(page, "pageno", S.x0 + S.cw - 360, S.foot_y, 360, 22, [["right", (f"Mohamed Ayman  ·  Data & BI  ·  {page_no}", 11, S.faint, False)]])
    else:
        textbox(page, "pageno", S.x0 + S.cw - 360, S.foot_y, 360, 22, [["right", ("Mohamed Ayman  ·  Data & BI", 11, S.faint, False)]])


def _mix(a, b_, t):
    """Blend hex colour a toward b_ by t (0..1)."""
    ca = [int(a[k:k + 2], 16) for k in (1, 3, 5)]
    cb = [int(b_[k:k + 2], 16) for k in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(ca, cb))


def kpi_band(page, key, measures, deltas=None, y=None, h=None, x=None, w=None, trend=None):
    """Headline KPIs in the active style: one divided band, or separate tiles (optionally with sparklines)."""
    y = S.kpi_y if y is None else y
    h = S.kpi_h if h is None else h
    x = S.x0 if x is None else x
    w = S.cw if w is None else w
    if S.kpi == "tiles":
        return kpi_tiles(page, key, measures, deltas, x, y, w, h, trend)
    if S.kpi == "column":
        return kpi_column(page, key, measures, deltas, S.x0, S.kpi_y, S.kpi_w, S.foot_y - 16 - S.kpi_y)
    if S.kpi == "hero":
        return page.add(key + "-values", "cardVisual", x - 12, y, w + 12, h,
                        query={"Data": [proj(FLD(m), display=m[1]) if isinstance(m, tuple) else proj(m) for m in measures]},
                        objects={"value": idprops(fontSize=n(32), fontColor=color(S.header_ink), fontFamily=s(S.font_num), horizontalAlignment=s("left"),
                                                  labelDisplayUnits=s("0")),
                                 "label": idprops(show=b(True), fontSize=n(12), fontColor=color(S.header_muted), position=s("aboveValue"),
                                                  horizontalAlignment=s("left")),
                                 "outline": idprops(show=b(False)), "fillCustom": [{"properties": {"show": b(False)}}],
                                 "layout": idprops(style=s("Table"), customizeLines=b(True), gridlineWidth=n(1),
                                                   gridlineColor=color(_mix(S.header_bg, "#FFFFFF", .22)), gridlineStyle=s("solid")),
                                 "padding": idprops(leftMargin=i(12), rightMargin=i(8), topMargin=i(0), bottomMargin=i(0))},
                        vco=dict(NOBOX), alt="Headline KPIs")
    panel(page, key + "-panel", x, y, w, h)
    vals_h = 84 if deltas else h - 24
    page.add(key + "-values", "cardVisual", x + 8, y + 12, w - 16, vals_h,
             query={"Data": [proj(f) for f in measures]},
             objects={"value": idprops(fontSize=n(30), fontColor=color(S.ink), fontFamily=s(S.font_num), horizontalAlignment=s("left"),
                                       labelDisplayUnits=lit("'0'")),
                      "label": idprops(show=b(True), fontSize=n(12), fontColor=color(S.muted), position=s("aboveValue"), horizontalAlignment=s("left")),
                      "outline": idprops(show=b(False)),
                      "layout": idprops(style=s("Table"), customizeLines=b(True), gridlineWidth=n(1), gridlineColor=color(S.line), gridlineStyle=s("solid")),
                      "padding": idprops(leftMargin=i(20), rightMargin=i(8), topMargin=i(0), bottomMargin=i(0))},
             vco=dict(NOBOX), alt="Headline KPIs")
    if deltas:
        page.add(key + "-deltas", "cardVisual", x + 8, y + 92, w - 16, 30, query={"Data": [proj(f) for f in deltas]},
                 objects={"value": idprops(fontSize=n(12), fontColor=color(_mix(S.ink, S.muted, .5)), fontFamily=s(S.font_b), horizontalAlignment=s("left")),
                          "label": idprops(show=b(False)), "outline": idprops(show=b(False)),
                          "padding": idprops(leftMargin=i(20), rightMargin=i(8), topMargin=i(0), bottomMargin=i(0))},
                 vco=dict(NOBOX), alt="Change versus prior period")


def kpi_tiles(page, key, measures, deltas, x, y, w, h, trend=None, gap=16):
    """One tile per KPI: surface + accent bar + value card (+ caption card, + sparkline when trend is a category field)."""
    k = len(measures)
    tw = (w - gap * (k - 1)) // k
    for j, m in enumerate(measures):
        tx = x + j * (tw + gap)
        panel(page, f"{key}-t{j}", tx, y, tw, h)
        if S.accent_bar == "Left":
            shape(page, f"{key}-a{j}", tx, y + 18, 3, h - 36, S.accent if j == 0 else S.primary)
        elif S.accent_bar == "Top":
            shape(page, f"{key}-a{j}", tx, y, tw, 3, S.accent if j == 0 else S.primary)
        else:
            shape(page, f"{key}-a{j}", tx + 18, y + h - 4, 40, 4, S.primary)
        q = {"Data": [proj(FLD(m), display=m[1]) if isinstance(m, tuple) else proj(m)]}
        units = m[2] if isinstance(m, tuple) and len(m) > 2 else None
        vprops = dict(fontSize=n(28), fontColor=color(S.ink), fontFamily=s(S.font_num), horizontalAlignment=s("left"),
                      labelDisplayUnits=s(str(units) if units else "0"))
        if units:
            vprops["labelPrecision"] = i(1)
        objs = {"value": idprops(**vprops), "fillCustom": idprops(show=b(False)),
                "label": idprops(show=b(True), fontSize=n(11), fontColor=color(S.muted), position=s("aboveValue"), horizontalAlignment=s("left"),
                                 fontFamily=s(S.font_b)),
                "outline": idprops(show=b(False)),
                "padding": idprops(leftMargin=i(18), rightMargin=i(8), topMargin=i(0), bottomMargin=i(0))}
        vh = 84
        if trend is not None:
            q["TrendVisualCategory"] = [P(trend)]
            objs["trendVisualLibrary"] = props(show=b(True), chartType=s("line"), includePrimaryMeasure=b(True), showTooltip=b(True), legendShow=b(False))
            objs["lineTrendVisual"] = props(lineShow=b(True), lineWidth=n(2), lineColor=color(S.accent if j == 0 else S.primary),
                                            fillArea=b(True), fillAreaTransparency=n(80), showMarkers=b(False))
            objs["referenceLabelLayout"] = props(position=s("below"))
            vh = h - 20
        page.add(f"{key}-v{j}", "cardVisual", tx + 4, y + 14, tw - 8, vh, query=q, objects=objs, vco=dict(NOBOX), alt="Headline KPI")
        if deltas and j < len(deltas) and deltas[j] is not None:
            page.add(f"{key}-d{j}", "cardVisual", tx + 4, y + h - 38, tw - 8, 28, query={"Data": [proj(FLD(deltas[j]))]},
                     objects={"value": idprops(fontSize=n(11), fontColor=color(_mix(S.ink, S.muted, .4)), fontFamily=s(S.font_b),
                                               horizontalAlignment=s("left")),
                              "label": idprops(show=b(False)), "outline": idprops(show=b(False)),
                              "padding": idprops(leftMargin=i(18), rightMargin=i(8), topMargin=i(0), bottomMargin=i(0))},
                     vco=dict(NOBOX), alt="KPI context")


def kpi_column(page, key, measures, deltas, x, y, w, h, gap=14):
    """KPIs stacked in a left column: each card has a coloured left rule, label, value and an optional caption."""
    k = len(measures)
    th = (h - gap * (k - 1)) // k
    for j, m in enumerate(measures):
        ty = y + j * (th + gap)
        panel(page, f"{key}-t{j}", x, ty, w, th)
        shape(page, f"{key}-a{j}", x, ty + 16, 4, th - 32, S.accent if j == 0 else S.primary)
        units = m[2] if isinstance(m, tuple) and len(m) > 2 else None
        vprops = dict(fontSize=n(30), fontColor=color(S.ink), fontFamily=s(S.font_num), horizontalAlignment=s("left"),
                      labelDisplayUnits=s(str(units) if units else "0"))
        if units:
            vprops["labelPrecision"] = i(1)
        has_cap = deltas and j < len(deltas) and deltas[j] is not None
        page.add(f"{key}-v{j}", "cardVisual", x + 8, ty + 10, w - 16, th - (40 if has_cap else 20),
                 query={"Data": [proj(FLD(m), display=m[1]) if isinstance(m, tuple) else proj(m)]},
                 objects={"value": idprops(**vprops), "fillCustom": [{"properties": {"show": b(False)}}],
                          "label": idprops(show=b(True), fontSize=n(11), fontColor=color(S.muted), position=s("aboveValue"),
                                           horizontalAlignment=s("left"), fontFamily=s(S.font_b)),
                          "outline": idprops(show=b(False)),
                          "padding": idprops(leftMargin=i(18), rightMargin=i(8), topMargin=i(0), bottomMargin=i(0))},
                 vco=dict(NOBOX), alt="Headline KPI")
        if has_cap:
            page.add(f"{key}-d{j}", "cardVisual", x + 8, ty + th - 36, w - 16, 26, query={"Data": [proj(FLD(deltas[j]))]},
                     objects={"value": idprops(fontSize=n(11), fontColor=color(S.muted), fontFamily=s(S.font), horizontalAlignment=s("left")),
                              "label": idprops(show=b(False)), "outline": idprops(show=b(False)), "fillCustom": [{"properties": {"show": b(False)}}],
                              "padding": idprops(leftMargin=i(18), rightMargin=i(8), topMargin=i(0), bottomMargin=i(0))},
                     vco=dict(NOBOX), alt="KPI context")


def P(x):
    """A field, or (field, display name) -> projection with a human-readable label."""
    return proj(x[0], display=x[1]) if isinstance(x, tuple) else proj(x)


def FLD(x):
    return x[0] if isinstance(x, tuple) else x


def _title(t, sub=None):
    out = {"title": props(show=b(True), text=s(t))}
    if sub:
        out["subTitle"] = props(show=b(True), text=s(sub))
    return out


def fill_for(field, hexv):
    return {"properties": {"fill": color(hexv)}, "selector": {"metadata": qref(FLD(field))}}


def _split(field):
    kind = "Column" if "Column" in field else "Measure"
    return field[kind]["Expression"]["SourceRef"]["Entity"], field[kind]["Property"]


def fill_gradient(measure, lo, hi, mid=None):
    """Data-driven colour on bars/columns (FillRule on dataPoint.fill)."""
    if mid:
        rule = {"linearGradient3": {"min": {"color": L(f"'{lo}'")}, "mid": {"color": L(f"'{mid}'")}, "max": {"color": L(f"'{hi}'")},
                                    "nullColoringStrategy": {"strategy": L("'asZero'")}}}
    else:
        rule = {"linearGradient2": {"min": {"color": L(f"'{lo}'")}, "max": {"color": L(f"'{hi}'")},
                                    "nullColoringStrategy": {"strategy": L("'asZero'")}}}
    return [{"properties": {"fill": {"solid": {"color": {"expr": {"FillRule": {"Input": M(*_split(FLD(measure))), "FillRule": rule}}}}}},
             "selector": {"data": [{"dataViewWildcard": {"matchingOption": 1}}]}}]


def bar(page, key, x, y, w, h, category, measure, title, sub=None, horizontal=True, sort_desc=True, fmt_units=None,
        gradient="default", filters=None, sort_cat=None, field_param=None, parent=None):
    """Single-series bar/column chart, bars shaded by value (light -> primary) unless gradient=None.
    gradient may be (lo, hi) or (lo, mid, hi)."""
    vtype = "clusteredBarChart" if horizontal else "clusteredColumnChart"
    if gradient == "default":
        gradient = (S.light, S.primary)
    if gradient:
        lo, hi = gradient[0], gradient[-1]
        objs = {"dataPoint": fill_gradient(measure, lo, hi, gradient[1] if len(gradient) == 3 else None)}
    else:
        objs = {"dataPoint": [{"properties": {"defaultColor": color(S.primary)}}]}
    if fmt_units:
        objs["labels"] = props(show=b(True), labelDisplayUnits=lit(f"{fmt_units}D"), labelPrecision=i(1), fontSize=n(10), color=color(S.ink))
    sort = [(FLD(measure), "Descending")] if sort_desc else ([(FLD(sort_cat), "Ascending")] if sort_cat else None)
    fp = {"Category": field_param} if field_param else None
    return page.add(key, vtype, x, y, w, h, query={"Category": [P(category)], "Y": [P(measure)]}, objects=objs,
                    vco=_title(title, sub), sort=sort, alt=title, filters=filters, field_params=fp, parent=parent)


def combo(page, key, x, y, w, h, category, cols, lines, title, sub=None, col_colors=None, line_colors=None, sort_cat=None):
    col_colors = col_colors or (S.primary,)
    line_colors = line_colors or (S.accent,)
    objs = {"dataPoint": [fill_for(f, c) for f, c in zip(cols, col_colors)] + [fill_for(f, c) for f, c in zip(lines, line_colors)]}
    sort = [(FLD(sort_cat), "Ascending")] if sort_cat else None
    return page.add(key, "lineClusteredColumnComboChart", x, y, w, h,
                    query={"Category": [P(category)], "Y": [P(f) for f in cols], "Y2": [P(f) for f in lines]},
                    objects=objs, vco=_title(title, sub), sort=sort, alt=title)


def line(page, key, x, y, w, h, category, ys, title, sub=None, colors=None, y2=None, y2_colors=None, sort_cat=None,
         area=False, series=None, markers=False):
    colors = colors or (S.accent, S.primary, S.grey, S.warn)
    y2_colors = y2_colors or (S.grey,)
    objs = {"dataPoint": [fill_for(f, c) for f, c in zip(ys, colors)] + ([fill_for(f, c) for f, c in zip(y2, y2_colors)] if y2 else [])}
    if area or markers:
        objs["lineStyles"] = props(areaShow=b(area), showMarker=b(markers))
    q = {"Category": [P(category)], "Y": [P(f) for f in ys]}
    if y2:
        q["Y2"] = [P(f) for f in y2]
    if series:
        q["Series"] = [P(series)]
        objs.pop("dataPoint")
    sort = [(FLD(sort_cat), "Ascending")] if sort_cat else None
    return page.add(key, "lineChart", x, y, w, h, query=q, objects=objs, vco=_title(title, sub), sort=sort, alt=title)


def stacked_column(page, key, x, y, w, h, category, measures, title, sub=None, colors=None, sort_cat=None, horizontal=False,
                   series=None, pct=False):
    colors = colors or (S.primary, S.accent, S.grey, S.warn)
    vtype = ("hundredPercentStackedBarChart" if pct else "barChart") if horizontal else ("hundredPercentStackedColumnChart" if pct else "columnChart")
    q = {"Category": [P(category)], "Y": [P(f) for f in measures]}
    objs = {"labels": props(show=b(False)), "categoryAxis": props(showAxisTitle=b(False)), "valueAxis": props(showAxisTitle=b(False))}
    if series:
        q["Series"] = [P(series)]
    else:
        objs["dataPoint"] = [fill_for(f, c) for f, c in zip(measures, colors)]
    sort = [(FLD(sort_cat), "Ascending")] if sort_cat else None
    return page.add(key, vtype, x, y, w, h, query=q, objects=objs, vco=_title(title, sub), sort=sort, alt=title)


def gradient_rule(field, lo=None, hi=None, mid=None, mid_value=None):
    """Colour scale; with mid, a diverging scale centred on mid_value (e.g. 0 for uplift, 1 for lift) when given."""
    lo, hi = lo or S.panel, hi or S.primary
    rule = {"linearGradient2": {"min": {"color": L(f"'{lo}'")}, "max": {"color": L(f"'{hi}'")},
                                "nullColoringStrategy": {"strategy": L("'asZero'")}}}
    if mid:
        midstop = {"color": L(f"'{mid}'")}
        if mid_value is not None:
            midstop["value"] = L(f"{mid_value}D")
        rule = {"linearGradient3": {"min": {"color": L(f"'{lo}'")}, "mid": midstop, "max": {"color": L(f"'{hi}'")},
                                    "nullColoringStrategy": {"strategy": L("'asZero'")}}}
    f = FLD(field)
    return {"properties": {"backColor": {"solid": {"color": {"expr": {"FillRule": {"Input": {"SelectRef": {"ExpressionName": qref(f)}}, "FillRule": rule}}}}}},
            "selector": {"data": [{"dataViewWildcard": {"matchingOption": 1}}], "metadata": qref(f)}}


def matrix(page, key, x, y, w, h, rows, values, title, sub=None, columns=None, gradients=(), databars=(), sort=None, filters=None, font=10,
           field_param=None, font_colors=()):
    """Matrix with numeric headers right-aligned over their values, optional colour scales
    (gradients: (field, low, high[, mid])), in-cell data bars (databars: [field]) and font colour scales (font_colors)."""
    q = {"Rows": [P(f) for f in rows], "Values": [P(f) for f in values]}
    if columns:
        q["Columns"] = [P(f) for f in columns]
    bars_set = {qref(FLD(f)) for f in databars}
    cf = []
    for v in values:
        ref = qref(FLD(v))
        pr = {"alignment": s("Right"), "styleHeader": b(True), "styleValues": b(True)}
        if ref in bars_set:
            pr["dataBars"] = {"positiveColor": color(S.light), "negativeColor": color(S.soft_neg), "axisColor": color(S.grey),
                              "reverseDirection": b(False), "hideText": b(False)}
        cf.append({"properties": pr, "selector": {"metadata": ref}})
    vals = [{"properties": {"fontSize": n(font)}}] + [gradient_rule(*g) if isinstance(g, tuple) else gradient_rule(g) for g in gradients]
    for fld, lo, hi in font_colors:
        rule = {"linearGradient2": {"min": {"color": L(f"'{lo}'")}, "max": {"color": L(f"'{hi}'")}, "nullColoringStrategy": {"strategy": L("'asZero'")}}}
        vals.append({"properties": {"fontColor": {"solid": {"color": {"expr": {"FillRule": {"Input": {"SelectRef": {"ExpressionName": qref(FLD(fld))}},
                                                                                              "FillRule": rule}}}}}},
                     "selector": {"data": [{"dataViewWildcard": {"matchingOption": 1}}], "metadata": qref(FLD(fld))}})
    # pivoted matrices: column headers sit over numbers, so right-align them; plain matrices keep a left-aligned corner
    # header and right-align each numeric header through columnFormatting.
    objs = {"columnHeaders": props(columnAdjustment=s("growToFit"), autoSizeColumnWidth=b(True), fontSize=n(font),
                                   alignment=s("Right" if columns else "Left"), titleAlignment=s("Left"), wordWrap=b(True)),
            "values": vals,
            "rowHeaders": props(fontSize=n(font), alignment=s("Left")),
            "columnFormatting": cf,
            "subTotals": props(rowSubtotals=b(False), columnSubtotals=b(False))}
    vco = _title(title, sub)
    vco["stylePreset"] = props(name=s("None"))
    fp = {"Rows": field_param} if field_param else None
    return page.add(key, "pivotTable", x, y, w, h, query=q, objects=objs, vco=vco, sort=[(FLD(f), d) for f, d in sort] if sort else None,
                    alt=title, filters=filters, field_params=fp)


def table(page, key, x, y, w, h, columns, title, sub=None, gradients=(), databars=(), sort=None, filters=None, font=10, top_n=None):
    """Flat table (tableEx); numeric columns right-aligned with styled headers."""
    bars_set = {qref(FLD(f)) for f in databars}
    cf = []
    for v in columns:
        f = FLD(v)
        if "Measure" in f:
            pr = {"alignment": s("Right"), "styleHeader": b(True), "styleValues": b(True)}
            if qref(f) in bars_set:
                pr["dataBars"] = {"positiveColor": color(S.light), "negativeColor": color(S.soft_neg), "axisColor": color(S.grey),
                                  "reverseDirection": b(False), "hideText": b(False)}
            cf.append({"properties": pr, "selector": {"metadata": qref(f)}})
    objs = {"columnHeaders": props(fontSize=n(font), alignment=s("Left"), wordWrap=b(True), columnAdjustment=s("growToFit"), autoSizeColumnWidth=b(True)),
            "values": [{"properties": {"fontSize": n(font)}}] + [gradient_rule(*g) if isinstance(g, tuple) else gradient_rule(g) for g in gradients],
            "columnFormatting": cf, "total": props(totals=b(False))}
    vco = _title(title, sub)
    vco["stylePreset"] = props(name=s("None"))
    return page.add(key, "tableEx", x, y, w, h, query={"Values": [P(c) for c in columns]}, objects=objs, vco=vco,
                    sort=[(FLD(f), d) for f, d in sort] if sort else None, alt=title, filters=filters)


def _in_condition(entity, prop, values, src="t"):
    return {"Condition": {"In": {"Expressions": [{"Column": {"Expression": {"SourceRef": {"Source": src}}, "Property": prop}}],
                                 "Values": [[lit(f"'{v}'" if isinstance(v, str) else f"{v}L")["expr"]] for v in values]}}}


def button_slicer(page, key, x, y, w, h, field, title, selected=None, columns=4, rows=1, single=False, parent=None, hidden=False):
    objs = {"layout": props(columnCount=i(columns), rowCount=i(rows)),
            "value": idprops(fontSize=n(11), fontFamily=s(S.font_b), fontColor=color(S.ink)),
            "fillCustom": [{"properties": {"show": b(True), "fillColor": color(S.panel)}, "selector": {"id": "default"}},
                           {"properties": {"show": b(True), "fillColor": color(S.sel_fill)}, "selector": {"id": "selected"}}],
            "outline": [{"properties": {"show": b(True), "lineColor": color(S.line), "weight": n(1)}, "selector": {"id": "default"}},
                        {"properties": {"show": b(True), "lineColor": color(S.primary), "weight": n(2)}, "selector": {"id": "selected"}}],
            "selection": props(singleSelect=b(single))}
    if S.radius > 4:
        objs["shapeCustomRectangle"] = props(tileShape=s("rectangleRounded"), rectangleRoundedCurve=i(min(S.radius, 10)))
    if selected is not None:
        ent, prop = _split(field)
        objs["general"] = [{"properties": {"filter": {"filter": {"Version": 2, "From": [{"Name": "d", "Entity": ent, "Type": 0}],
                                                                 "Where": [_in_condition(ent, prop, selected, "d")]}}}}]
    return page.add(key, "advancedSlicerVisual", x, y, w, h, query={"Values": [proj(field)]}, objects=objs,
                    vco={"title": props(show=b(True), text=s(title), fontSize=n(10), fontColor=color(S.muted), fontFamily=s(S.font)),
                         "background": props(show=b(False)), "border": props(show=b(False)), "dropShadow": props(show=b(False)),
                         "padding": props(top=n(0), bottom=n(0), left=n(0), right=n(0))}, alt=title, parent=parent, hidden=hidden)


def dropdown(page, key, x, y, w, h, field, title, parent=None, hidden=False, mode="Dropdown"):
    return page.add(key, "slicer", x, y, w, h, query={"Values": [proj(field)]},
                    objects={"data": props(mode=s(mode)), "header": props(show=b(True), text=s(title))},
                    vco={"background": props(show=b(False)), "border": props(show=b(False)), "title": props(show=b(False)),
                         "dropShadow": props(show=b(False)), "padding": props(top=n(0), bottom=n(0), left=n(0), right=n(0))},
                    alt=title, parent=parent, hidden=hidden)


def in_filter(entity, prop, values, key):
    """Visual-level categorical IN filter."""
    return {"name": "Filter" + hexid(entity, prop, key, n=24), "field": C(entity, prop), "type": "Categorical",
            "filter": {"Version": 2, "From": [{"Name": "t", "Entity": entity, "Type": 0}], "Where": [_in_condition(entity, prop, values)]}}


def button(page, key, x, y, w, h, text, bookmark=None, action=None, icon="blank", fill=None, ink=None, outline=None, parent=None,
           hidden=False, tooltip=None, size=12):
    fill = fill or S.primary
    ink = ink or "#FFFFFF"
    objs = {"text": [{"properties": {"show": b(True)}},
                     {"properties": {"text": s(text), "fontColor": color(ink), "fontSize": n(size), "fontFamily": s(S.font_b),
                                     "horizontalAlignment": s("center")}, "selector": {"id": "default"}}],
            "fill": [{"properties": {"show": b(True)}},
                     {"properties": {"fillColor": color(fill), "transparency": n(0)}, "selector": {"id": "default"}},
                     {"properties": {"fillColor": color(_mix(fill, "#000000", .12)), "transparency": n(0)}, "selector": {"id": "hover"}}],
            "outline": [{"properties": {"show": b(bool(outline))}}] + ([{"properties": {"lineColor": color(outline), "weight": n(1)},
                                                                         "selector": {"id": "default"}}] if outline else []),
            "icon": [{"properties": {"show": b(icon != "blank")}}, {"properties": {"shapeType": s(icon)}, "selector": {"id": "default"}}],
            "shape": props(tileShape=s("rectangleRounded"), rectangleRoundedCurve=i(min(max(S.radius, 4), 18)))}
    link = {"show": b(True)}
    if bookmark:
        link.update(type=s("Bookmark"), bookmark=s(bookmark))
    elif action:
        link.update(type=s(action))
    if tooltip:
        link["tooltip"] = s(tooltip)
    vco = dict(NOBOX)
    vco["visualLink"] = props(**link)
    return page.add(key, "actionButton", x, y, w, h, objects=objs, vco=vco, alt=text, parent=parent, hidden=hidden)


def filter_panel(report, page, slicers, label="Filters", x_btn=None, y_btn=None, width=440):
    """A slide-in filter panel that opens and closes with a button (two display-only bookmarks).

    slicers: list of callables f(page, x, y, w, parent) -> height used, stacked top to bottom inside the panel.
    The panel starts hidden; slicer selections survive opening and closing (bookmarks suppress data).
    The open button is drawn first (below the panel in z-order) so the open panel covers it."""
    import os as _os
    _show = bool(_os.environ.get("PANEL_DEBUG"))             # render the panel open, for screenshot review only
    px, py, ph = W - width, 0, H
    gkey = f"fp-{label}"
    show_bm, hide_bm = report.bookmark_names(page, hexid(page.name, gkey))
    bx = x_btn if x_btn is not None else S.x0 + S.cw - 150
    by = y_btn if y_btn is not None else S.title_y + 8
    button(page, "fp-open", bx, by, 150, 44, f"≡  {label}", bookmark=show_bm, fill=S.panel, ink=S.ink, outline=S.line,
           tooltip="Open the filter panel")
    grp = page.group(gkey, f"{label} panel", px - 4, py, width + 4, ph, hidden=not _show)
    hd = not _show
    shape(page, "fp-shadow", px - 4, py, 4, ph, _mix(S.page, "#000000", .12), parent=grp, hidden=hd)
    shape(page, "fp-bg", px, py, width, ph, S.panel, line=S.line, parent=grp, hidden=hd)
    shape(page, "fp-head", px, py, width, 76, S.header_bg if not S.dark else S.panel, parent=grp, hidden=hd)
    shape(page, "fp-head-rule", px, 76, width, 3, S.primary, parent=grp, hidden=hd)
    textbox(page, "fp-title", px + 28, 22, width - 140, 32, [[(label, 20, S.header_ink if not S.dark else S.ink, True)]], parent=grp, hidden=hd)
    yy = 104
    inner_w = width - 56
    for fn in slicers:
        used = fn(page, px + 28, yy, inner_w, grp)
        yy += used + 20
    reset_y = H - 92
    button(page, "fp-clear", px + 28, reset_y, (inner_w - 16) // 2, 44, "Clear all", action="ClearAllSlicers", fill=S.panel,
           ink=S.ink, outline=S.line, parent=grp, hidden=hd)
    button(page, "fp-done", px + 28 + (inner_w - 16) // 2 + 16, reset_y, (inner_w - 16) // 2, 44, "Apply & close", bookmark=hide_bm,
           parent=grp, hidden=hd)
    button(page, "fp-x", px + width - 72, 18, 44, 40, "✕", bookmark=hide_bm, fill=S.header_bg if not S.dark else S.panel,
           ink=S.header_ink if not S.dark else S.ink, parent=grp, hidden=hd, size=16, tooltip="Close filters")
    report.toggle_bookmarks(page, grp, label)
    return grp


def azure_map(page, key, x, y, w, h, location, size, title, sub=None, fill=None):
    return page.add(key, "azureMap", x, y, w, h,
                    query={"Category": [dict(P(location), active=True)], "Size": [P(size)]},
                    objects={"mapControls": props(autoZoom=b(True), showStylePicker=b(False), showNavigationControls=b(False),
                                                  defaultStyle=s(S.map_style), showLabels=b(True)),
                             "bubbleLayer": props(bubbleRadius=i(9), maxRadius=i(34), strokeColor=color(S.panel), bubbleStrokeWidth=i(1)),
                             "dataPoint": [{"properties": {"fill": color(fill or S.primary)}}]},
                    vco=_title(title, sub), alt=title)


def about_page(page, blocks, definitions, credit, y0=None):
    """Three method panels + a definitions panel, with presentation-size type."""
    y0 = y0 if y0 is not None else S.title_y + 120
    full = S.cw
    col_w = (full - 2 * GUT) // 3
    bh = 400
    for k, (head, lines) in enumerate(blocks):
        x = S.x0 + k * (col_w + GUT)
        panel(page, f"blk{k}", x, y0, col_w, bh)
        shape(page, f"blk{k}-accent", x, y0, 6 if S.accent_bar != "Top" else col_w, bh if S.accent_bar != "Top" else 4,
              S.accent if k == 1 else S.primary)
        paras = [[(head, 24, S.primary if not S.dark else S.ink, True, S.font_title)], [(" ", 10, S.ink, False)]]
        for t in lines:
            paras += [[("•  " + t, 17, S.ink, False)], [(" ", 6, S.ink, False)]]
        textbox(page, f"blk{k}-t", x + 32, y0 + 26, col_w - 60, bh - 40, paras)
    y1 = y0 + bh + GUT
    h1 = S.foot_y - 20 - y1
    panel(page, "def", S.x0, y1, full, h1)
    paras = [[("Key definitions", 24, S.primary if not S.dark else S.ink, True, S.font_title)], [(" ", 10, S.ink, False)]]
    for term, text in definitions:
        paras += [[(term, 17, S.ink, True), ("   " + text, 17, S.ink, False)], [(" ", 5, S.ink, False)]]
    paras += [[(" ", 8, S.ink, False)], [(credit, 15, S.muted, False)]]
    textbox(page, "def-t", S.x0 + 32, y1 + 26, full - 64, h1 - 40, paras)
