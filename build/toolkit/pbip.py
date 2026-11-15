"""Tiny, dependency-free generator for Power BI Projects (PBIP): TMDL semantic model + PBIR report.

Used by each portfolio repo's `build_pbip.py`. Deterministic IDs (uuid5 / sha1) keep diffs stable.
"""
import hashlib, json, os, shutil, uuid

SCHEMA = "https://developer.microsoft.com/json-schemas/fabric"


def uid(*parts):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "/".join(map(str, parts))))


def hexid(*parts, n=20):
    return hashlib.sha1("/".join(map(str, parts)).encode()).hexdigest()[:n]


def tq(name):
    """Quote a TMDL object name when needed."""
    plain = name.replace("_", "").isalnum() and name[0].isalpha()
    return name if plain else "'" + name.replace("'", "''") + "'"


def _w(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _json(path, obj):
    _w(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def _indent(text, tabs):
    return "\n".join(("\t" * tabs + line) if line.strip() else "" for line in text.strip("\n").split("\n"))


# ============================================================================================
# Semantic model (TMDL)
# ============================================================================================
def col(name, dtype="string", fmt=None, hidden=False, sort=None, cat=None, folder=None, desc=None,
        summarize="none", src=None, key=False):
    return dict(name=name, dtype=dtype, fmt=fmt, hidden=hidden, sort=sort, cat=cat, folder=folder, desc=desc,
                summarize=summarize, src=src or name, key=key)


def meas(name, expr, fmt=None, folder=None, desc=None):
    return dict(name=name, expr=expr, fmt=fmt, folder=folder, desc=desc)


def _doc(desc, tabs):
    if not desc:
        return ""
    return "".join("\t" * tabs + "/// " + line + "\n" for line in desc.strip().split("\n"))


def _measure_tmdl(m):
    out = _doc(m["desc"], 1)
    expr = m["expr"].strip("\n")
    if "\n" in expr:
        out += f"\tmeasure {tq(m['name'])} =\n{_indent(expr, 3)}\n"
    else:
        out += f"\tmeasure {tq(m['name'])} = {expr}\n"
    if m["fmt"]:
        out += f"\t\tformatString: {m['fmt']}\n"
    if m["folder"]:
        out += f"\t\tdisplayFolder: {m['folder']}\n"
    return out + "\n"


def _column_tmdl(c, calculated_src=False):
    out = _doc(c["desc"], 1) + f"\tcolumn {tq(c['name'])}\n"
    out += f"\t\tdataType: {c['dtype']}\n"
    if c["key"]:
        out += "\t\tisKey\n"
    if c["hidden"]:
        out += "\t\tisHidden\n"
    if c["fmt"]:
        out += f"\t\tformatString: {c['fmt']}\n"
    if c["cat"]:
        out += f"\t\tdataCategory: {c['cat']}\n"
    if c["folder"]:
        out += f"\t\tdisplayFolder: {c['folder']}\n"
    out += f"\t\tsummarizeBy: {c['summarize']}\n"
    src = c["src"]
    out += f"\t\tsourceColumn: {'[' + src + ']' if calculated_src else src}\n"
    if c["sort"]:
        out += f"\t\tsortByColumn: {tq(c['sort'])}\n"
    if c["dtype"] == "dateTime":
        out += "\n\t\tannotation UnderlyingDateTimeDataType = Date\n"
    return out + "\n"


def m_table(name, file, columns, measures=(), desc=None, hidden=False, extra_steps=""):
    """An import table loaded from a CSV through the shared fnLoadCsv function."""
    types = {"string": "type text", "int64": "Int64.Type", "double": "type number", "decimal": "Currency.Type",
             "dateTime": "type date", "boolean": "type logical"}
    src_cols = [c for c in columns if not c.get("computed")]
    typed = ", ".join(f'{{"{c["src"]}", {types[c["dtype"]]}}}' for c in src_cols)
    m = (f'let\n    Source = fnLoadCsv("{file}"),\n    Typed = Table.TransformColumnTypes(Source, {{{typed}}}, "en-US")'
         + (",\n" + extra_steps.strip("\n") if extra_steps else "") + f'\nin\n    {"Result" if extra_steps else "Typed"}')
    return dict(name=name, kind="m", source=m, columns=columns, measures=list(measures), desc=desc, hidden=hidden)


def calc_table(name, dax, columns, measures=(), desc=None, hidden=False, is_date=False):
    return dict(name=name, kind="calculated", source=dax, columns=columns, measures=list(measures), desc=desc,
                hidden=hidden, is_date=is_date)


def calc_group(name, items, desc=None, precedence=10, column=None):
    """items: [(name, dax, format_string_dax_or_None)] in display order."""
    NL, T = chr(10), chr(9)
    out = f"table {tq(name)}" + NL + NL + _doc(desc, 1) + T + "calculationGroup" + NL + T * 2 + f"precedence: {precedence}" + NL + NL
    for nm, expr, fmt in items:
        expr = expr.strip(NL)
        if NL in expr:
            out += T * 2 + f"calculationItem {tq(nm)} =" + NL + _indent(expr, 4) + NL + NL
        else:
            out += T * 2 + f"calculationItem {tq(nm)} = {expr}" + NL + NL
        if fmt:
            out += T * 3 + f"formatStringDefinition = {fmt}" + NL + NL
    out += T + f"column {tq(column or name)}" + NL + T * 2 + "dataType: string" + NL + NL
    out += T + f"partition {tq('Partition_' + name)} = calculationGroup" + NL
    return dict(name=name, kind="raw", text=out)


def field_param(name, fields, desc=None):
    """Field parameter table. fields: [(display name, "'Table'[Column]")] in order (format exported from Desktop)."""
    NL, T = chr(10), chr(9)
    q = tq(name)
    out = _doc(desc, 0) + f"table {q}" + NL + NL
    out += T + f"column {tq(name + ' Order')}" + NL + T * 2 + "dataType: int64" + NL + T * 2 + "isHidden" + NL + T * 2 + "formatString: 0" + NL
    out += T * 2 + "summarizeBy: sum" + NL + T * 2 + "sourceColumn: [Value3]" + NL + NL
    out += T + f"column {tq(name + ' Fields')}" + NL + T * 2 + "dataType: string" + NL + T * 2 + "isHidden" + NL + T * 2 + "summarizeBy: none" + NL
    out += T * 2 + "sourceColumn: [Value2]" + NL + T * 2 + f"sortByColumn: {tq(name + ' Order')}" + NL + NL
    out += T * 2 + 'extendedProperty ParameterMetadata = {"version":3,"kind":2}' + NL + NL
    out += T + f"column {q}" + NL + T * 2 + "dataType: string" + NL + T * 2 + "summarizeBy: none" + NL + T * 2 + "sourceColumn: [Value1]" + NL
    out += T * 2 + f"sortByColumn: {tq(name + ' Order')}" + NL + NL + T * 2 + "relatedColumnDetails" + NL + T * 3 + f"groupByColumn: {tq(name + ' Fields')}" + NL + NL
    rows = (", " + NL).join(f'    ("{d}", NAMEOF({ref}), {k})' for k, (d, ref) in enumerate(fields))
    out += T + f"partition {q} = calculated" + NL + T * 2 + "mode: import" + NL + T * 2 + "source =" + NL
    out += _indent("{" + NL + rows + NL + "}", 4) + NL
    return dict(name=name, kind="raw", text=out)


def measure_table(name, measures, desc=None):
    return dict(name=name, kind="measures", source="{BLANK()}", measures=list(measures), desc=desc,
                columns=[col("Value", "string", hidden=True, src="Value")])


def write_table(path, t):
    if t["kind"] == "raw":
        _w(path, t["text"])
        return
    out = _doc(t.get("desc"), 0) + f"table {tq(t['name'])}\n"
    if t.get("hidden"):
        out += "\tisHidden\n"
    if t.get("is_date"):
        out += "\tdataCategory: Time\n"
    out += "\n"
    for m in t["measures"]:
        out += _measure_tmdl(m)
    calc = t["kind"] in ("calculated", "measures")
    for c in t["columns"]:
        out += _column_tmdl(c, calculated_src=calc)
    pname = tq(t["name"])
    if t["kind"] == "m":
        out += f"\tpartition {pname} = m\n\t\tmode: import\n\t\tsource =\n{_indent(t['source'], 4)}\n\n"
    else:
        out += f"\tpartition {pname} = calculated\n\t\tmode: import\n\t\tsource =\n{_indent(t['source'], 4)}\n\n"
    if t["kind"] == "m":
        out += "\tannotation PBI_ResultType = Table\n"
    _w(path, out)


def rel(from_tbl, from_col, to_tbl, to_col, active=True, both=False):
    return dict(f=(from_tbl, from_col), t=(to_tbl, to_col), active=active, both=both)


def write_model(root, name, tables, relationships, expressions, desc=None, culture="en-US", functions=()):
    """root = <Name>.SemanticModel folder."""
    d = os.path.join(root, "definition")
    if os.path.isdir(d):
        shutil.rmtree(d)
    _json(os.path.join(root, "definition.pbism"), {
        "$schema": f"{SCHEMA}/item/semanticModel/definitionProperties/1.0.0/schema.json", "version": "4.2", "settings": {}})
    _json(os.path.join(root, ".platform"), {
        "$schema": f"{SCHEMA}/gitIntegration/platformProperties/2.0.0/schema.json",
        "metadata": {"type": "SemanticModel", "displayName": name}, "config": {"version": "2.0", "logicalId": uid(name, "model")}})
    _w(os.path.join(d, "database.tmdl"), "database\n\tcompatibilityLevel: 1702\n")
    refs = "".join(f"ref table {tq(t['name'])}\n" for t in tables)
    model = (_doc(desc, 0) + f"model Model\n\tculture: {culture}\n\tdefaultPowerBIDataSourceVersion: powerBI_V3\n"
             "\tdiscourageImplicitMeasures\n\tsourceQueryCulture: en-US\n\tdataAccessOptions\n\t\tlegacyRedirects\n\t\treturnErrorValuesAsNull\n\n"
             "annotation __PBI_TimeIntelligenceEnabled = 0\n\nannotation PBI_ProTooling = [\"DevMode\"]\n\n" + refs)
    _w(os.path.join(d, "model.tmdl"), model)
    _w(os.path.join(d, "expressions.tmdl"), "\n".join(expressions) + "\n")
    for t in tables:
        write_table(os.path.join(d, "tables", t["name"] + ".tmdl"), t)
    out = ""
    for r in relationships:
        (ft, fc), (tt, tc) = r["f"], r["t"]
        out += f"relationship {uid(name, ft, fc, tt, tc)}\n"
        if r["both"]:
            out += "\tcrossFilteringBehavior: bothDirections\n"
        if not r["active"]:
            out += "\tisActive: false\n"
        out += f"\tfromColumn: {tq(ft)}.{tq(fc)}\n\ttoColumn: {tq(tt)}.{tq(tc)}\n\n"
    _w(os.path.join(d, "relationships.tmdl"), out)
    if functions:
        fx = ""
        for f in functions:
            fx += _doc(f.get("desc"), 0) + f"function {tq(f['name'])} =\n{_indent(f['body'], 2)}\n\n"
        _w(os.path.join(d, "functions.tmdl"), fx)


def csv_expressions(repo, local_folder, default="GitHub"):
    """DataSource switch + a CSV loader that reads from GitHub raw (refreshable in the Service) or a local folder."""
    return [
        f'/// Where the CSVs are read from: "GitHub" (public raw files, no setup) or "Local" (LocalDataFolder).\n'
        f'expression DataSource = "{default}" meta [IsParameterQuery=true, List={{"GitHub", "Local"}}, DefaultValue="GitHub", Type="Text", IsParameterQueryRequired=true]\n'
        "\tannotation PBI_ResultType = Text\n",
        f'/// Folder holding the CSVs when DataSource = "Local". Must end with a backslash.\n'
        f'expression LocalDataFolder = "{local_folder}" meta [IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]\n'
        "\tannotation PBI_ResultType = Text\n",
        "/// Loads one CSV from the repo's data/ folder. The GitHub branch uses a static base URL + RelativePath,\n"
        "/// so the Power BI Service can refresh it without a gateway.\n"
        "expression fnLoadCsv =\n"
        + _indent(
            "(file as text) as table =>\n"
            "let\n"
            "    Bytes = if DataSource = \"Local\"\n"
            "        then File.Contents(LocalDataFolder & file)\n"
            f"        else Web.Contents(\"https://raw.githubusercontent.com\", [RelativePath = \"ayman-stat/{repo}/main/data/\" & file]),\n"
            "    Csv = Csv.Document(Bytes, [Delimiter = \",\", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),\n"
            "    Promoted = Table.PromoteHeaders(Csv, [PromoteAllScalars = true])\n"
            "in\n"
            "    Promoted", 2)
        + "\n\tannotation PBI_ResultType = Function\n",
    ]


# ============================================================================================
# Report (PBIR)
# ============================================================================================
def lit(v):
    return {"expr": {"Literal": {"Value": v}}}


def s(text):
    return lit("'" + str(text).replace("'", "''") + "'")


def b(v):
    return lit("true" if v else "false")


def n(v):
    return lit(f"{v}D")


def i(v):
    return lit(f"{int(v)}L")


def color(hexv):
    return {"solid": {"color": lit(f"'{hexv}'")}}


def C(entity, prop):
    return {"Column": {"Expression": {"SourceRef": {"Entity": entity}}, "Property": prop}}


def M(entity, prop):
    return {"Measure": {"Expression": {"SourceRef": {"Entity": entity}}, "Property": prop}}


def proj(field, display=None, active=None):
    kind = "Column" if "Column" in field else "Measure"
    e, p = field[kind]["Expression"]["SourceRef"]["Entity"], field[kind]["Property"]
    out = {"field": field, "queryRef": f"{e}.{p}", "nativeQueryRef": display or p}
    if display:
        out["displayName"] = display
    if active is not None:
        out["active"] = active
    return out


def qref(field):
    kind = "Column" if "Column" in field else "Measure"
    return f"{field[kind]['Expression']['SourceRef']['Entity']}.{field[kind]['Property']}"


def props(**kw):
    return [{"properties": kw}]


def idprops(**kw):
    return [{"properties": kw, "selector": {"id": "default"}}]


class Page:
    def __init__(self, report, key, display, width=1920, height=1080, bg="#F4F6F9"):
        self.report, self.name, self.display = report, hexid(report.name, key), display
        self.width, self.height, self.bg, self.visuals, self.z = width, height, bg, [], 1000
        self.filters = []
        self.interactions = []  # (source visual name, target visual name, type)

    def group(self, key, display, x, y, w, h, hidden=False, parent=None):
        """A visual group container (children pass parent=<returned name>). Positions of children are page-absolute."""
        name = hexid(self.name, key)
        self.groups = getattr(self, "groups", {})
        self.groups[name] = (x, y)
        cont = {"$schema": f"{SCHEMA}/item/report/definition/visualContainer/2.9.0/schema.json", "name": name,
                "position": {"x": x, "y": y, "z": self.z, "height": h, "width": w, "tabOrder": self.z},
                "visualGroup": {"displayName": display, "groupMode": "ScaleMode"}}
        if hidden:
            cont["isHidden"] = True
        if parent:
            cont["parentGroupName"] = parent
        self.visuals.append(cont)
        self.z += 100
        return name

    def add(self, key, vtype, x, y, w, h, query=None, objects=None, vco=None, title=None, subtitle=None,
            sort=None, alt=None, extra=None, filters=None, field_params=None, parent=None, hidden=False):
        """field_params: {role: parameter-column field}; the role's projections then hold the parameter's default field."""
        v = {"visualType": vtype}
        if query:
            q = {"queryState": {role: {"projections": ps} for role, ps in query.items()}}
            for role, pf in (field_params or {}).items():
                q["queryState"][role]["fieldParameters"] = [{"parameterExpr": pf, "index": 0, "length": len(query[role])}]
            if sort:
                q["sortDefinition"] = {"sort": [{"field": f, "direction": d} for f, d in sort], "isDefaultSort": True}
            v["query"] = q
        if objects:
            v["objects"] = objects
        vco = dict(vco or {})
        if title is not None:
            vco["title"] = props(show=b(True), text=s(title))
        if subtitle:
            vco["subTitle"] = props(show=b(True), text=s(subtitle))
        if vco:
            v["visualContainerObjects"] = vco
        if extra:
            v.update(extra)
        name = hexid(self.name, key)
        if parent:                                           # PBIR: positions inside a group are relative to the group
            gx, gy = getattr(self, "groups", {}).get(parent, (0, 0))
            x, y = x - gx, y - gy
        if alt and "general" not in v.get("visualContainerObjects", {}):
            v.setdefault("visualContainerObjects", {})["general"] = props(altText=s(alt))
        cont = {"$schema": f"{SCHEMA}/item/report/definition/visualContainer/2.9.0/schema.json",
                "name": name, "position": {"x": x, "y": y, "z": self.z, "height": h, "width": w, "tabOrder": self.z},
                "visual": v}
        if filters:
            cont["filterConfig"] = {"filters": filters}
        if parent:
            cont["parentGroupName"] = parent
        if hidden:
            cont["isHidden"] = True
        self.visuals.append(cont)
        self.z += 100
        return name

    def members(self, group_name):
        """(name, visualType) of every visual inside a group (one level)."""
        return [(v["name"], v["visual"]["visualType"]) for v in self.visuals if v.get("parentGroupName") == group_name and "visual" in v]


class Report:
    def __init__(self, root, name, model_folder, theme):
        self.root, self.name, self.model_folder, self.theme, self.pages = root, name, model_folder, theme, []
        self.bookmarks = []
        self.n_pages = 5

    def bookmark_names(self, page, group):
        return hexid(self.name, page.name, group, "show"), hexid(self.name, page.name, group, "hide")

    def toggle_bookmarks(self, page, group, label):
        """Two display-only bookmarks (show / hide a group) that never touch data or other visuals.
        Returns (show_name, hide_name) for button actions."""
        out = []
        for show in (True, False):
            bname = self.bookmark_names(page, group)[0 if show else 1]
            vis = {n: {"singleVisual": dict({"visualType": t}, **({} if show else {"display": {"mode": "hidden"}}))}
                   for n, t in page.members(group)}
            self.bookmarks.append({
                "$schema": f"{SCHEMA}/item/report/definition/bookmark/2.1.0/schema.json",
                "displayName": f"{label} - {'open' if show else 'close'} ({page.display})", "name": bname,
                "options": {"targetVisualNames": [group] + list(vis), "applyOnlyToTargetVisuals": True, "suppressData": True,
                            "suppressActiveSection": True},
                "explorationState": {"version": "1.3", "activeSection": page.name,
                                     "sections": {page.name: {"visualContainers": vis,
                                                              "visualContainerGroups": {group: {"isHidden": not show}}}}}})
            out.append(bname)
        return tuple(out)

    def page(self, key, display, **kw):
        p = Page(self, key, display, **kw)
        self.pages.append(p)
        return p

    def write(self):
        d = os.path.join(self.root, "definition")
        if os.path.isdir(d):
            shutil.rmtree(d)
        _json(os.path.join(self.root, "definition.pbir"), {
            "$schema": f"{SCHEMA}/item/report/definitionProperties/2.0.0/schema.json", "version": "4.0",
            "datasetReference": {"byPath": {"path": f"../{self.model_folder}"}}})
        _json(os.path.join(self.root, ".platform"), {
            "$schema": f"{SCHEMA}/gitIntegration/platformProperties/2.0.0/schema.json",
            "metadata": {"type": "Report", "displayName": self.name}, "config": {"version": "2.0", "logicalId": uid(self.name, "report")}})
        _json(os.path.join(d, "version.json"), {"$schema": f"{SCHEMA}/item/report/definition/versionMetadata/1.0.0/schema.json", "version": "2.0.0"})
        # cache-busting: Desktop caches themes by name, so the file name carries a hash of the content.
        # The validator requires the theme's internal "name" to equal the registered file name.
        body = {k: v for k, v in self.theme.items() if k != "name"}
        theme_file = f"{self.theme['name']}-{hexid(json.dumps(body, sort_keys=True), n=8)}.json"
        res = os.path.join(self.root, "StaticResources", "RegisteredResources")
        if os.path.isdir(res):
            shutil.rmtree(res)
        _json(os.path.join(res, theme_file), dict(self.theme, name=theme_file))
        base_src = os.path.join(os.path.dirname(__file__), "CY24SU10.json")
        os.makedirs(os.path.join(self.root, "StaticResources", "SharedResources", "BaseThemes"), exist_ok=True)
        shutil.copy(base_src, os.path.join(self.root, "StaticResources", "SharedResources", "BaseThemes", "CY24SU10.json"))
        _json(os.path.join(d, "report.json"), {
            "$schema": f"{SCHEMA}/item/report/definition/report/3.0.0/schema.json",
            "themeCollection": {
                "baseTheme": {"name": "CY24SU10", "reportVersionAtImport": {"visual": "1.8.95", "report": "2.0.95", "page": "1.3.95"}, "type": "SharedResources"},
                "customTheme": {"name": theme_file, "reportVersionAtImport": {"visual": "1.8.95", "report": "2.0.95", "page": "1.3.95"}, "type": "RegisteredResources"}},
            "objects": {"outspacePane": [{"properties": {"expanded": b(False), "visible": b(True)}}]},
            "resourcePackages": [
                {"name": "SharedResources", "type": "SharedResources", "items": [{"name": "CY24SU10", "path": "BaseThemes/CY24SU10.json", "type": "BaseTheme"}]},
                {"name": "RegisteredResources", "type": "RegisteredResources", "items": [{"name": theme_file, "path": theme_file, "type": "CustomTheme"}]}],
            "settings": {"useStylableVisualContainerHeader": True, "exportDataMode": "AllowSummarized", "defaultDrillFilterOtherVisuals": True,
                         "allowChangeFilterTypes": True, "useEnhancedTooltips": True, "useDefaultAggregateDisplayName": True}})
        _json(os.path.join(d, "pages", "pages.json"), {
            "$schema": f"{SCHEMA}/item/report/definition/pagesMetadata/1.0.0/schema.json",
            "pageOrder": [p.name for p in self.pages], "activePageName": self.pages[0].name})
        for p in self.pages:
            pj = {"$schema": f"{SCHEMA}/item/report/definition/page/2.0.0/schema.json", "name": p.name, "displayName": p.display,
                  "displayOption": "FitToPage", "height": p.height, "width": p.width,
                  "objects": {"background": [{"properties": {"color": color(p.bg), "transparency": n(0)}}],
                              "outspace": [{"properties": {"color": color(p.bg)}}]}}
            if p.filters:
                pj["filterConfig"] = {"filters": p.filters}
            if p.interactions:
                pj["visualInteractions"] = [{"source": a, "target": t, "type": k} for a, t, k in p.interactions]
            _json(os.path.join(d, "pages", p.name, "page.json"), pj)
            for v in p.visuals:
                _json(os.path.join(d, "pages", p.name, "visuals", v["name"], "visual.json"), v)
        if self.bookmarks:
            for bm in self.bookmarks:
                _json(os.path.join(d, "bookmarks", f"{bm['name']}.bookmark.json"), bm)
            _json(os.path.join(d, "bookmarks", "bookmarks.json"), {
                "$schema": f"{SCHEMA}/item/report/definition/bookmarksMetadata/1.0.0/schema.json",
                "items": [{"name": bm["name"]} for bm in self.bookmarks]})


def write_pbip(path, report_folder):
    _json(path, {"$schema": f"{SCHEMA}/pbip/pbipProperties/1.0.0/schema.json", "version": "1.0",
                 "artifacts": [{"report": {"path": report_folder}}], "settings": {"enableAutoRecovery": True}})
