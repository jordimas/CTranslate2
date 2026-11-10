"""Parse .prof dumps + manifest.json, emit a sortable reports_<cpu_id>.html.

Structure:
  1. Global summary (across all tasks & compute types) — by op, by (op, shape)
  2. Summary per compute_type (across tasks) — by op, by (op, shape)
  3. Per (task, compute_type) breakdown — raw profiler rows

Usage: python summarize.py --results results --cpu-id apple_m4_pro
"""

import argparse
import collections
import html
import json
import os
import pathlib
import re
import sys

from machine_info import cpu_id

ROW_RE = re.compile(
    r"^\s*(\d+\.\d+)%\s+(\d+\.\d+)%\s+(\d+\.\d+)%\s+"
    r"(\S+(?:\[[^\]]*\])?)"
    r"\s+(\d+(?:\.\d+)?)ms\s+calls=(\d+)"
)
SCOPE_RE = re.compile(r"^([^\[]+)(?:\[([^\]]*)\])?$")


def parse_prof(path):
    rows = []
    with open(path) as f:
        for line in f:
            m = ROW_RE.match(line)
            if not m:
                continue
            self_pct = float(m.group(1))
            incl_pct = float(m.group(2))
            raw_name = m.group(4)
            ms = float(m.group(5))
            calls = int(m.group(6))
            s = SCOPE_RE.match(raw_name)
            op, shape = (s.group(1), s.group(2) or "") if s else (raw_name, "")
            rows.append({
                "op": op, "shape": shape,
                "self_pct": self_pct, "incl_pct": incl_pct,
                "ms": ms, "calls": calls,
                "mean_ms": ms / calls if calls else 0.0,
            })
    return rows


CSS = """
<style>
body { font-family: -apple-system, system-ui, sans-serif; margin: 24px; color: #222; }
h1 { font-size: 20px; }
h2 { font-size: 17px; margin-top: 36px; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
h3 { font-size: 14px; color: #555; margin-top: 20px; }
h4 { font-size: 13px; color: #666; margin-top: 14px; }
table { border-collapse: collapse; margin: 6px 0 20px; font-size: 12px; }
th, td { padding: 4px 10px; border-bottom: 1px solid #e4e4e4; text-align: left;
         white-space: nowrap; }
th { background: #f5f5f5; cursor: pointer; }
tr:hover td { background: #fafafa; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.failed { color: #a00; }
nav { position: sticky; top: 0; background: white; padding: 8px 0;
      border-bottom: 1px solid #ddd; font-size: 12px; z-index: 10;
      max-height: 120px; overflow-y: auto; }
nav a { margin-right: 10px; white-space: nowrap; display: inline-block; }
.shape { font-family: ui-monospace, Menlo, monospace; color: #555; }
.meta { color: #666; font-size: 12px; margin-bottom: 8px; }
details { margin: 8px 0 20px; }
details > summary { cursor: pointer; font-weight: 600; font-size: 13px;
                     padding: 4px 0; }
.section { margin-left: 12px; }
</style>
"""

TABLE_JS = """
<script>
function sortTable(table, col, numeric) {
  const tb = table.tBodies[0];
  const rows = Array.from(tb.rows);
  const asc = table.getAttribute('data-sort-col') != col
            || table.getAttribute('data-sort-dir') == 'desc';
  rows.sort((a, b) => {
    let va = a.cells[col].innerText, vb = b.cells[col].innerText;
    if (numeric) { va = parseFloat(va)||0; vb = parseFloat(vb)||0; return asc?va-vb:vb-va; }
    return asc ? va.localeCompare(vb) : vb.localeCompare(va);
  });
  rows.forEach(r => tb.appendChild(r));
  table.setAttribute('data-sort-col', col);
  table.setAttribute('data-sort-dir', asc ? 'asc' : 'desc');
}
document.querySelectorAll('table.sortable').forEach(t => {
  Array.from(t.tHead.rows[0].cells).forEach((th, i) => {
    th.onclick = () => sortTable(t, i, th.dataset.type == 'num');
  });
});
</script>
"""


def table_html(headers, rows, numeric_cols):
    th = "".join(
        f'<th data-type="{"num" if i in numeric_cols else "str"}">{html.escape(h)}</th>'
        for i, h in enumerate(headers))
    body_lines = []
    for r in rows:
        cells = []
        for i, v in enumerate(r):
            cls = "num" if i in numeric_cols else ""
            if cls == "num" and isinstance(v, float):
                text = f"{v:.3f}" if abs(v) < 1000 else f"{v:.1f}"
            else:
                text = str(v)
            cells.append(f'<td class="{cls}">{html.escape(text)}</td>')
        body_lines.append("<tr>" + "".join(cells) + "</tr>")
    return (f'<table class="sortable"><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(body_lines)}</tbody></table>')


def rollup_by_op(prof_rows_list):
    """Collapse multiple runs' rows by op (dropping shape)."""
    agg = collections.defaultdict(lambda: {"ms": 0.0, "calls": 0})
    for rows in prof_rows_list:
        for r in rows:
            a = agg[r["op"]]
            a["ms"] += r["ms"]
            a["calls"] += r["calls"]
    total = sum(v["ms"] for v in agg.values()) or 1.0
    out = [(op, v["calls"], v["ms"], v["ms"] / v["calls"] if v["calls"] else 0.0,
            100.0 * v["ms"] / total)
           for op, v in agg.items()]
    return sorted(out, key=lambda r: r[2], reverse=True)


def rollup_by_op_shape(prof_rows_list, top_n=None):
    """Collapse by (op, shape)."""
    agg = collections.defaultdict(lambda: {"ms": 0.0, "calls": 0, "runs": 0})
    for rows in prof_rows_list:
        for r in rows:
            k = (r["op"], r["shape"])
            agg[k]["ms"] += r["ms"]
            agg[k]["calls"] += r["calls"]
            agg[k]["runs"] += 1
    total = sum(v["ms"] for v in agg.values()) or 1.0
    out = [(op, shape, v["calls"], v["ms"],
            v["ms"] / v["calls"] if v["calls"] else 0.0,
            100.0 * v["ms"] / total, v["runs"])
           for (op, shape), v in agg.items()]
    out = sorted(out, key=lambda r: r[3], reverse=True)
    if top_n:
        out = out[:top_n]
    return out


def op_table(rows):
    headers = ["op", "calls", "total ms", "mean ms", "share %"]
    return table_html(headers, rows, numeric_cols={1, 2, 3, 4})


def op_shape_table(rows):
    headers = ["op", "shape", "calls", "total ms", "mean ms", "share %",
               "appears in N runs"]
    return table_html(headers, rows, numeric_cols={2, 3, 4, 5, 6})


def shape_diversity(prof_rows_list):
    """Per-op: distinct shape count, instrumented share, total time."""
    agg = collections.defaultdict(
        lambda: {"ms": 0.0, "calls": 0, "ms_with_shape": 0.0, "shapes": set()})
    for rows in prof_rows_list:
        for r in rows:
            a = agg[r["op"]]
            a["ms"] += r["ms"]
            a["calls"] += r["calls"]
            if r["shape"]:
                a["ms_with_shape"] += r["ms"]
                a["shapes"].add(r["shape"])
    total = sum(v["ms"] for v in agg.values()) or 1.0
    out = []
    for op, v in agg.items():
        pct_shaped = (100.0 * v["ms_with_shape"] / v["ms"]) if v["ms"] else 0.0
        out.append((op, len(v["shapes"]), v["calls"], v["ms"],
                    100.0 * v["ms"] / total, pct_shaped))
    return sorted(out, key=lambda r: r[3], reverse=True)


def shape_diversity_table(rows):
    headers = ["op", "distinct shapes", "calls", "total ms", "share %",
               "% time with shape"]
    return table_html(headers, rows, numeric_cols={1, 2, 3, 4, 5})


def top_shapes_global(prof_rows_list, top_n=50):
    """Shape across all ops: which shape burns the most time?"""
    agg = collections.defaultdict(lambda: {"ms": 0.0, "calls": 0, "ops": set()})
    for rows in prof_rows_list:
        for r in rows:
            if not r["shape"]:
                continue
            a = agg[r["shape"]]
            a["ms"] += r["ms"]
            a["calls"] += r["calls"]
            a["ops"].add(r["op"])
    out = [(shape, ",".join(sorted(v["ops"])), v["calls"], v["ms"])
           for shape, v in agg.items()]
    out = sorted(out, key=lambda r: r[3], reverse=True)[:top_n]
    return out


def top_shapes_table(rows):
    headers = ["shape", "ops", "calls", "total ms"]
    return table_html(headers, rows, numeric_cols={2, 3})


def per_run_table(rows):
    headers = ["op", "shape", "calls", "self ms", "mean ms", "self %", "incl %"]
    sorted_rows = sorted(rows, key=lambda r: r["ms"], reverse=True)
    body = [(r["op"], r["shape"], r["calls"], r["ms"], r["mean_ms"],
             r["self_pct"], r["incl_pct"]) for r in sorted_rows]
    return table_html(headers, body, numeric_cols={2, 3, 4, 5, 6})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results",
                   default=str(pathlib.Path(__file__).parent / "results"))
    p.add_argument("--out", help="Override the reports_<cpu_id>.html output path")
    p.add_argument("--cpu-id", help="Benchmark CPU identifier (overrides run metadata)")
    p.add_argument("--json-out",
                   default=str(pathlib.Path(__file__).parent / "summary.json"))
    a = p.parse_args()
    if not a.out:
        identifier = a.cpu_id
        metadata_path = pathlib.Path(a.results) / "run-metadata.json"
        if not identifier and metadata_path.is_file():
            identifier = json.loads(metadata_path.read_text()).get("cpu_id")
        if not identifier:
            p.error("Results lack cpu_id; supply --cpu-id for the benchmark machine or --out")
        a.out = str(pathlib.Path(__file__).parent / f"reports_{cpu_id(identifier)}.html")

    # Re-build manifest from .json files on disk (resume-safe).
    manifest = []
    for fname in sorted(os.listdir(a.results)):
        if not fname.endswith(".json") or fname in ("manifest.json", "run-metadata.json"):
            continue
        with open(os.path.join(a.results, fname)) as f:
            manifest.append(json.load(f))

    rows_by_tag = {}
    for e in manifest:
        if e.get("status") != "ok":
            continue
        tag = f"{e['task']}__{e['device']}__{e['compute_type']}"
        # Prefer the supplied results directory, including after relocation.
        prof_path = pathlib.Path(a.results) / f"{tag}.prof"
        if not prof_path.is_file():
            raise FileNotFoundError(f"Missing profile for successful run: {prof_path}")
        rows = parse_prof(prof_path)
        if not rows:
            raise ValueError(f"No call-count profiling rows found in {prof_path}")
        rows_by_tag[tag] = rows

    with open(a.json_out, "w") as f:
        json.dump({"manifest": manifest, "rows_by_tag": rows_by_tag},
                  f, indent=2)

    ok = [e for e in manifest if e.get("status") == "ok"]
    failed = [e for e in manifest if e.get("status") != "ok"]

    # Group runs by compute_type.
    by_ct = collections.defaultdict(list)
    for e in ok:
        tag = f"{e['task']}__{e['device']}__{e['compute_type']}"
        by_ct[e["compute_type"]].append((e, rows_by_tag.get(tag, [])))

    # 1. Global.
    all_rows = [rows_by_tag[f"{e['task']}__{e['device']}__{e['compute_type']}"]
                for e in ok
                if f"{e['task']}__{e['device']}__{e['compute_type']}" in rows_by_tag]

    sections = []
    sections.append('<h2 id="global">Global summary (all tasks × compute types)</h2>')
    sections.append(f'<div class="meta">{len(ok)} runs · {len(failed)} failed</div>')
    sections.append('<h3>By op</h3>')
    sections.append(op_table(rollup_by_op(all_rows)))
    sections.append('<h3>Top 100 (op, shape)</h3>')
    sections.append(op_shape_table(rollup_by_op_shape(all_rows, top_n=100)))
    sections.append('<h3>Shape instrumentation per op</h3>')
    sections.append('<div class="meta">Ops with <code>% time with shape</code> &lt; 100% '
                    'are not yet fully annotated — adding <code>PROFILE_WITH_SHAPES</code> '
                    'to them would increase coverage.</div>')
    sections.append(shape_diversity_table(shape_diversity(all_rows)))
    sections.append('<h3>Top 50 shapes globally (any op)</h3>')
    sections.append(top_shapes_table(top_shapes_global(all_rows, top_n=50)))

    if failed:
        sections.append('<h3 class="failed">Failed runs</h3><ul>')
        for e in failed:
            sections.append(
                f'<li class="failed">{html.escape(e["task"])} / '
                f'{html.escape(e["device"])} / '
                f'{html.escape(e["compute_type"])}: '
                f'{html.escape(e.get("error",""))}</li>')
        sections.append("</ul>")

    # 2. Per compute_type.
    sections.append('<h2 id="by-ct">Summary per compute_type</h2>')
    for ct in sorted(by_ct):
        entries = by_ct[ct]
        rows_list = [rows for _, rows in entries]
        tasks_in_ct = ", ".join(
            sorted({f"{e['task']}/{e['device']}" for e, _ in entries}))
        sections.append(f'<h3 id="ct-{html.escape(ct)}">{html.escape(ct)}</h3>')
        sections.append(
            f'<div class="meta">{len(entries)} runs: {html.escape(tasks_in_ct)}</div>')
        sections.append('<h4>By op</h4>')
        sections.append(op_table(rollup_by_op(rows_list)))
        sections.append('<details><summary>Top 50 (op, shape)</summary>')
        sections.append(op_shape_table(rollup_by_op_shape(rows_list, top_n=50)))
        sections.append('</details>')

    # 3. Per (task, compute_type).
    sections.append('<h2 id="per-run">Per (task, device, compute_type)</h2>')
    for e in sorted(ok, key=lambda e: (e["task"], e["device"], e["compute_type"])):
        tag = f"{e['task']}__{e['device']}__{e['compute_type']}"
        wall = e.get("metrics", {}).get("wall_sec")
        meta = (f'wall={wall:.3f}s · '
                f'{len(rows_by_tag.get(tag, []))} scopes') if wall else ""
        sections.append(
            f'<details id="{html.escape(tag)}"><summary>'
            f'{html.escape(e["task"])} / {html.escape(e["device"])} / '
            f'{html.escape(e["compute_type"])} '
            f'<span class="meta">({html.escape(meta)})</span>'
            f'</summary><div class="section">')
        sections.append(per_run_table(rows_by_tag.get(tag, [])))
        sections.append('</div></details>')

    nav_links = (
        '<a href="#global"><b>global</b></a>'
        '<a href="#by-ct"><b>per compute_type</b></a>'
        '<a href="#per-run"><b>per run</b></a>'
    )

    doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>CT2 common operations profiling</title>"
        + CSS +
        "</head><body>"
        "<h1>CTranslate2 — common operations profiling</h1>"
        f"<nav>{nav_links}</nav>"
        + "".join(sections)
        + TABLE_JS
        + "</body></html>")
    with open(a.out, "w") as f:
        f.write(doc)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
