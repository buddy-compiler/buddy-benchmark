#!/usr/bin/env python3
"""
Turn every *.json under <src> into <dst>/<same-name>.html.
If a twin *.log exists (same stem), show it in a collapsible <details>.
If the JSON is unreadable, generate a red “FAILED” page instead of aborting.
"""

import html, json, pathlib, datetime, sys, traceback

class BrokenJSON(RuntimeError):
    pass

src, dst = map(pathlib.Path, sys.argv[1:3])
dst.mkdir(parents=True, exist_ok=True)
stamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

CSS = """
<style>
body{font-family:system-ui,Arial,sans-serif;margin:2rem;max-width:80ch}
table{border-collapse:collapse;margin:1rem 0}
th,td{border:1px solid #bbb;padding:.3rem .6rem;text-align:right}
th{text-align:center;background:#f0f0f0}
tr:nth-child(even){background:#fafafa}
details{border:1px solid #ccc;border-radius:.4rem;padding:.6rem}
summary{font-weight:600;cursor:pointer}
.err{border:2px solid #c00;background:#fee;padding:1rem;border-radius:.5rem}
</style>
"""

def gbench_json_to_table(js_path: pathlib.Path) -> str:
    """Turn one Google-Benchmark JSON file into an HTML <table>."""
    try:
        payload = json.loads(js_path.read_text())
    except json.JSONDecodeError as e:
        raise BrokenJSON(f"JSON parse error: {e.msg}") from e

    if "benchmarks" not in payload:
        raise BrokenJSON("Missing top-level ‘benchmarks’ array")

    data = payload["benchmarks"]
    if not data:
        raise BrokenJSON("Empty ‘benchmarks’ array")

    first = next((b for b in data if b.get("run_type") == "iteration"), None)
    if not first:
        raise BrokenJSON("No ‘iteration’ rows found")

    unit = html.escape(first.get("time_unit", "ns"))

    head = (f"<tr><th>Name</th><th>Time&nbsp;({unit})</th>"
            f"<th>CPU&nbsp;({unit})</th><th>Iterations</th></tr>")

    rows = "\n".join(
        f"<tr><td style='text-align:left'>{html.escape(b['name'])}</td>"
        f"<td>{b['real_time']:.3g}</td>"
        f"<td>{b['cpu_time']:.3g}</td>"
        f"<td>{b['iterations']:,}</td></tr>"
        for b in data
        if b.get("run_type") == "iteration"
    )
    return f"<h3>{js_path.name}</h3>\n<table>{head}\n{rows}</table>"

# ---------------------------------------------------------------------------

for js in src.rglob("*.json"):
    print("→ parsing", js)
    log  = js.with_suffix(".log")
    rel  = js.relative_to(src)
    page = dst / rel.with_suffix(".html")
    page.parent.mkdir(parents=True, exist_ok=True)

    body = [CSS, f"<h2>{rel}</h2><p><em>{stamp}</em></p>"]

    try:
        body.append(gbench_json_to_table(js))
    except (BrokenJSON, json.JSONDecodeError) as err:
        # Build a failure stub but keep the run going
        body.append(f"<div class='err'><strong>⚠ FAILED:</strong> "
                    f"{html.escape(str(err))}</div>")

    # Always embed the console log if available
    if log.exists():
        body.append("<details><summary>Console output</summary>\n"
                    f"<pre>{html.escape(log.read_text())}</pre></details>")

    page.write_text("\n".join(body))

# ---------------------------------------------------------------------------
# rebuild index
# ---------------------------------------------------------------------------
links = "\n".join(
    f'<li><a href="{p.relative_to(dst).as_posix()}">'
    f'{p.relative_to(dst).as_posix()}</a></li>'
    for p in sorted(dst.rglob("*.html"))
    if p.name != "index.html"
)
(dst / "index.html").write_text(
    CSS + "<h1>Buddy-Benchmark results</h1><ul>\n" + links + "\n</ul>")
