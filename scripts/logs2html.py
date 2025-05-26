#!/usr/bin/env python3
"""
Turn every *.log under <src> into <dst>/<same-name>.html
If a sibling *.json produced by Google Benchmark exists, render
its numbers as an HTML table right under the log.
"""
import html, json, pathlib, datetime, sys

src, dst = map(pathlib.Path, sys.argv[1:3])
dst.mkdir(parents=True, exist_ok=True)
stamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

def gbench_json_to_table(js_path):
    data = json.loads(js_path.read_text())["benchmarks"]
    head = "<tr><th>Name</th><th>Time (ns)</th><th>CPU (ns)</th><th>Iterations</th></tr>"
    rows = "\n".join(
        f"<tr><td>{b['name']}</td><td>{b['real_time']:.1f}</td>"
        f"<td>{b['cpu_time']:.1f}</td><td>{b['iterations']}</td></tr>"
        for b in data if "name" in b
    )
    return f"<h3>Parsed numbers</h3><table>{head}{rows}</table>"

for log in src.rglob("*.log"):
    rel = log.relative_to(src)
    page = dst / rel.with_suffix(".html")
    page.parent.mkdir(parents=True, exist_ok=True)

    body = [f"<h2>{rel}</h2><p>{stamp}</p>",
            f"<pre>{html.escape(log.read_text())}</pre>"]

    json_peer = log.with_suffix(".json")
    if json_peer.exists():
        body.append(gbench_json_to_table(json_peer))

    page.write_text("\n".join(body))

# rebuild index
links = "\n".join(
    f'<li><a href="{p.relative_to(dst).as_posix()}">{p.relative_to(dst).as_posix()}</a></li>'
    for p in sorted(dst.rglob("*.html")) if p.name != "index.html"
)
(dst / "index.html").write_text(f"<h1>Buddy-Benchmark results</h1><ul>{links}</ul>")
