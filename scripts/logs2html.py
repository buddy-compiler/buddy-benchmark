#!/usr/bin/env python3
"""Turn every *.log under <src> into <dst>/<same-name>.html + an index.html."""
import html, pathlib, datetime, sys

src, dst = map(pathlib.Path, sys.argv[1:3])
dst.mkdir(parents=True, exist_ok=True)
stamp = datetime.datetime.utcnow().isoformat(' ', 'seconds')

for log in src.rglob("*.log"):
    rel = log.relative_to(src)
    page = dst / rel.with_suffix(".html")
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(f"<h2>{rel}</h2><p>{stamp} UTC</p><pre>{html.escape(log.read_text())}</pre>")

links = "\n".join(f'<li><a href="{p.as_posix()}">{p.as_posix()}</a></li>'
                  for p in sorted(dst.rglob("*.html")) if p.name != "index.html")
(dst / "index.html").write_text(f"<h1>Buddy-Benchmark results</h1><ul>{links}</ul>")
