#!/usr/bin/env python3
"""Convert every *.log under <src> into HTML inside <dst> and build an index."""

import html, pathlib, datetime, sys

src, dst = map(pathlib.Path, sys.argv[1:3])
dst.mkdir(parents=True, exist_ok=True)
stamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

# ---------------------------------------------------------------------------
# 1) turn each *.log into an HTML page at the same relative location
# ---------------------------------------------------------------------------
for log in src.rglob("*.log"):
    rel = log.relative_to(src)
    page = dst / rel.with_suffix(".html")
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(
        f"<h2>{rel}</h2><p>{stamp}</p><pre>{html.escape(log.read_text())}</pre>"
    )

# ---------------------------------------------------------------------------
# 2) build index.html with links **relative to `dst`**
# ---------------------------------------------------------------------------
links = "\n".join(
    f'<li><a href="{p.relative_to(dst).as_posix()}">'
    f'{p.relative_to(dst).as_posix()}</a></li>'
    for p in sorted(dst.rglob("*.html"))
    if p.name != "index.html"
)

(dst / "index.html").write_text(
    f"<h1>Buddy-Benchmark results</h1><ul>\n{links}\n</ul>"
)
