"""Self-contained HTML report assembly."""
from __future__ import annotations

import datetime
import html
from pathlib import Path
from typing import Iterable

REPORT_DIR = Path(__file__).parent / "reports"


def _header(title: str) -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 980px; margin: 1.5rem auto; padding: 0 1rem; color: #1a1a1a; }}
h1 {{ border-bottom: 2px solid #ccc; padding-bottom: 0.25rem; }}
h2 {{ margin-top: 2rem; border-left: 4px solid #4a7; padding-left: 0.5rem; }}
img {{ max-width: 100%; display: block; margin: 0.5rem 0; border: 1px solid #ddd; }}
table {{ border-collapse: collapse; margin: 0.5rem 0; }}
th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; font-variant-numeric: tabular-nums; }}
th {{ background: #f4f4f4; }}
.pass {{ background: #d7f7d0; color: #0b6c00; font-weight: bold; padding: 2px 8px; border-radius: 4px; }}
.fail {{ background: #ffd7d0; color: #8a0000; font-weight: bold; padding: 2px 8px; border-radius: 4px; }}
.meta {{ color: #666; font-size: 0.9em; }}
code, pre {{ background: #f6f6f6; padding: 1px 4px; border-radius: 3px; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="meta">Generated {datetime.datetime.now().isoformat(timespec="seconds")}</p>
"""


def _footer() -> str:
    return "</body></html>\n"


def write_report(title: str, filename: str, sections: Iterable[str]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / filename
    body = _header(title) + "\n".join(sections) + _footer()
    path.write_text(body, encoding="utf-8")
    return path


def badge(passed: bool) -> str:
    return f'<span class="{"pass" if passed else "fail"}">{ "PASS" if passed else "FAIL" }</span>'


def metrics_table(rows) -> str:
    """rows: list of (label, computed, reference, tolerance, pass_bool)."""
    h = '<table><tr><th>Metric</th><th>Computed</th><th>Reference</th><th>Tolerance</th><th>Result</th></tr>'
    for lab, comp, ref, tol, ok in rows:
        h += (f'<tr><td>{html.escape(lab)}</td>'
              f'<td>{comp}</td><td>{ref}</td><td>±{tol}</td><td>{badge(ok)}</td></tr>')
    return h + "</table>"


def plain_table(headers, rows) -> str:
    h = '<table><tr>' + ''.join(f'<th>{html.escape(str(c))}</th>' for c in headers) + '</tr>'
    for r in rows:
        h += '<tr>' + ''.join(f'<td>{html.escape(str(c))}</td>' for c in r) + '</tr>'
    return h + '</table>'


def img(data_uri: str, alt: str = "") -> str:
    return f'<img src="{data_uri}" alt="{html.escape(alt)}">'


def section(title: str, *bodies: str) -> str:
    return f'<h2>{html.escape(title)}</h2>\n' + "\n".join(bodies)


def p(text: str) -> str:
    return f"<p>{html.escape(text)}</p>"
