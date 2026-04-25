"""Run all benchmarks and emit reports/index.html linking them."""
from __future__ import annotations

import sys
from pathlib import Path

from . import _report
from .bench_cylinder_re40 import build_report as build_cyl
from .bench_poiseuille import build_report as build_poi


def main(fidelity: str = "medium") -> int:
    print("=== Running Poiseuille benchmark ===")
    r_poi = build_poi()
    print(f"-> {r_poi['path']}  pass={r_poi['passed']}  order={r_poi['order']:.2f}")

    print("=== Running Cylinder Re=40 benchmark ===")
    if fidelity == "coarse":
        r_cyl = build_cyl(nx=160, ny=160, max_iters=8000)
    else:
        r_cyl = build_cyl(nx=200, ny=200, max_iters=15000)
    print(f"-> {r_cyl['path']}  pass={r_cyl['passed']}")

    idx = _report.REPORT_DIR / "index.html"
    links = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CFD Benchmarks</title>
<style>body {{ font-family: system-ui; max-width: 720px; margin: 2rem auto; }}
.pass {{ color: green; }} .fail {{ color: red; }}</style>
</head><body>
<h1>CFD Verification Benchmarks</h1>
<ul>
  <li><a href="poiseuille.html">Penalized plane Poiseuille</a>
      — <span class="{'pass' if r_poi['passed'] else 'fail'}">{ 'PASS' if r_poi['passed'] else 'FAIL' }</span>
      (order {r_poi['order']:.2f})</li>
  <li><a href="cylinder_re40.html">Cylinder at Re=40 vs Gautier</a>
      — <span class="{'pass' if r_cyl['passed'] else 'fail'}">{ 'PASS' if r_cyl['passed'] else 'FAIL' }</span></li>
</ul>
</body></html>"""
    idx.write_text(links, encoding="utf-8")
    print(f"=== Index written to {idx} ===")
    return 0 if (r_poi["passed"] and r_cyl["passed"]) else 1


if __name__ == "__main__":
    fid = sys.argv[1] if len(sys.argv) > 1 else "medium"
    sys.exit(main(fid))
