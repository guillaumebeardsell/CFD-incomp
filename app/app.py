"""Flask backend for the incompressible CFD app.

Usage (from repo root): `python -m app.app`  or  `python app/app.py`.

Endpoints:
  POST /api/solve            -> {"job_id": "...", "status": "running"}
  GET  /api/solve/status?id= -> progress snapshot
  GET  /api/solve/result?id= -> full payload once status == "done"
"""
from __future__ import annotations

import base64
import csv
import datetime as _dt
import hmac
import logging
import os
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np

# Make `import app.solver ...` work regardless of cwd when this file is run
# directly (python app/app.py) rather than as a package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from flask import Flask, Response, jsonify, request, send_from_directory   # noqa: E402

try:                                                             # pragma: no cover
    from app.solver import Solver, SolverConfig, FIDELITY
    from app.solver import mesh as mesh_mod
    from app.solver.diagnostics import vorticity
    from app.solver.report import write_residuals_xlsx
except ModuleNotFoundError:   # when running `python app/app.py`
    from solver import Solver, SolverConfig, FIDELITY
    from solver import mesh as mesh_mod
    from solver.diagnostics import vorticity
    from solver.report import write_residuals_xlsx


_STATIC_DIR = str(Path(__file__).resolve().parent / "static")
if getattr(sys, "frozen", False):
    _LOG_DIR = Path.home() / ".cfd-incomp" / "logs"
else:
    _LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
app = Flask(__name__, static_folder=_STATIC_DIR, static_url_path="")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cfd")

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

_AUTH_PASSWORD = os.environ.get("APP_PASSWORD")


@app.before_request
def _require_password():
    if not _AUTH_PASSWORD:
        return None
    auth = request.authorization
    if auth and hmac.compare_digest(auth.password or "", _AUTH_PASSWORD):
        return None
    return Response(
        "Authentication required.", 401,
        {"WWW-Authenticate": 'Basic realm="cfd"'},
    )


def _b64(arr, dtype=np.float32) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=dtype).tobytes()).decode("ascii")


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


def _parse_spec(spec: dict) -> dict:
    tunnel = spec["tunnel"]
    width = float(tunnel.get("width", 20.0))
    height = float(tunnel.get("height", 10.0))
    fidelity = spec.get("fidelity", "medium")
    if fidelity in FIDELITY:
        nx_d, ny_d = FIDELITY[fidelity]
    else:
        nx_d = int(tunnel.get("nx", 320))
        ny_d = int(tunnel.get("ny", 160))

    # Force square cells so geometric MG stays well-conditioned.
    # Pick whichever direction sets the smaller dx, then match the other.
    dx = min(width / nx_d, height / ny_d)
    nx = max(16, int(round(width / dx)))
    ny = max(16, int(round(height / dx)))
    # Nudge to even for clean MG coarsening.
    if nx % 2:
        nx += 1
    if ny % 2:
        ny += 1

    mode = spec.get("mode", "steady")
    if mode not in ("steady", "transient"):
        raise ValueError(f"unknown mode: {mode!r}")

    bcs = spec.get("bcs") or {}
    polygon = (spec.get("obstacle") or {}).get("polygon")

    return {
        "width": width, "height": height, "nx": nx, "ny": ny,
        "polygon": polygon,
        "bcs": bcs,
        "mode": mode,
        "Re": float(spec.get("Re", 40.0)),
        "U_ref": float(spec.get("U_ref", 1.0)),
        "eta": float(spec.get("eta", 1e-3)),
        "max_iters": int(spec.get("max_iters", 20000)),
        "res_drop": float(spec.get("res_drop", 1e-4)),
    }


def _run_job(job_id: str, p: dict) -> None:
    job = _jobs[job_id]
    stamp = _dt.datetime.fromtimestamp(job["t0"]).strftime("%Y%m%d_%H%M%S")
    slug = f"sim_{stamp}_{job_id[:8]}"
    log_path = _LOG_DIR / f"{slug}.log"
    csv_path = _LOG_DIR / f"{slug}_residuals.csv"
    job["log_path"] = str(log_path)
    job["residuals_csv"] = str(csv_path)

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)

    xlsx_path = _LOG_DIR / f"{slug}_residuals.xlsx"
    job["residuals_xlsx"] = str(xlsx_path)

    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["iter", "R_mom", "R_div"])

    def on_residual(it, r_mom, r_div):
        csv_writer.writerow([it, f"{r_mom:.6e}", f"{r_div:.6e}"])

    result = None

    try:
        log.info("simulation %s started; log=%s  residuals=%s",
                 slug, log_path.name, csv_path.name)
        mesh = mesh_mod.build_mesh(p["width"], p["height"], p["nx"], p["ny"])
        poly = p["polygon"]
        chi = (
            mesh_mod.rasterize_polygon(poly, mesh)
            if poly else np.zeros((p["ny"], p["nx"]), dtype=bool)
        )
        cfg = SolverConfig(
            mode=p["mode"], Re=p["Re"], U_ref=p["U_ref"], eta=p["eta"],
            max_iters=p["max_iters"], res_drop=p["res_drop"],
        )
        n_poly = 0 if not poly else len(poly)
        log.info("solve start: grid=%dx%d (%.3fx%.3f) mode=%s Re=%.1f  "
                 "polygon=%d pts, solid_cells=%d / %d",
                 p["nx"], p["ny"], p["width"], p["height"], p["mode"], p["Re"],
                 n_poly, int(chi.sum()), chi.size)
        if poly and n_poly >= 3:
            xs = [pt[0] for pt in poly]
            ys = [pt[1] for pt in poly]
            log.info("  polygon bbox: x=[%.3f,%.3f] y=[%.3f,%.3f]",
                     min(xs), max(xs), min(ys), max(ys))

        def on_progress(snap: dict):
            job["iter"] = snap.get("iter", 0)
            job["residual"] = snap.get("residual")
            job["r0"] = snap.get("r0")
            job["phase"] = snap.get("phase")
            job["phase_max_iters"] = snap.get("phase_max_iters")
            job["phase_res_drop"] = snap.get("phase_res_drop")
            job["t"] = snap.get("t")
            job["t_end"] = snap.get("t_end")
            job["t_buffer"] = snap.get("t_buffer")

        solver = Solver(
            mesh, chi, p["bcs"], cfg,
            log=lambda s: log.info(s),
            progress=on_progress,
            residual_log=on_residual,
            should_stop=job["cancel_event"].is_set,
        )
        result = solver.solve()
        elapsed = result.elapsed_s
        log.info("solve done: iters=%d converged=%s elapsed=%.1fs",
                 result.iters, result.converged, elapsed)

        residuals = result.residuals
        if len(residuals) > 400:
            step = max(1, len(residuals) // 400)
            residuals = residuals[::step]

        payload = {
            "status": "ok",
            "mode": result.mode,
            "grid": {
                "nx": mesh["nx"], "ny": mesh["ny"],
                "dx": mesh["dx"], "dy": mesh["dy"],
                "x0": mesh["x0"], "y0": mesh["y0"],
                "width": mesh["width"], "height": mesh["height"],
            },
            "mask":      base64.b64encode(
                np.ascontiguousarray(result.mask, dtype=np.uint8).tobytes()
            ).decode("ascii"),
            "u":         _b64(result.u),
            "v":         _b64(result.v),
            "p":         _b64(result.p),
            "vorticity": _b64(result.vorticity),
            "res_mom":   _b64(result.res_mom_field),
            "res_div":   _b64(result.res_div_field),
            "residuals": residuals,
            "cont_residuals": (result.cont_residuals[::max(1, len(result.cont_residuals) // 400)]
                               if len(result.cont_residuals) > 400 else result.cont_residuals),
            "log_file": log_path.name,
            "residuals_csv": csv_path.name,
            "residuals_xlsx": xlsx_path.name,
            "iters": result.iters,
            "converged": result.converged,
            "cancelled": result.cancelled,
            "elapsed_s": elapsed,
        }
        if result.mode == "transient" and result.frames_u:
            # Cap frame count and drop p-frames so the JSON response stays small
            # enough for the browser to fetch reliably (see logs/ for prior OOMs).
            FRAME_CAP = 120
            n_full = len(result.frames_u)
            stride = max(1, (n_full + FRAME_CAP - 1) // FRAME_CAP)
            idx = list(range(0, n_full, stride))
            fu = np.stack([result.frames_u[i] for i in idx], axis=0)
            fv = np.stack([result.frames_v[i] for i in idx], axis=0)
            times = [result.frame_times[i] for i in idx]
            payload["frames"] = {
                "n_frames": int(len(idx)),
                "u": _b64(fu),
                "v": _b64(fv),
                "times": times,
            }
            log.info("payload: shipping %d/%d frames (stride=%d), u+v only",
                     len(idx), n_full, stride)
        job["result"] = payload
        job["status"] = "done"
    except Exception as exc:                              # noqa: BLE001
        log.exception("solver failed")
        job["status"] = "error"
        job["error"] = str(exc)
    finally:
        try:
            csv_file.flush(); csv_file.close()
        except Exception:                                 # noqa: BLE001
            pass
        try:
            meta = {
                "slug": slug,
                "started": _dt.datetime.fromtimestamp(job["t0"]).isoformat(timespec="seconds"),
                "mode": p["mode"],
                "Re": p["Re"],
                "nx": p["nx"], "ny": p["ny"],
                "width": p["width"], "height": p["height"],
                "res_drop": p["res_drop"],
                "log_file": log_path.name,
                "iters": getattr(result, "iters", job.get("iter", 0)),
                "converged": getattr(result, "converged", False),
                "elapsed_s": round(getattr(result, "elapsed_s", time.time() - job["t0"]), 3),
            }
            write_residuals_xlsx(csv_path, xlsx_path, meta)
            log.info("residuals workbook written: %s", xlsx_path.name)
        except Exception:                                 # noqa: BLE001
            log.exception("failed to write residuals xlsx")
        log.removeHandler(fh)
        fh.close()


@app.post("/api/solve")
def api_solve():
    try:
        spec = request.get_json(force=True)
        p = _parse_spec(spec)
    except Exception as exc:                              # noqa: BLE001
        return jsonify({"status": "error", "message": f"bad spec: {exc}"}), 400

    job_id = uuid.uuid4().hex
    with _jobs_lock:
        while len(_jobs) >= 3:
            _jobs.pop(next(iter(_jobs)))
        _jobs[job_id] = {
            "status": "running",
            "iter": 0,
            "residual": None,
            "max_iters": p["max_iters"],
            "t0": time.time(),
            "cancel_event": threading.Event(),
        }
    threading.Thread(target=_run_job, args=(job_id, p), daemon=True).start()
    return jsonify({"status": "running", "job_id": job_id})


def _get_job_or_404(job_id):
    j = _jobs.get(job_id or "")
    if not j:
        return None, (jsonify({"status": "not_found"}), 404)
    return j, None


@app.get("/api/solve/status")
def api_status():
    j, err = _get_job_or_404(request.args.get("id"))
    if err:
        return err
    return jsonify({
        "status": j["status"],
        "iter": j.get("iter", 0),
        "max_iters": j.get("max_iters"),
        "residual": j.get("residual"),
        "r0": j.get("r0"),
        "phase": j.get("phase"),
        "phase_max_iters": j.get("phase_max_iters"),
        "phase_res_drop": j.get("phase_res_drop"),
        "t": j.get("t"),
        "t_end": j.get("t_end"),
        "t_buffer": j.get("t_buffer"),
        "elapsed_s": time.time() - j["t0"],
        "error": j.get("error"),
        "cancelling": bool(j.get("cancel_event") and j["cancel_event"].is_set()),
    })


@app.post("/api/solve/cancel")
def api_cancel():
    j, err = _get_job_or_404(request.args.get("id"))
    if err:
        return err
    ev = j.get("cancel_event")
    if ev is not None:
        ev.set()
    return jsonify({"status": "cancelling"})


@app.get("/api/solve/result")
def api_result():
    j, err = _get_job_or_404(request.args.get("id"))
    if err:
        return err
    if j["status"] == "done":
        return jsonify(j["result"])
    if j["status"] == "error":
        return jsonify({"status": "error", "message": j.get("error", "unknown")}), 500
    return jsonify({"status": j["status"]}), 202


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5150))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
