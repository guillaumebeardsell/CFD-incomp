"""Flask backend for the interactive CFD app.

Serves the static PWA. The solver can take minutes on medium/fine grids, which
is longer than the ~100s quick-tunnel idle timeout, so solves run in a
background thread and the client polls:

  POST /api/solve            -> {"job_id": "...", "status": "running"}
  GET  /api/solve/status?id= -> progress snapshot
  GET  /api/solve/result?id= -> full payload once status == "done"
"""
from __future__ import annotations

import base64
import logging
import threading
import time
import uuid

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from solver import config, mesh as mesh_mod
from solver.solver import Solver

app = Flask(__name__, static_folder="static", static_url_path="")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cfd")

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _b64(arr, dtype=np.float32):
    return base64.b64encode(np.ascontiguousarray(arr, dtype=dtype).tobytes()).decode("ascii")


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


def _parse_spec(spec):
    tunnel = spec["tunnel"]
    bcs = spec["bcs"]
    fidelity = spec.get("fidelity", "medium")
    scheme = spec.get("scheme", "explicit")
    if scheme not in ("explicit", "irs"):
        raise ValueError(f"unknown scheme: {scheme!r}")
    mode = spec.get("mode", "steady")
    if mode not in ("steady", "transient"):
        raise ValueError(f"unknown mode: {mode!r}")
    if mode == "transient":
        scheme = "explicit"  # IRS is time-inaccurate
    if fidelity in config.FIDELITY:
        nx, ny = config.FIDELITY[fidelity]
    else:
        nx = int(tunnel.get("nx", 240))
        ny = int(tunnel.get("ny", 120))
    width = float(tunnel.get("width", 2.0))
    height = float(tunnel.get("height", 1.0))
    return {
        "bcs": bcs,
        "nx": nx, "ny": ny,
        "width": width, "height": height,
        "polygon": (spec.get("obstacle") or {}).get("polygon"),
        "gas": spec.get("gas") or {"gamma": config.GAMMA, "R": config.R_GAS},
        "viscous": bool(spec.get("viscous", False)),
        "max_iters": int(spec.get("max_iters", 12000)),
        "res_drop": float(spec.get("res_drop", 1e-4)),
        "scheme": scheme,
        "mode": mode,
    }


def _run_job(job_id: str, p: dict) -> None:
    job = _jobs[job_id]
    try:
        m = mesh_mod.build_mesh(p["width"], p["height"], p["nx"], p["ny"])
        solid = (
            mesh_mod.rasterize_polygon(p["polygon"], m)
            if p["polygon"] else np.zeros((p["ny"], p["nx"]), dtype=bool)
        )
        log.info("solve start: grid=%dx%d max_iters=%d", p["nx"], p["ny"], p["max_iters"])

        def on_progress(it, res):
            job["iter"] = it
            job["residual"] = res

        solver = Solver(
            m, solid, p["bcs"],
            gas=p["gas"],
            viscous=p["viscous"],
            max_iters=p["max_iters"],
            res_drop=p["res_drop"],
            log=lambda s: log.info(s),
            progress=on_progress,
            polygon_xy=p["polygon"],
            scheme=p["scheme"],
            mode=p["mode"],
        )
        result = solver.solve()
        elapsed = time.time() - job["t0"]
        log.info("solve done: iters=%d converged=%s elapsed=%.1fs",
                 result.iters, result.converged, elapsed)

        residuals = result.residuals
        if len(residuals) > 400:
            step = len(residuals) // 400
            residuals = residuals[::step]

        payload = {
            "status": "ok",
            "mode": p["mode"],
            "grid": {
                "nx": p["nx"], "ny": p["ny"],
                "dx": m["dx"], "dy": m["dy"],
                "x0": m["x0"], "y0": m["y0"],
                "width": p["width"], "height": p["height"],
            },
            "mask": base64.b64encode(
                np.ascontiguousarray(result.mask, dtype=np.uint8).tobytes()
            ).decode("ascii"),
            "rho":  _b64(result.W[0]),
            "u":    _b64(result.W[1]),
            "v":    _b64(result.W[2]),
            "p":    _b64(result.W[3]),
            "T":    _b64(result.T),
            "mach": _b64(result.mach),
            "res":  _b64(result.res_field),
            "residuals": residuals,
            "iters": result.iters,
            "converged": result.converged,
            "elapsed_s": elapsed,
        }
        if p["mode"] == "transient" and result.frames_mach is not None:
            payload["frames"] = {
                "n_frames": int(result.frames_mach.shape[0]),
                "mach": _b64(result.frames_mach),
                "u":    _b64(result.frames_u),
                "v":    _b64(result.frames_v),
                "times": result.frame_times.tolist(),
            }
        job["result"] = payload
        job["status"] = "done"
    except Exception as exc:
        log.exception("solver failed")
        job["status"] = "error"
        job["error"] = str(exc)


@app.post("/api/solve")
def api_solve():
    try:
        spec = request.get_json(force=True)
        p = _parse_spec(spec)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"bad spec: {exc}"}), 400

    job_id = uuid.uuid4().hex
    with _jobs_lock:
        # Bounded FIFO so previously-issued job_ids still resolve while the
        # client finishes polling, but memory stays capped.
        while len(_jobs) >= 3:
            _jobs.pop(next(iter(_jobs)))
        _jobs[job_id] = {
            "status": "running",
            "iter": 0,
            "residual": None,
            "max_iters": p["max_iters"],
            "t0": time.time(),
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
        "elapsed_s": time.time() - j["t0"],
        "error": j.get("error"),
    })


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
    import os
    port = int(os.environ.get("PORT", 5050))
    # threaded=True so the polling requests can be served while a solve is running.
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
