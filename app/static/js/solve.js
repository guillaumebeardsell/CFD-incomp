// POST spec to /api/solve, poll status, decode response for incompressible fields.
(function () {
  const C = (window.CFD = window.CFD || {});

  function b64ToFloat32(b64) {
    if (!b64) return null;
    const bin = atob(b64);
    const buf = new ArrayBuffer(bin.length);
    const view = new Uint8Array(buf);
    for (let i = 0; i < bin.length; i++) view[i] = bin.charCodeAt(i);
    return new Float32Array(buf);
  }
  function b64ToUint8(b64) {
    if (!b64) return null;
    const bin = atob(b64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    return arr;
  }

  async function fetchJson(url, opts) {
    const resp = await fetch(url, opts);
    const text = await resp.text();
    let data;
    try { data = JSON.parse(text); }
    catch (_) { throw new Error(`HTTP ${resp.status}: ${text.slice(0, 120)}`); }
    if (!resp.ok && resp.status !== 202) {
      throw new Error(data.message || `HTTP ${resp.status}`);
    }
    return { status: resp.status, data };
  }

  function computeVelMag(u, v) {
    const n = u.length;
    const out = new Float32Array(n);
    for (let k = 0; k < n; k++) out[k] = Math.hypot(u[k], v[k]);
    return out;
  }

  C.runSolve = async function () {
    if (!C.shape) {
      alert("Draw an obstacle first.");
      C.setStep("draw");
      return;
    }
    const btn = document.getElementById("btn-solve");
    if (btn.disabled) return;
    const prev = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Solving…";
    const spec = {
      tunnel: { width: C.tunnel.width, height: C.tunnel.height },
      fidelity: document.getElementById("fidelity").value,
      mode: document.getElementById("mode").value,
      Re: parseFloat(document.getElementById("re-input").value) || 40.0,
      U_ref: 1.0,
      obstacle: { polygon: C.shape.map((p) => [p[0], p[1]]) },
      bcs: C.bcs,
      max_iters: 20000,
      res_drop: 1e-4,
    };
    const solveMode = spec.mode;
    showProgress(solveMode);

    try {
      const start = await fetchJson("/api/solve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(spec),
      });
      if (start.data.status !== "running" || !start.data.job_id) {
        throw new Error(start.data.message || "failed to start job");
      }
      const jobId = start.data.job_id;
      C._activeJobId = jobId;
      enableStopButton(jobId);

      while (true) {
        await sleep(1000);
        const s = await fetchJson(`/api/solve/status?id=${encodeURIComponent(jobId)}`);
        const st = s.data;
        if (st.status === "running") {
          renderProgress(st, solveMode);
          continue;
        }
        if (st.status === "error") throw new Error(st.error || "solver failed");
        if (st.status === "done") break;
        throw new Error(`unexpected status: ${st.status}`);
      }

      const r = await fetchJson(`/api/solve/result?id=${encodeURIComponent(jobId)}`);
      const data = r.data;
      if (data.status !== "ok") throw new Error(data.message || "bad result");
      renderComplete(data);
      setTimeout(hideProgress, 450);

      const u = b64ToFloat32(data.u);
      const v = b64ToFloat32(data.v);
      C.result = {
        grid: data.grid,
        mask: b64ToUint8(data.mask),
        u, v,
        p: b64ToFloat32(data.p),
        vorticity: b64ToFloat32(data.vorticity),
        vel_mag: computeVelMag(u, v),
        res_mom: b64ToFloat32(data.res_mom) || new Float32Array(u.length),
        res_div: b64ToFloat32(data.res_div) || new Float32Array(u.length),
        residuals: data.residuals,
        iters: data.iters,
        converged: data.converged,
        elapsed: data.elapsed_s,
        mode: data.mode || "steady",
        nx: data.grid.nx, ny: data.grid.ny,
      };
      if (data.mode === "transient" && data.frames && data.frames.n_frames > 0) {
        C.result.frames = {
          n: data.frames.n_frames,
          u: b64ToFloat32(data.frames.u),
          v: b64ToFloat32(data.frames.v),
          // p frames intentionally dropped: Pressure is hidden in transient mode
          times: data.frames.times || [],
        };
      }
      const fsel = document.getElementById("field-select");
      if (fsel) {
        fsel.disabled = false;
        const pOpt = fsel.querySelector('option[value="p"]');
        if (pOpt) {
          const hidePressure = data.mode === "transient";
          pOpt.hidden = hidePressure;
          pOpt.disabled = hidePressure;
          if (hidePressure && fsel.value === "p") {
            fsel.value = "vel_mag";
            C.viewOpts.field = "vel_mag";
          }
        }
      }
      C.setStep("view");
    } catch (err) {
      hideProgress();
      alert("Solve failed: " + err.message);
      console.error(err);
    } finally {
      btn.disabled = false;
      btn.textContent = prev;
      disableStopButton();
      C._activeJobId = null;
    }
  };

  function enableStopButton(jobId) {
    const b = document.getElementById("btn-stop-solve");
    if (!b) return;
    b.disabled = false;
    b.textContent = "Stop";
    b.onclick = async () => {
      if (b.disabled) return;
      b.disabled = true;
      b.textContent = "Stopping…";
      try {
        await fetch(`/api/solve/cancel?id=${encodeURIComponent(jobId)}`, { method: "POST" });
      } catch (err) {
        console.error("cancel failed:", err);
      }
    };
  }
  function disableStopButton() {
    const b = document.getElementById("btn-stop-solve");
    if (!b) return;
    b.disabled = true;
    b.onclick = null;
  }

  function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

  let _progressFloor = 0;
  function showProgress(mode) {
    const card = document.querySelector(".progress-card");
    if (card) card.classList.remove("is-done");
    _progressFloor = 0;
    const title = mode === "transient" ? "Preparing transient solve" : "Preparing steady solve";
    setTitle(title, "Initializing…");
    setProgressBar(0, 0);
    setStat("a", "Residual", "—");
    setStat("b", "Iteration", "—");
    setElapsed("—");
    document.getElementById("progress").classList.remove("hidden");
  }
  function hideProgress() { document.getElementById("progress").classList.add("hidden"); }

  function setTitle(title, subtitle) {
    document.getElementById("progress-title").textContent = title;
    document.getElementById("progress-subtitle").textContent = subtitle || "";
  }
  function setProgressBar(frac, pctOverride) {
    const clamped = Math.max(0, Math.min(1, frac));
    document.getElementById("progress-bar").style.width = `${(clamped * 100).toFixed(1)}%`;
    const pct = pctOverride != null ? pctOverride : Math.round(clamped * 100);
    document.getElementById("progress-percent").textContent = `${pct}%`;
  }
  function setStat(slot, label, value) {
    document.getElementById(`stat-${slot}-label`).textContent = label;
    document.getElementById(`stat-${slot}-value`).textContent = value;
  }
  function setElapsed(v) { document.getElementById("stat-c-value").textContent = v; }

  function fmtElapsed(s) {
    if (s == null || !isFinite(s)) return "—";
    if (s < 60) return `${s.toFixed(1)}s`;
    const m = Math.floor(s / 60);
    const sec = Math.round(s - 60 * m);
    if (m < 60) return `${m}m ${String(sec).padStart(2, "0")}s`;
    const h = Math.floor(m / 60);
    return `${h}h ${String(m - 60 * h).padStart(2, "0")}m`;
  }
  function fmtInt(n) {
    if (n == null) return "—";
    return Math.round(n).toLocaleString();
  }

  function renderProgress(st, solveMode) {
    const cancelling = !!st.cancelling;
    const phase = st.phase || (solveMode === "transient" ? "warm-start" : "steady");
    const r = st.residual, r0 = st.r0;

    let titleMain, subtitle;
    if (phase === "transient") {
      titleMain = "Time integration";
      const tEnd = st.t_end, tBuf = st.t_buffer;
      if (tEnd != null && tBuf != null && st.t != null && st.t < tBuf) {
        subtitle = `settling flow before recording (t_buffer = ${tBuf.toFixed(1)}s)`;
      } else if (tEnd != null) {
        subtitle = `integrating to t = ${tEnd.toFixed(1)}s`;
      } else {
        subtitle = "marching in physical time";
      }
    } else if (phase === "warm-start") {
      titleMain = "Warm-up · steady";
      subtitle = "converging initial field before time integration";
    } else {
      titleMain = "Steady solve";
      const target = st.phase_res_drop;
      subtitle = target != null
        ? `driving residual ${target.toExponential(0)}× below initial`
        : "driving momentum residual down";
    }
    if (cancelling) subtitle = "stopping…";
    setTitle(titleMain, subtitle);

    let frac = 0;
    if (phase === "transient") {
      if (st.t != null && st.t_end && st.t_end > 0) frac = st.t / st.t_end;
    } else {
      const target = st.phase_res_drop || 1e-4;
      const maxIters = st.phase_max_iters || st.max_iters || 1;
      if (r != null && r0 != null && r > 0 && r0 > 0) {
        const logTarget = Math.log(1 / target);
        const logCurrent = Math.max(0, Math.log(r0 / r));
        frac = Math.max(frac, Math.min(1, logCurrent / logTarget));
      }
      frac = Math.max(frac, Math.min(1, (st.iter || 0) / maxIters));
    }
    frac = Math.min(frac, 0.99);
    if (frac < _progressFloor) frac = _progressFloor;
    else _progressFloor = frac;
    setProgressBar(frac);

    setStat("a", "Residual", r != null ? r.toExponential(2) : "—");
    if (phase === "transient") {
      const tStr = st.t != null ? `${st.t.toFixed(2)}` : "—";
      const tEndStr = st.t_end != null ? ` / ${st.t_end.toFixed(1)}s` : "";
      setStat("b", "Sim time", `${tStr}${tEndStr}`);
    } else {
      setStat("b", "Iteration", fmtInt(st.iter));
    }
    setElapsed(fmtElapsed(st.elapsed_s));
  }

  function renderComplete(data) {
    const card = document.querySelector(".progress-card");
    if (card) card.classList.add("is-done");
    const stopped = !!data.cancelled;
    const converged = !!data.converged;
    let title;
    if (stopped) title = "Stopped";
    else if (converged) title = data.mode === "transient" ? "Integration complete" : "Converged";
    else title = "Finished (not converged)";
    setTitle(title, data.log_file ? `log: ${data.log_file}` : "");
    setProgressBar(1.0, 100);
    setStat("a", "Residual", "done");
    setStat("b", "Iterations", fmtInt(data.iters));
    setElapsed(fmtElapsed(data.elapsed_s));
  }
})();
