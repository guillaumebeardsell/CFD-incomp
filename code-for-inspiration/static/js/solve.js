// POST the spec to /api/solve, show progress, decode the response.
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
    try {
      data = JSON.parse(text);
    } catch (_) {
      const snippet = text.slice(0, 120).replace(/\s+/g, " ");
      throw new Error(`HTTP ${resp.status}: ${snippet}`);
    }
    if (!resp.ok && resp.status !== 202) {
      throw new Error(data.message || `HTTP ${resp.status}`);
    }
    return { status: resp.status, data };
  }

  C.runSolve = async function () {
    if (!C.shape) {
      alert("Draw an obstacle first.");
      C.setStep("draw");
      return;
    }
    const btn = document.getElementById("btn-solve");
    if (btn.disabled) return;
    const prevLabel = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Solving…";
    const spec = {
      tunnel: { width: C.tunnel.width, height: C.tunnel.height },
      fidelity: document.getElementById("fidelity").value,
      scheme: document.getElementById("scheme").value,
      mode: document.getElementById("mode").value,
      viscous: document.getElementById("viscous").checked,
      obstacle: { polygon: C.shape.map((p) => [p[0], p[1]]) },
      bcs: C.bcs,
      max_iters: 12000,
      res_drop: 1e-4,
    };
    showProgress("Starting solve...");
    setProgressBar(0);

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

      // Poll until the job finishes or errors.
      while (true) {
        await sleep(1000);
        const s = await fetchJson(`/api/solve/status?id=${encodeURIComponent(jobId)}`);
        const st = s.data;
        if (st.status === "running") {
          const frac = st.max_iters > 0 ? Math.min(0.98, st.iter / st.max_iters) : 0;
          setProgressBar(frac);
          const resStr = st.residual != null ? `res=${st.residual.toExponential(2)}` : "";
          setProgressDetail(`iter ${st.iter}/${st.max_iters}  ${resStr}  ${st.elapsed_s.toFixed(1)}s`);
          continue;
        }
        if (st.status === "error") throw new Error(st.error || "solver failed");
        if (st.status === "done") break;
        throw new Error(`unexpected status: ${st.status}`);
      }

      const r = await fetchJson(`/api/solve/result?id=${encodeURIComponent(jobId)}`);
      const data = r.data;
      if (data.status !== "ok") throw new Error(data.message || "bad result");
      setProgressBar(1.0);
      setProgressDetail(`${data.iters} iters, ${data.elapsed_s.toFixed(1)}s`);
      setTimeout(hideProgress, 250);

      const { nx, ny } = data.grid;
      C.result = {
        grid: data.grid,
        mask: b64ToUint8(data.mask),
        rho:  b64ToFloat32(data.rho),
        u:    b64ToFloat32(data.u),
        v:    b64ToFloat32(data.v),
        p:    b64ToFloat32(data.p),
        T:    b64ToFloat32(data.T),
        mach: b64ToFloat32(data.mach),
        res:  b64ToFloat32(data.res),
        residuals: data.residuals,
        iters: data.iters,
        converged: data.converged,
        elapsed: data.elapsed_s,
        mode: data.mode || "steady",
        nx, ny,
      };
      if (data.mode === "transient" && data.frames) {
        C.result.frames = {
          n: data.frames.n_frames,
          mach: b64ToFloat32(data.frames.mach),
          u:    b64ToFloat32(data.frames.u),
          v:    b64ToFloat32(data.frames.v),
          times: data.frames.times || [],
        };
        // Lock field select to mach — other fields aren't streamed per-frame.
        const fsel = document.getElementById("field-select");
        if (fsel) {
          fsel.value = "mach";
          C.viewOpts.field = "mach";
          fsel.disabled = true;
        }
      } else {
        const fsel = document.getElementById("field-select");
        if (fsel) fsel.disabled = false;
      }
      C.setStep("view");
    } catch (err) {
      hideProgress();
      alert("Solve failed: " + err.message);
      console.error(err);
    } finally {
      btn.disabled = false;
      btn.textContent = prevLabel;
    }
  };

  function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

  function showProgress(msg) {
    document.getElementById("progress-msg").textContent = msg;
    document.getElementById("progress").classList.remove("hidden");
    setProgressBar(0);
  }
  function hideProgress() {
    document.getElementById("progress").classList.add("hidden");
  }
  function setProgressBar(v) {
    document.getElementById("progress-bar").style.width = `${(v * 100).toFixed(1)}%`;
  }
  function setProgressDetail(s) {
    document.getElementById("progress-detail").textContent = s;
  }
})();
