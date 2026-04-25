// Visualization: colormap contour + quiver arrows + streamlines (incompressible).
(function () {
  const C = (window.CFD = window.CFD || {});

  const STOPS = [
    [0.0,  [48,  18,  59 ]],
    [0.15, [37, 114, 245]],
    [0.30, [32, 207, 190]],
    [0.50, [176, 223,  60]],
    [0.65, [248, 191,  38]],
    [0.80, [234,  82,  37]],
    [1.0,  [122,  4,   3 ]],
  ];
  function turbo(t) {
    t = Math.max(0, Math.min(1, t));
    for (let i = 1; i < STOPS.length; i++) {
      if (t <= STOPS[i][0]) {
        const [t0, c0] = STOPS[i - 1], [t1, c1] = STOPS[i];
        const f = (t - t0) / (t1 - t0);
        return [c0[0] + f * (c1[0] - c0[0]),
                c0[1] + f * (c1[1] - c0[1]),
                c0[2] + f * (c1[2] - c0[2])];
      }
    }
    return STOPS[STOPS.length - 1][1];
  }

  function computeVorticity(u, v, nx, ny, dx, dy) {
    const out = new Float32Array(nx * ny);
    for (let j = 0; j < ny; j++) {
      const jp = j < ny - 1 ? j + 1 : j;
      const jm = j > 0 ? j - 1 : j;
      const hy = (jp - jm) * dy || dy;
      for (let i = 0; i < nx; i++) {
        const ip = i < nx - 1 ? i + 1 : i;
        const im = i > 0 ? i - 1 : i;
        const hx = (ip - im) * dx || dx;
        const dvdx = (v[j * nx + ip] - v[j * nx + im]) / hx;
        const dudy = (u[jp * nx + i] - u[jm * nx + i]) / hy;
        out[j * nx + i] = dvdx - dudy;
      }
    }
    return out;
  }

  function fieldArray(r, key) {
    if (key === "vel_mag" && !r.vel_mag) {
      const n = r.u.length;
      r.vel_mag = new Float32Array(n);
      for (let k = 0; k < n; k++) r.vel_mag[k] = Math.hypot(r.u[k], r.v[k]);
    }
    if (key === "vorticity" && !r.vorticity && r.frames) {
      r.vorticity = computeVorticity(r.u, r.v, r.nx, r.ny, r.grid.dx, r.grid.dy);
    }
    return r[key];
  }

  function frameFieldStableRange(r, key) {
    if (!r.frames) return null;
    if (key !== "u" && key !== "v" && key !== "p" &&
        key !== "vel_mag" && key !== "vorticity") return null;
    // Pressure frames may be omitted from the payload; fall back to static snapshot.
    if (key === "p" && !r.frames.p) return null;
    const N = r.nx * r.ny;
    let lo = Infinity, hi = -Infinity;
    for (let f = 0; f < r.frames.n; f++) {
      const off = f * N;
      const u = r.frames.u.subarray(off, off + N);
      const v = r.frames.v.subarray(off, off + N);
      const p = r.frames.p ? r.frames.p.subarray(off, off + N) : null;
      let arr;
      if (key === "u") arr = u;
      else if (key === "v") arr = v;
      else if (key === "p") arr = p;
      else if (key === "vel_mag") {
        arr = new Float32Array(N);
        for (let k = 0; k < N; k++) arr[k] = Math.hypot(u[k], v[k]);
      } else {
        arr = computeVorticity(u, v, r.nx, r.ny, r.grid.dx, r.grid.dy);
      }
      for (let k = 0; k < N; k++) {
        if (r.mask[k]) continue;
        const val = arr[k];
        if (val < lo) lo = val;
        if (val > hi) hi = val;
      }
    }
    if (!isFinite(lo) || !isFinite(hi) || hi <= lo) return null;
    return { lo, hi };
  }
  function fieldLabel(key) {
    return { u: "u", v: "v", p: "Pressure", vel_mag: "|u|",
             vorticity: "ω",
             res_mom: "R_mom (log)", res_div: "R_div (log)" }[key] || key;
  }
  function isLogField(key) { return key === "res_mom" || key === "res_div"; }

  function logRange(f, mask) {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < f.length; i++) {
      if (mask[i]) continue;
      const v = f[i];
      if (!isFinite(v) || v <= 0) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (!isFinite(lo) || !isFinite(hi) || hi <= lo) {
      return { lo: 1e-12, hi: 1e-6, log: true };
    }
    // Snap to decades for a clean, readable colorbar.
    const lLo = Math.floor(Math.log10(lo));
    const lHi = Math.ceil(Math.log10(hi));
    return { lo: Math.pow(10, lLo), hi: Math.pow(10, lHi), log: true };
  }

  function normalize(f, mask) {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < f.length; i++) {
      if (mask[i]) continue;
      const v = f[i];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (!isFinite(lo) || !isFinite(hi) || hi === lo) return { lo: 0, hi: 1 };
    return { lo, hi };
  }

  function inpaintField(f, mask, nx, ny) {
    const out = new Float32Array(f);
    const known = new Uint8Array(mask.length);
    for (let k = 0; k < known.length; k++) known[k] = mask[k] ? 0 : 1;
    const newKnown = new Uint8Array(known.length);
    for (let pass = 0; pass < 8; pass++) {
      newKnown.set(known);
      let changed = false;
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const k = j * nx + i;
          if (known[k]) continue;
          let sum = 0, cnt = 0;
          for (let dj = -1; dj <= 1; dj++) {
            for (let di = -1; di <= 1; di++) {
              if (dj === 0 && di === 0) continue;
              const nj = j + dj, ni = i + di;
              if (nj < 0 || nj >= ny || ni < 0 || ni >= nx) continue;
              const nk = nj * nx + ni;
              if (known[nk]) { sum += out[nk]; cnt++; }
            }
          }
          if (cnt > 0) { out[k] = sum / cnt; newKnown[k] = 1; changed = true; }
        }
      }
      known.set(newKnown);
      if (!changed) break;
    }
    return out;
  }

  function buildFieldImage(r, key, rangeOverride) {
    const { nx, ny } = r;
    const raw = fieldArray(r, key);
    if (!raw) return null;
    const logMode = isLogField(key);
    const f = inpaintField(raw, r.mask, nx, ny);
    const range = rangeOverride || (logMode ? logRange(raw, r.mask) : normalize(raw, r.mask));
    const { lo, hi } = range;
    const log = !!range.log;
    const lLo = log ? Math.log10(lo) : 0;
    const lHi = log ? Math.log10(hi) : 0;
    const img = new ImageData(nx, ny);
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const idxSrc = j * nx + i;
        const idxDst = (ny - 1 - j) * nx + i;
        const k = 4 * idxDst;
        let t;
        if (log) {
          const v = f[idxSrc];
          const lv = v > 0 ? Math.log10(v) : lLo;
          t = (lv - lLo) / (lHi - lLo + 1e-30);
        } else {
          t = (f[idxSrc] - lo) / (hi - lo + 1e-30);
        }
        const [R, G, B] = turbo(t);
        img.data[k] = R; img.data[k + 1] = G; img.data[k + 2] = B; img.data[k + 3] = 255;
      }
    }
    return { img, lo, hi, log };
  }

  function maxSpeed(r) {
    let m = 0;
    for (let i = 0; i < r.u.length; i++) {
      const s = Math.hypot(r.u[i], r.v[i]);
      if (s > m) m = s;
    }
    return m;
  }

  function drawQuiver(ctx, r) {
    const { dx, dy } = r.grid;
    const step = Math.max(4, Math.round(r.nx / 30));
    const vmax = Math.max(1e-6, maxSpeed(r));
    ctx.strokeStyle = "rgba(255,255,255,0.85)";
    ctx.fillStyle = "rgba(255,255,255,0.85)";
    ctx.lineWidth = 1.5 * C.dpr;
    for (let j = step / 2 | 0; j < r.ny; j += step) {
      for (let i = step / 2 | 0; i < r.nx; i += step) {
        if (r.mask[j * r.nx + i]) continue;
        const x = (i + 0.5) * dx;
        const y = (j + 0.5) * dy;
        const u = r.u[j * r.nx + i];
        const v = r.v[j * r.nx + i];
        const s = Math.hypot(u, v);
        if (s < 0.02 * vmax) continue;
        const [px, py] = C.worldToCanvas([x, y]);
        const len = (step * dx * 0.45) * (s / vmax) * C.transform.scale;
        const dir = [u / s, v / s];
        const x2 = px + dir[0] * len;
        const y2 = py - dir[1] * len;
        ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(x2, y2); ctx.stroke();
        const ah = 4 * C.dpr;
        const ang = Math.atan2(-dir[1], dir[0]);
        ctx.beginPath();
        ctx.moveTo(x2, y2);
        ctx.lineTo(x2 - ah * Math.cos(ang - 0.4), y2 - ah * Math.sin(ang - 0.4));
        ctx.lineTo(x2 - ah * Math.cos(ang + 0.4), y2 - ah * Math.sin(ang + 0.4));
        ctx.closePath();
        ctx.fill();
      }
    }
  }

  function drawObstacleOutline(ctx, r) {
    if (!C.shape) return;
    ctx.beginPath();
    C.shape.forEach((p, i) => {
      const [x, y] = C.worldToCanvas(p);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fillStyle = "#0b1220";
    ctx.fill();
    ctx.strokeStyle = "#e6edf7";
    ctx.lineWidth = 1.5 * C.dpr;
    ctx.stroke();
  }

  function drawLegend(ctx, label, lo, hi, log) {
    const W = C.canvas.width;
    const pad = 14 * C.dpr;
    const w = 18 * C.dpr;
    const h = 160 * C.dpr;
    const x = W - pad - w;
    const y = pad + 44 * C.dpr;
    const grad = ctx.createLinearGradient(0, y + h, 0, y);
    for (let i = 0; i <= 10; i++) {
      const [R, G, B] = turbo(i / 10);
      grad.addColorStop(i / 10, `rgb(${R | 0},${G | 0},${B | 0})`);
    }
    ctx.fillStyle = grad; ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = "rgba(255,255,255,0.35)"; ctx.lineWidth = 1 * C.dpr;
    ctx.strokeRect(x, y, w, h);

    ctx.fillStyle = "#e6edf7";
    ctx.font = `${14 * C.dpr}px -apple-system, sans-serif`;
    ctx.textAlign = "right"; ctx.textBaseline = "middle";

    if (log) {
      const lLo = Math.log10(lo), lHi = Math.log10(hi);
      // Choose up to ~6 decade ticks; decimate if the span is large.
      const span = Math.max(1, Math.round(lHi - lLo));
      const stride = Math.max(1, Math.ceil(span / 6));
      for (let e = Math.ceil(lLo); e <= Math.floor(lHi); e += stride) {
        const frac = (e - lLo) / (lHi - lLo);
        const ty = y + h - frac * h;
        ctx.strokeStyle = "rgba(255,255,255,0.5)";
        ctx.beginPath();
        ctx.moveTo(x - 3 * C.dpr, ty); ctx.lineTo(x, ty); ctx.stroke();
        ctx.fillText(`10${supScript(e)}`, x - 6 * C.dpr, ty);
      }
    } else {
      ctx.fillText(fmt(hi), x - 6 * C.dpr, y);
      ctx.fillText(fmt((hi + lo) / 2), x - 6 * C.dpr, y + h / 2);
      ctx.fillText(fmt(lo), x - 6 * C.dpr, y + h);
    }
    ctx.textAlign = "right"; ctx.textBaseline = "bottom";
    ctx.fillText(label, x + w, y - 6 * C.dpr);
  }
  function supScript(n) {
    const map = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³",
                  "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
    return String(n).split("").map((c) => map[c] || c).join("");
  }
  function fmt(v) {
    if (!isFinite(v)) return "";
    if (Math.abs(v) >= 1000 || (Math.abs(v) > 0 && Math.abs(v) < 0.01)) return v.toExponential(1);
    return v.toPrecision(3);
  }

  function renderBase(target, r, field, rangeOverride) {
    target.clearRect(0, 0, C.canvas.width, C.canvas.height);
    const info = buildFieldImage(r, field, rangeOverride);
    if (!info) { drawObstacleOutline(target, r); return; }
    const { img, lo, hi, log } = info;

    const off = document.createElement("canvas");
    off.width = r.nx; off.height = r.ny;
    off.getContext("2d").putImageData(img, 0, 0);

    const [x0, y0] = C.worldToCanvas([0, C.tunnel.height]);
    const sx = C.tunnel.width * C.transform.scale;
    const sy = C.tunnel.height * C.transform.scale;
    target.imageSmoothingEnabled = true;
    target.drawImage(off, x0, y0, sx, sy);

    if (C.viewOpts.showArrows) drawQuiver(target, r);
    drawObstacleOutline(target, r);
    drawLegend(target, fieldLabel(field), lo, hi, log);

    target.fillStyle = "rgba(17, 26, 46, 0.8)";
    target.strokeStyle = "rgba(100, 130, 200, 0.5)";
    target.lineWidth = 1 * C.dpr;
    target.font = `${11 * C.dpr}px -apple-system, sans-serif`;
    const hud = `grid ${r.nx}x${r.ny}  iters ${r.iters}  ${r.converged ? "converged" : "capped"}  ${r.elapsed.toFixed(1)}s`;
    const tw = target.measureText(hud).width + 12 * C.dpr;
    target.fillRect(10 * C.dpr, C.canvas.height - 28 * C.dpr, tw, 20 * C.dpr);
    target.strokeRect(10 * C.dpr, C.canvas.height - 28 * C.dpr, tw, 20 * C.dpr);
    target.fillStyle = "#e6edf7"; target.textAlign = "left"; target.textBaseline = "middle";
    target.fillText(hud, 16 * C.dpr, C.canvas.height - 18 * C.dpr);
  }

  // Streamlines
  let _stream = null;

  function insideShape(x, y) {
    const poly = C.shape;
    if (!poly || poly.length < 3) return false;
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const xi = poly[i][0], yi = poly[i][1];
      const xj = poly[j][0], yj = poly[j][1];
      if (((yi > y) !== (yj > y)) &&
          (x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-30) + xi)) {
        inside = !inside;
      }
    }
    return inside;
  }

  function seedParticle(r) {
    for (let tries = 0; tries < 30; tries++) {
      const i = Math.floor(Math.random() * r.nx);
      const j = Math.floor(Math.random() * r.ny);
      if (r.mask[j * r.nx + i]) continue;
      const x = (i + Math.random()) * r.grid.dx;
      const y = (j + Math.random()) * r.grid.dy;
      if (insideShape(x, y)) continue;
      return { x, y, age: 0, maxAge: 60 + Math.random() * 60 };
    }
    return null;
  }
  function sampleVel(r, x, y) {
    const i = Math.floor(x / r.grid.dx);
    const j = Math.floor(y / r.grid.dy);
    if (i < 0 || i >= r.nx || j < 0 || j >= r.ny) return null;
    const k = j * r.nx + i;
    if (r.mask[k]) return null;
    return [r.u[k], r.v[k]];
  }
  function advectRK2(r, p, dt) {
    const v1 = sampleVel(r, p.x, p.y);
    if (!v1) return false;
    const xm = p.x + 0.5 * dt * v1[0];
    const ym = p.y + 0.5 * dt * v1[1];
    const v2 = sampleVel(r, xm, ym) || v1;
    p.x += dt * v2[0]; p.y += dt * v2[1];
    return true;
  }
  function applyFrame(r, idx) {
    const N = r.nx * r.ny;
    r.u = r.frames.u.subarray(idx * N, (idx + 1) * N);
    r.v = r.frames.v.subarray(idx * N, (idx + 1) * N);
    // p-frames may be absent (omitted to keep the payload small); leave r.p as
    // the final-step snapshot in that case — Pressure renders as a static map.
    if (r.frames.p) {
      r.p = r.frames.p.subarray(idx * N, (idx + 1) * N);
    }
    r.vel_mag = null;     // recompute next time it is read
    r.vorticity = null;   // ditto — per-frame curl
  }

  function streamFrame() {
    if (!_stream) return;
    const s = _stream;
    const { r, base, trail, trailCtx, particles, vmax } = s;
    const dt = s.dt;

    if (r.frames) {
      const elapsed = performance.now() - s.animStart;
      const idx = Math.floor(elapsed / s.msPerFrame) % r.frames.n;
      if (idx !== s.lastIdx) {
        applyFrame(r, idx);
        renderBase(base.getContext("2d"), r, C.viewOpts.field, s.fieldRange);
        s.lastIdx = idx;
      }
    }

    trailCtx.globalCompositeOperation = "destination-out";
    trailCtx.fillStyle = "rgba(0,0,0,0.08)";
    trailCtx.fillRect(0, 0, trail.width, trail.height);
    trailCtx.globalCompositeOperation = "source-over";
    trailCtx.lineWidth = 1.2 * C.dpr;
    trailCtx.lineCap = "round";

    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      const ox = p.x, oy = p.y;
      const ok = advectRK2(r, p, dt);
      p.age++;
      const oob = p.x < 0 || p.x >= C.tunnel.width || p.y < 0 || p.y >= C.tunnel.height;
      if (!ok || oob || p.age > p.maxAge) {
        const np = seedParticle(r);
        if (np) particles[i] = np;
        continue;
      }
      const vv = sampleVel(r, p.x, p.y);
      const s2 = vv ? Math.hypot(vv[0], vv[1]) : 0;
      const t = Math.min(1, s2 / vmax);
      const a = 0.35 + 0.55 * t;
      trailCtx.strokeStyle = `rgba(255,255,255,${a})`;
      const [px0, py0] = C.worldToCanvas([ox, oy]);
      const [px1, py1] = C.worldToCanvas([p.x, p.y]);
      trailCtx.beginPath();
      trailCtx.moveTo(px0, py0);
      trailCtx.lineTo(px1, py1);
      trailCtx.stroke();
    }

    C.ctx.clearRect(0, 0, C.canvas.width, C.canvas.height);
    C.ctx.drawImage(base, 0, 0);
    C.ctx.drawImage(trail, 0, 0);
    drawObstacleOutline(C.ctx, r);
    _stream.raf = requestAnimationFrame(streamFrame);
  }

  function buildStream(r, wantParticles) {
    let fieldRange = null;
    let vmax = 0;
    if (r.frames) {
      applyFrame(r, 0);
      const N = r.nx * r.ny;
      const uAll = r.frames.u, vAll = r.frames.v;
      // vmax (for quiver/particle scaling) always uses |u|
      for (let f = 0; f < r.frames.n; f++) {
        const off = f * N;
        for (let k = 0; k < N; k++) {
          if (r.mask[k]) continue;
          const s = Math.hypot(uAll[off + k], vAll[off + k]);
          if (s > vmax) vmax = s;
        }
      }
      // Stable color range across frames for the currently-selected field.
      fieldRange = frameFieldStableRange(r, C.viewOpts.field);
    } else {
      for (let k = 0; k < r.u.length; k++) {
        if (r.mask[k]) continue;
        const s = Math.hypot(r.u[k], r.v[k]);
        if (s > vmax) vmax = s;
      }
    }
    vmax = Math.max(vmax, 1e-6);

    const base = document.createElement("canvas");
    base.width = C.canvas.width; base.height = C.canvas.height;
    renderBase(base.getContext("2d"), r, C.viewOpts.field, fieldRange);

    const trail = document.createElement("canvas");
    trail.width = C.canvas.width; trail.height = C.canvas.height;
    const trailCtx = trail.getContext("2d");

    const dt = 1.5 / (C.transform.scale * vmax);
    const particles = [];
    if (wantParticles) {
      const N = Math.min(3000, Math.max(500, Math.round(r.nx * r.ny * 0.08)));
      for (let i = 0; i < N; i++) {
        const p = seedParticle(r);
        if (p) { p.age = Math.random() * p.maxAge; particles.push(p); }
      }
    }
    return {
      r, base, trail, trailCtx, particles, dt, vmax, raf: 0,
      fieldRange, animStart: performance.now(), msPerFrame: 33, lastIdx: 0,
    };
  }

  C.stopStreamlines = function () {
    if (_stream && _stream.raf) cancelAnimationFrame(_stream.raf);
    _stream = null;
  };

  C.drawViewScene = function () {
    C.computeTransform();
    C.stopStreamlines();
    const r = C.result;
    if (!r) { C.ctx.clearRect(0, 0, C.canvas.width, C.canvas.height); return; }
    if (r.frames || C.viewOpts.showStreams) {
      _stream = buildStream(r, C.viewOpts.showStreams);
      C.ctx.drawImage(_stream.base, 0, 0);
      _stream.raf = requestAnimationFrame(streamFrame);
    } else {
      renderBase(C.ctx, r, C.viewOpts.field);
    }
  };

  C.initViewHandlers = function () {
    document.getElementById("field-select").onchange = (e) => {
      C.viewOpts.field = e.target.value;
      C.redraw();
    };
    document.getElementById("show-arrows").onchange = (e) => {
      C.viewOpts.showArrows = e.target.checked;
      C.redraw();
    };
    document.getElementById("show-streams").onchange = (e) => {
      C.viewOpts.showStreams = e.target.checked;
      C.redraw();
    };
  };
})();
