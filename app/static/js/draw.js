// Canvas drawing + scene rendering for the draw/BC steps.
(function () {
  const C = (window.CFD = window.CFD || {});

  C.computeTransform = function () {
    const W = C.canvas.width, H = C.canvas.height;
    const ww = C.tunnel.width, wh = C.tunnel.height;
    const pad = (C.immersive ? 6 : 24) * C.dpr;
    const s = Math.min((W - 2 * pad) / ww, (H - 2 * pad) / wh);
    const ox = (W - s * ww) / 2;
    const oy = (H - s * wh) / 2;
    C.transform = { scale: s, ox, oy };
  };

  C.worldToCanvas = function (p) {
    const { scale, ox, oy } = C.transform;
    // World y=0 at bottom, canvas y grows downward -> flip.
    return [ox + p[0] * scale, oy + (C.tunnel.height - p[1]) * scale];
  };
  C.canvasToWorld = function (px) {
    const { scale, ox, oy } = C.transform;
    return [(px[0] - ox) / scale, C.tunnel.height - (px[1] - oy) / scale];
  };

  function drawTunnel(ctx) {
    const p0 = C.worldToCanvas([0, 0]);
    const p1 = C.worldToCanvas([C.tunnel.width, C.tunnel.height]);
    const x = Math.min(p0[0], p1[0]), y = Math.min(p0[1], p1[1]);
    const w = Math.abs(p1[0] - p0[0]), h = Math.abs(p1[1] - p0[1]);
    ctx.strokeStyle = "#35507a";
    ctx.lineWidth = 2 * C.dpr;
    ctx.strokeRect(x, y, w, h);
  }

  function drawShape(ctx, poly, closed, color) {
    if (!poly || poly.length < 2) return;
    ctx.beginPath();
    poly.forEach((p, i) => {
      const [x, y] = C.worldToCanvas(p);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    if (closed) ctx.closePath();
    ctx.fillStyle = "rgba(245,158,11,0.16)";
    if (closed) ctx.fill();
    ctx.strokeStyle = color || "#f59e0b";
    ctx.lineWidth = 2 * C.dpr;
    ctx.stroke();
  }

  C.FIDELITY = { coarse: [200, 100], medium: [320, 160], fine: [480, 240] };

  function drawGridPreview(ctx) {
    const sel = document.getElementById("fidelity");
    const [nx, ny] = C.FIDELITY[sel ? sel.value : "medium"] || [240, 120];
    const dx = C.tunnel.width / nx;
    const dy = C.tunnel.height / ny;
    const { scale } = C.transform;
    const [x0, y0] = C.worldToCanvas([0, C.tunnel.height]);
    const wPx = C.tunnel.width * scale;
    const hPx = C.tunnel.height * scale;

    // Solid-body fill: just fill the polygon exactly. Grid lines are drawn
    // on top afterwards so the resolution is still visible through the fill.
    if (C.shape) {
      ctx.beginPath();
      C.shape.forEach((p, i) => {
        const [px, py] = C.worldToCanvas(p);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.closePath();
      ctx.fillStyle = "rgba(245,158,11,0.55)";
      ctx.fill();
    }

    // Only draw lines if they won't be too dense to see (>=3px/cell).
    const pxPerCellX = dx * scale;
    const pxPerCellY = dy * scale;
    ctx.strokeStyle = "rgba(120,150,200,0.22)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    const stepX = pxPerCellX < 3 ? Math.ceil(3 / pxPerCellX) : 1;
    const stepY = pxPerCellY < 3 ? Math.ceil(3 / pxPerCellY) : 1;
    for (let i = 0; i <= nx; i += stepX) {
      const px = x0 + i * dx * scale;
      ctx.moveTo(px, y0); ctx.lineTo(px, y0 + hPx);
    }
    for (let j = 0; j <= ny; j += stepY) {
      const py = y0 + j * dy * scale;
      ctx.moveTo(x0, py); ctx.lineTo(x0 + wPx, py);
    }
    ctx.stroke();
  }

  C.drawDrawScene = function () {
    C.computeTransform();
    const ctx = C.ctx;
    ctx.clearRect(0, 0, C.canvas.width, C.canvas.height);
    drawTunnel(ctx);
    if (C.step === "solve") drawGridPreview(ctx);
    if (C.step === "bc") C.drawBCOverlays(ctx);
    const path = C.drawingPath || C.shape;
    if (path) drawShape(ctx, path, !!C.shape, "#f59e0b");
  };

  // Douglas-Peucker polyline simplification (in world coords).
  C.simplifyDP = function (points, eps) {
    if (points.length < 3) return points.slice();
    const n = points.length;
    const keep = new Uint8Array(n);
    keep[0] = 1; keep[n - 1] = 1;
    const stack = [[0, n - 1]];
    while (stack.length) {
      const [a, b] = stack.pop();
      let maxD = 0, idx = -1;
      for (let i = a + 1; i < b; i++) {
        const d = perpDist(points[i], points[a], points[b]);
        if (d > maxD) { maxD = d; idx = i; }
      }
      if (maxD > eps && idx > 0) {
        keep[idx] = 1;
        stack.push([a, idx]); stack.push([idx, b]);
      }
    }
    return points.filter((_, i) => keep[i]);
  };
  function perpDist(p, a, b) {
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const L = Math.hypot(dx, dy) || 1e-12;
    return Math.abs(dx * (a[1] - p[1]) - (a[0] - p[0]) * dy) / L;
  }

  // Clamp polygon vertices to stay inside the tunnel box.
  C.clipToTunnel = function (poly) {
    const pad = 0.01;
    return poly.map(([x, y]) => [
      Math.max(pad, Math.min(C.tunnel.width - pad, x)),
      Math.max(pad, Math.min(C.tunnel.height - pad, y)),
    ]);
  };

  C.initDrawHandlers = function () {
    const canvas = C.canvas;
    let activeId = null;
    let tapStart = null;   // {x, y, t} in client coords
    let isDrag = false;
    const TAP_MOVE_PX = 8;
    const TAP_MAX_MS = 350;

    function evtToCanvas(e) {
      const rect = canvas.getBoundingClientRect();
      return [(e.clientX - rect.left) * C.dpr, (e.clientY - rect.top) * C.dpr];
    }

    canvas.addEventListener("pointerdown", (e) => {
      if (activeId !== null) return;
      activeId = e.pointerId;
      tapStart = { x: e.clientX, y: e.clientY, t: Date.now() };
      isDrag = false;
      canvas.setPointerCapture(e.pointerId);
      if (C.step === "draw") {
        const w = C.canvasToWorld(evtToCanvas(e));
        C.drawingPath = [w];
        C.shape = null;
        C.redraw();
      }
      e.preventDefault();
    });
    canvas.addEventListener("pointermove", (e) => {
      if (e.pointerId !== activeId) return;
      if (!isDrag) {
        const dx = e.clientX - tapStart.x, dy = e.clientY - tapStart.y;
        if (Math.hypot(dx, dy) > TAP_MOVE_PX) isDrag = true;
      }
      if (C.step === "draw" && isDrag && C.drawingPath) {
        const w = C.canvasToWorld(evtToCanvas(e));
        C.drawingPath.push(w);
        C.redraw();
      }
    });
    const endDraw = (e) => {
      if (e.pointerId !== activeId) return;
      activeId = null;
      try { canvas.releasePointerCapture(e.pointerId); } catch (_) {}
      const wasTap = !isDrag && (Date.now() - tapStart.t) < TAP_MAX_MS;

      if (wasTap) {
        // Discard any accidental single-point stroke from the draw step.
        C.drawingPath = null;
        // In BC step, a tap on a boundary opens the editor — otherwise toggle.
        let handled = false;
        if (C.step === "bc") handled = !!C.onBCTap(evtToCanvas(e));
        if (!handled) C.toggleImmersive();
        C.redraw();
        return;
      }

      // Drag: commit the drawn shape if we're in the draw step.
      if (C.step === "draw" && C.drawingPath && C.drawingPath.length >= 3) {
        const eps = 0.002 * Math.max(C.tunnel.width, C.tunnel.height);
        let s = C.simplifyDP(C.drawingPath, eps);
        s = C.clipToTunnel(s);
        C.shape = s.length >= 3 ? s : null;
      } else if (C.step === "draw") {
        C.shape = null;
      }
      C.drawingPath = null;
      C.redraw();
    };
    canvas.addEventListener("pointerup", endDraw);
    canvas.addEventListener("pointercancel", endDraw);
  };
})();
