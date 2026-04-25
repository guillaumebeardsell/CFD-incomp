// Boundary condition UI: tap-to-select + bottom-sheet editor.
(function () {
  const C = (window.CFD = window.CFD || {});

  C.defaultBCs = function () {
    return {
      inlet:    { type: "inlet_subsonic", mach: 0.3, p: 101325, T: 300 },
      outlet:   { type: "outlet_subsonic", p: 101325 },
      top:      { type: "slip_wall" },
      bottom:   { type: "slip_wall" },
      obstacle: { type: "no_slip_wall" },
    };
  };

  const TYPES = {
    inlet:    ["inlet_subsonic", "inlet_supersonic", "slip_wall", "no_slip_wall"],
    outlet:   ["outlet_subsonic", "outlet_supersonic", "slip_wall"],
    top:      ["slip_wall", "no_slip_wall", "symmetry"],
    bottom:   ["slip_wall", "no_slip_wall", "symmetry"],
    obstacle: ["no_slip_wall", "slip_wall"],
  };
  const FIELDS = {
    inlet_subsonic: [["mach", "Mach"], ["p", "p [Pa]"], ["T", "T [K]"]],
    inlet_supersonic: [["mach", "Mach"], ["p", "p [Pa]"], ["T", "T [K]"]],
    outlet_subsonic: [["p", "Back pressure [Pa]"]],
    outlet_supersonic: [],
    slip_wall: [],
    no_slip_wall: [],
    symmetry: [],
  };
  const LABEL_COLORS = {
    inlet: "#34d399", outlet: "#f87171", top: "#60a5fa", bottom: "#60a5fa",
    obstacle: "#fbbf24",
  };

  // Hit-test areas for each boundary in canvas pixels.
  function boundaryHitzones() {
    const zones = {};
    const tl = C.worldToCanvas([0, C.tunnel.height]);
    const tr = C.worldToCanvas([C.tunnel.width, C.tunnel.height]);
    const bl = C.worldToCanvas([0, 0]);
    const br = C.worldToCanvas([C.tunnel.width, 0]);
    const hit = 28 * C.dpr;
    zones.inlet  = { kind: "line", a: tl, b: bl, hit };
    zones.outlet = { kind: "line", a: tr, b: br, hit };
    zones.top    = { kind: "line", a: tl, b: tr, hit };
    zones.bottom = { kind: "line", a: bl, b: br, hit };
    if (C.shape) zones.obstacle = { kind: "poly", poly: C.shape };
    return zones;
  }

  function distToSegment(px, a, b) {
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const len2 = dx * dx + dy * dy || 1e-12;
    let t = ((px[0] - a[0]) * dx + (px[1] - a[1]) * dy) / len2;
    t = Math.max(0, Math.min(1, t));
    const cx = a[0] + t * dx, cy = a[1] + t * dy;
    return Math.hypot(px[0] - cx, px[1] - cy);
  }

  function pointInPoly(px, poly) {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = C.worldToCanvas(poly[i]);
      const [xj, yj] = C.worldToCanvas(poly[j]);
      const intersect = ((yi > px[1]) !== (yj > px[1]))
        && (px[0] < ((xj - xi) * (px[1] - yi)) / (yj - yi + 1e-12) + xi);
      if (intersect) inside = !inside;
    }
    return inside;
  }

  C.onBCTap = function (px) {
    const zones = boundaryHitzones();
    let best = null, bestD = Infinity;
    for (const key in zones) {
      const z = zones[key];
      if (z.kind === "line") {
        const d = distToSegment(px, z.a, z.b);
        if (d < z.hit && d < bestD) { best = key; bestD = d; }
      } else if (z.kind === "poly") {
        if (pointInPoly(px, z.poly)) { best = key; bestD = 0; break; }
      }
    }
    if (best) { openBCSheet(best); return true; }
    return false;
  };

  C.drawBCOverlays = function (ctx) {
    const labels = { inlet: "Inlet", outlet: "Outlet", top: "Top", bottom: "Bottom", obstacle: "Body" };
    const zones = boundaryHitzones();
    for (const key in zones) {
      const z = zones[key];
      ctx.strokeStyle = LABEL_COLORS[key] || "#888";
      ctx.lineWidth = 5 * C.dpr;
      if (z.kind === "line") {
        ctx.beginPath();
        ctx.moveTo(z.a[0], z.a[1]); ctx.lineTo(z.b[0], z.b[1]);
        ctx.stroke();
        const mx = (z.a[0] + z.b[0]) / 2;
        const my = (z.a[1] + z.b[1]) / 2;
        drawChip(ctx, mx, my, `${labels[key]}: ${bcSummary(C.bcs[key])}`);
      }
    }
  };

  function bcSummary(bc) {
    if (!bc) return "";
    if (bc.type === "inlet_subsonic" || bc.type === "inlet_supersonic")
      return `M=${bc.mach}, p=${(bc.p / 1000).toFixed(0)}kPa`;
    if (bc.type === "outlet_subsonic") return `p=${(bc.p / 1000).toFixed(0)}kPa`;
    return bc.type.replace(/_/g, " ");
  }

  function drawChip(ctx, x, y, text) {
    ctx.font = `${12 * C.dpr}px -apple-system, sans-serif`;
    const pad = 6 * C.dpr;
    const w = ctx.measureText(text).width + 2 * pad;
    const h = 18 * C.dpr;
    ctx.fillStyle = "rgba(17, 26, 46, 0.9)";
    ctx.strokeStyle = "rgba(100, 130, 200, 0.5)";
    ctx.lineWidth = 1 * C.dpr;
    const rx = x - w / 2, ry = y - h / 2;
    ctx.beginPath();
    ctx.roundRect ? ctx.roundRect(rx, ry, w, h, 6 * C.dpr) : ctx.rect(rx, ry, w, h);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = "#e6edf7";
    ctx.textBaseline = "middle"; ctx.textAlign = "center";
    ctx.fillText(text, x, y);
  }

  // ---- bottom sheet ----
  function openBCSheet(key) {
    C.selectedBC = key;
    const sheet = document.getElementById("bc-sheet");
    const title = document.getElementById("bc-sheet-title");
    title.textContent = {
      inlet: "Inlet (left)", outlet: "Outlet (right)",
      top: "Top wall", bottom: "Bottom wall",
      obstacle: "Obstacle surface",
    }[key];
    renderBCSheet(key, C.bcs[key]);
    sheet.classList.remove("hidden");
  }
  function closeBCSheet() {
    document.getElementById("bc-sheet").classList.add("hidden");
    C.selectedBC = null;
  }
  function renderBCSheet(key, bc) {
    const typeSel = document.getElementById("bc-type");
    typeSel.innerHTML = "";
    for (const t of TYPES[key]) {
      const o = document.createElement("option");
      o.value = t; o.textContent = t.replace(/_/g, " ");
      if (t === bc.type) o.selected = true;
      typeSel.appendChild(o);
    }
    typeSel.onchange = () => renderFields(typeSel.value, bc);
    renderFields(typeSel.value, bc);
  }
  function renderFields(type, bc) {
    const wrap = document.getElementById("bc-params");
    wrap.innerHTML = "";
    const defaults = { mach: 0.3, p: 101325, T: 300 };
    for (const [name, label] of FIELDS[type]) {
      const row = document.createElement("label");
      row.innerHTML = `${label}<input type="number" name="${name}" step="any" value="${bc[name] ?? defaults[name]}">`;
      wrap.appendChild(row);
    }
  }
  function applyBCSheet() {
    const key = C.selectedBC;
    const type = document.getElementById("bc-type").value;
    const updated = { type };
    document.querySelectorAll("#bc-params input").forEach((el) => {
      updated[el.name] = parseFloat(el.value);
    });
    C.bcs[key] = updated;
    closeBCSheet();
    C.redraw();
  }

  C.initBCHandlers = function () {
    document.getElementById("bc-apply").onclick = applyBCSheet;
    document.getElementById("bc-sheet-close").onclick = closeBCSheet;
    document.getElementById("btn-bc-preset").onclick = () => {
      C.bcs = C.defaultBCs();
      C.redraw();
    };
  };
})();
