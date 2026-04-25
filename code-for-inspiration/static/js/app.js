// Main controller: state machine + canvas setup.
(function () {
  const C = (window.CFD = window.CFD || {});

  function init() {
    C.canvas = document.getElementById("canvas");
    C.ctx = C.canvas.getContext("2d");
    C.dpr = window.devicePixelRatio || 1;
    C.tunnel = { width: 2.0, height: 1.0 };
    C.shape = null;
    C.drawingPath = null;
    C.bcs = C.defaultBCs();
    C.viewOpts = { field: "mach", showArrows: false, showStreams: true };
    C.result = null;

    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);
    window.addEventListener("orientationchange", resizeCanvas);
    // Any size change to the canvas element (e.g. chrome collapsing in
    // immersive mode) must sync the drawing-buffer size. ResizeObserver
    // fires regardless of what triggered the change.
    if (typeof ResizeObserver !== "undefined") {
      new ResizeObserver(resizeCanvas).observe(C.canvas);
    }

    C.initDrawHandlers();
    C.initBCHandlers();
    C.initViewHandlers();

    C.immersive = false;
    document.getElementById("btn-immersive").onclick = () => C.toggleImmersive();

    document.getElementById("btn-clear").onclick = () => {
      C.shape = null; C.drawingPath = null; C.redraw();
    };
    document.getElementById("btn-undo").onclick = () => {
      C.shape = null; C.drawingPath = null; C.redraw();
    };
    document.getElementById("btn-draw-next").onclick = () => {
      if (!C.shape) { alert("Draw a closed shape first."); return; }
      C.setStep("bc");
    };
    document.getElementById("btn-bc-back").onclick = () => C.setStep("draw");
    document.getElementById("btn-bc-next").onclick = () => C.setStep("solve");
    document.getElementById("btn-solve-back").onclick = () => C.setStep("bc");
    document.getElementById("btn-solve").onclick = () => C.runSolve();
    document.getElementById("fidelity").onchange = () => { if (C.step === "solve") C.redraw(); };
    const modeSel = document.getElementById("mode");
    const schemeSel = document.getElementById("scheme");
    const syncModeScheme = () => {
      if (modeSel.value === "transient") {
        schemeSel.value = "explicit";
        schemeSel.disabled = true;
      } else {
        schemeSel.disabled = false;
      }
    };
    modeSel.onchange = syncModeScheme;
    syncModeScheme();
    document.getElementById("btn-view-back").onclick = () => C.setStep("bc");

    document.querySelectorAll(".steps .step").forEach((btn) => {
      btn.onclick = () => {
        const s = btn.dataset.step;
        if (s === "solve" && !C.shape) return;
        C.setStep(s);
      };
    });

    C.setStep("draw");
  }

  function resizeCanvas() {
    C.dpr = window.devicePixelRatio || 1;
    const rect = C.canvas.getBoundingClientRect();
    C.canvas.width = Math.max(1, Math.round(rect.width * C.dpr));
    C.canvas.height = Math.max(1, Math.round(rect.height * C.dpr));
    C.redraw && C.redraw();
  }

  C.setStep = function (step) {
    C.step = step;
    document.querySelectorAll(".steps .step").forEach((b) => {
      b.classList.toggle("active", b.dataset.step === step || (step === "view" && b.dataset.step === "solve"));
    });
    for (const id of ["toolbar-draw", "toolbar-bc", "toolbar-solve", "toolbar-view"]) {
      document.getElementById(id).classList.add("hidden");
    }
    const id = step === "view" ? "toolbar-view" : `toolbar-${step}`;
    document.getElementById(id).classList.remove("hidden");
    const statusMsg = {
      draw: "Draw an obstacle",
      bc: "Tap a boundary to set its condition",
      solve: "Choose fidelity and solve",
      view: "Results",
    }[step];
    document.getElementById("status").textContent = statusMsg;
    C.redraw();
  };

  C.setImmersive = function (flag) {
    C.immersive = !!flag;
    document.body.classList.toggle("immersive", C.immersive);
    // ResizeObserver will catch the size change, but trigger an immediate
    // redraw so the transform updates with the new padding without a flash.
    C.redraw();
  };
  C.toggleImmersive = function () { C.setImmersive(!C.immersive); };

  C.redraw = function () {
    if (!C.canvas) return;
    if (C.step === "view" && C.result) {
      C.drawViewScene();
    } else {
      if (C.stopStreamlines) C.stopStreamlines();
      C.drawDrawScene();
    }
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
