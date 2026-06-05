/**
 * Bottom panel: a growing line chart of training loss (per sample) and test
 * loss, with per-iteration / per-epoch toggle, log-y option, horizontal zoom,
 * and a collapse button. Redrawn every animation frame by `main.ts`.
 *
 * Zoom interaction: drag horizontally to zoom into that fixed region; a single
 * click zooms to a growing window from the clicked x to the live edge (the
 * plot keeps extending as training adds points). Hovering the plot for >1s
 * shows a hint popup. "Reset Zoom" (visible only while zoomed) restores the
 * full view.
 */

import type { AppContext, LossView } from "../state";
import type { LossPoint } from "../training/loop";
import {
  makeButton,
  makeCheckbox,
  makeRadioGroup,
  type Checkbox,
  type RadioGroup,
} from "./controls";
import type { PanelHandle } from "./top-panel";

const TRAIN_COLOR = "#2b7cff";
const TEST_COLOR = "#ff6b35";
const CLICK_TOLERANCE_PX = 5; // below this a "drag" counts as a click

export function mountLossPanel(host: HTMLElement, ctx: AppContext): PanelHandle {
  host.classList.add("panel", "loss-panel");
  host.innerHTML = "";

  // --- header row ---
  const head = document.createElement("div");
  head.className = "panel-header-row";
  const title = document.createElement("span");
  title.className = "panel-header";
  title.textContent = "Loss";

  const viewRadios: RadioGroup<LossView> = makeRadioGroup(
    [
      { value: "iteration", label: "Per iteration" },
      { value: "epoch", label: "Per epoch" },
    ],
    ctx.state.lossView,
    (v) => {
      zoom = null; // x-units change between views
      ctx.apply({ lossView: v });
    },
  );

  const logCheck: Checkbox = makeCheckbox(
    "log y",
    ctx.state.lossLogScale,
    (c) => ctx.apply({ lossLogScale: c }),
  );

  const legend = document.createElement("span");
  legend.className = "legend";
  legend.innerHTML =
    `<span class="dot" style="background:${TRAIN_COLOR}"></span>train ` +
    `<span class="dot" style="background:${TEST_COLOR}"></span>test`;

  const resetBtn = makeButton("Reset Zoom", () => {
    zoom = null;
  });
  resetBtn.classList.add("reset-zoom-btn");

  const collapseBtn = document.createElement("button");
  collapseBtn.className = "collapse-btn";
  collapseBtn.addEventListener("click", () =>
    ctx.apply({ lossCollapsed: !ctx.state.lossCollapsed }),
  );

  head.append(title, viewRadios.el, logCheck.el, legend, resetBtn, collapseBtn);
  host.appendChild(head);

  // --- canvas ---
  const canvas = document.createElement("canvas");
  canvas.className = "loss-canvas";
  canvas.style.cursor = "crosshair";
  host.appendChild(canvas);
  const g = canvas.getContext("2d")!;

  // --- hover hint ---
  const hint = document.createElement("div");
  hint.className = "plot-hint";
  hint.textContent =
    "Drag: zoom to a region · Click: zoom from here (follows latest) · Reset Zoom: full view";
  host.appendChild(hint);
  let hintTimer: number | null = null;

  function hideHint(): void {
    if (hintTimer !== null) {
      window.clearTimeout(hintTimer);
      hintTimer = null;
    }
    hint.classList.remove("visible");
  }

  canvas.addEventListener("pointerenter", () => {
    if (ctx.state.lossCollapsed) return;
    if (hintTimer !== null) window.clearTimeout(hintTimer);
    hintTimer = window.setTimeout(() => hint.classList.add("visible"), 1000);
  });
  canvas.addEventListener("pointerleave", hideHint);

  // --- zoom state ---
  /** null = fully zoomed out; x1 null = right edge follows latest data. */
  let zoom: { x0: number; x1: number | null } | null = null;
  let dragStart: number | null = null; // px, canvas coords
  let dragCur: number | null = null;
  /** Last draw's x-mapping, for px<->data conversion in event handlers. */
  let lastMap: { domMin: number; domMax: number; padL: number; plotW: number } | null = null;

  function seriesFor(): LossPoint[] {
    return ctx.state.lossView === "epoch"
      ? ctx.state.loop.epochHistory
      : ctx.state.loop.iterHistory;
  }

  const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

  function toData(px: number): number {
    const m = lastMap!;
    return m.domMin + ((px - m.padL) / m.plotW) * (m.domMax - m.domMin);
  }

  canvas.addEventListener("pointerdown", (e) => {
    if (ctx.state.lossCollapsed || !lastMap || seriesFor().length < 2) return;
    hideHint();
    dragStart = dragCur = e.offsetX;
    canvas.setPointerCapture(e.pointerId);
  });

  canvas.addEventListener("pointermove", (e) => {
    if (dragStart !== null) dragCur = e.offsetX;
  });

  const endDrag = (e: PointerEvent) => {
    if (dragStart === null || !lastMap) {
      dragStart = dragCur = null;
      return;
    }
    const a = dragStart;
    const b = dragCur ?? a;
    dragStart = dragCur = null;
    canvas.releasePointerCapture?.(e.pointerId);

    const data = seriesFor();
    if (data.length < 2) return;
    const fullMin = data[0].x;
    const fullMax = data[data.length - 1].x;

    if (Math.abs(b - a) < CLICK_TOLERANCE_PX) {
      // Click: growing window from the clicked x to the live edge.
      const x0 = clamp(toData(a), fullMin, fullMax - 1);
      zoom = { x0, x1: null };
    } else {
      // Drag: fixed region.
      let x0 = toData(Math.min(a, b));
      let x1 = toData(Math.max(a, b));
      x0 = clamp(x0, fullMin, fullMax - 1);
      x1 = clamp(x1, x0 + 1, fullMax);
      zoom = { x0, x1 };
    }
  };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);

  function draw(): void {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || host.clientWidth;
    const h = canvas.clientHeight || 120;
    if (w <= 0 || h <= 0) return;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);

    const pad = { l: 38, r: 8, t: 8, b: 18 };
    const plotW = w - pad.l - pad.r;
    const plotH = h - pad.t - pad.b;

    // axes
    g.strokeStyle = "#ccd6e0";
    g.lineWidth = 1;
    g.beginPath();
    g.moveTo(pad.l, pad.t);
    g.lineTo(pad.l, pad.t + plotH);
    g.lineTo(pad.l + plotW, pad.t + plotH);
    g.stroke();

    const data = seriesFor();
    if (data.length === 0) {
      g.fillStyle = "#8a98a8";
      g.font = "12px system-ui, sans-serif";
      g.fillText("No data yet — Step or Go to start training.", pad.l + 8, pad.t + 18);
      lastMap = null;
      return;
    }

    // --- x-domain: full range, or the zoom window ---
    const fullMin = data[0].x;
    const fullMax = Math.max(1, data[data.length - 1].x);
    let domMin = fullMin;
    let domMax = fullMax;
    if (zoom) {
      domMin = clamp(zoom.x0, fullMin, fullMax - 1e-9);
      domMax = zoom.x1 === null ? fullMax : clamp(zoom.x1, domMin + 1e-9, fullMax);
      if (domMax - domMin < 1) domMax = Math.min(fullMax, domMin + 1);
    }
    const spanX = Math.max(1, domMax - domMin);
    const xOf = (x: number) => pad.l + ((x - domMin) / spanX) * plotW;
    lastMap = { domMin, domMax, padL: pad.l, plotW };

    // --- y stats over the visible window only ---
    let maxLoss = 0;
    let minPos = Infinity; // smallest positive value, for the log scale
    let anyVisible = false;
    for (const p of data) {
      if (p.x < domMin || p.x > domMax) continue;
      anyVisible = true;
      maxLoss = Math.max(maxLoss, p.trainLoss, p.testLoss ?? 0);
      for (const v of [p.trainLoss, p.testLoss]) {
        if (v !== null && v > 0 && v < minPos) minPos = v;
      }
    }
    if (!anyVisible) {
      for (const p of data) {
        maxLoss = Math.max(maxLoss, p.trainLoss, p.testLoss ?? 0);
        for (const v of [p.trainLoss, p.testLoss]) {
          if (v !== null && v > 0 && v < minPos) minPos = v;
        }
      }
    }

    const logScale = ctx.state.lossLogScale;
    let yOf: (loss: number) => number;
    let ticks: number[];
    if (logScale) {
      if (!isFinite(minPos)) minPos = 1e-3;
      const hi = Math.log10(maxLoss > 0 ? maxLoss : 1);
      const lo = Math.min(Math.log10(minPos), hi - 1e-6);
      const span = hi - lo;
      yOf = (loss) =>
        pad.t + plotH - ((Math.log10(Math.max(loss, minPos)) - lo) / span) * plotH;
      ticks = [Math.pow(10, lo), Math.pow(10, (lo + hi) / 2), Math.pow(10, hi)];
    } else {
      const top = maxLoss > 0 ? maxLoss * 1.1 : 1;
      yOf = (loss) => pad.t + plotH - (loss / top) * plotH;
      ticks = [0, top / 2, top];
    }

    // y-axis ticks
    const fmt = (v: number) =>
      v === 0 ? "0" : v >= 0.01 ? v.toFixed(2) : v.toExponential(0);
    g.fillStyle = "#8a98a8";
    g.font = "10px system-ui, sans-serif";
    for (const v of ticks) {
      g.fillText(fmt(v), 4, yOf(v) + 3);
    }

    // Epoch boundaries (per-iteration view): thin dashed gray verticals, plus
    // floating "epoch k" labels when the epochs are wide enough to fit them.
    if (ctx.state.lossView === "iteration") {
      const trainLen = ctx.state.dataset.train.length;
      if (trainLen > 0) {
        const lineShade = "#b6c0cb";
        g.strokeStyle = lineShade;
        g.lineWidth = 1;
        g.setLineDash([3, 3]);
        for (let k = 1; k * trainLen <= fullMax; k++) {
          const bx = xOf(k * trainLen);
          if (bx > pad.l && bx < pad.l + plotW) {
            g.beginPath();
            g.moveTo(bx, pad.t);
            g.lineTo(bx, pad.t + plotH);
            g.stroke();
          }
        }
        g.setLineDash([]);

        const epochPxW = (trainLen / spanX) * plotW;
        if (epochPxW >= 48) {
          g.fillStyle = lineShade;
          g.font = "10px system-ui, sans-serif";
          g.textAlign = "center";
          for (let k = 0; k * trainLen <= fullMax; k++) {
            const cx = xOf((k + 0.5) * trainLen);
            if (cx > pad.l && cx < pad.l + plotW) {
              g.fillText(`epoch ${k}`, cx, pad.t + 9);
            }
          }
          g.textAlign = "left";
        }
      }
    }

    // --- series (clipped to the plot rect so zoom windows cut cleanly) ---
    g.save();
    g.beginPath();
    g.rect(pad.l, pad.t, plotW, plotH);
    g.clip();

    const plotLine = (color: string, pick: (p: LossPoint) => number | null) => {
      g.strokeStyle = color;
      g.lineWidth = 1.5;
      g.beginPath();
      let started = false;
      for (const p of data) {
        const v = pick(p);
        if (v === null) continue;
        const x = xOf(p.x);
        const y = yOf(v);
        if (!started) {
          g.moveTo(x, y);
          started = true;
        } else {
          g.lineTo(x, y);
        }
      }
      g.stroke();
    };

    plotLine(TRAIN_COLOR, (p) => p.trainLoss);
    plotLine(TEST_COLOR, (p) => p.testLoss);
    g.restore();

    // --- drag-selection overlay ---
    if (dragStart !== null && dragCur !== null && Math.abs(dragCur - dragStart) >= CLICK_TOLERANCE_PX) {
      const x0 = clamp(Math.min(dragStart, dragCur), pad.l, pad.l + plotW);
      const x1 = clamp(Math.max(dragStart, dragCur), pad.l, pad.l + plotW);
      g.fillStyle = "rgba(43, 124, 255, 0.15)";
      g.fillRect(x0, pad.t, x1 - x0, plotH);
      g.strokeStyle = TRAIN_COLOR;
      g.lineWidth = 1;
      g.strokeRect(x0 + 0.5, pad.t + 0.5, x1 - x0 - 1, plotH - 1);
    }
  }

  function update(): void {
    const collapsed = ctx.state.lossCollapsed;
    host.classList.toggle("collapsed", collapsed);
    canvas.style.display = collapsed ? "none" : "";
    viewRadios.el.style.display = collapsed ? "none" : "";
    logCheck.el.style.display = collapsed ? "none" : "";
    legend.style.display = collapsed ? "none" : "";
    resetBtn.style.display = !collapsed && zoom ? "" : "none";
    collapseBtn.textContent = collapsed ? "▲ show" : "▼ hide";
    collapseBtn.title = collapsed ? "Restore the loss plot" : "Collapse the loss plot";
    if (collapsed) {
      hideHint();
      return;
    }
    viewRadios.set(ctx.state.lossView);
    logCheck.set(ctx.state.lossLogScale);
    draw();
  }

  update();
  return { update };
}
