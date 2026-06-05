/**
 * History panel: a small, roughly square, always-full-range per-epoch plot.
 * The source button cycles loss → s/epoch → test-set hits/epoch; a "log"
 * checkbox transforms the y-axis; the y extremes are labeled at the top and
 * bottom of the plot in every mode. The loss view marks the best (lowest)
 * test epoch with the same colored star as the main loss plot.
 */

import type { AppContext } from "../state";
import type { PanelHandle } from "./top-panel";

const TRAIN_COLOR = "#2b7cff";
const TEST_COLOR = "#ff6b35";
const SECS_COLOR = "#5b3fa8";
const HITS_COLOR = "#b58900";

type HistoryMode = "loss" | "secs" | "hits";
const MODE_LABEL: Record<HistoryMode, string> = {
  loss: "loss",
  secs: "s/epoch",
  hits: "hits/epoch",
};
const NEXT_MODE: Record<HistoryMode, HistoryMode> = {
  loss: "secs",
  secs: "hits",
  hits: "loss",
};

function drawStar(
  g: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  r: number,
  color: string,
): void {
  g.fillStyle = color;
  g.strokeStyle = "#444";
  g.lineWidth = 0.75;
  g.beginPath();
  for (let i = 0; i < 10; i++) {
    const rad = i % 2 === 0 ? r : r * 0.45;
    const a = -Math.PI / 2 + (i * Math.PI) / 5;
    if (i === 0) g.moveTo(cx + rad * Math.cos(a), cy + rad * Math.sin(a));
    else g.lineTo(cx + rad * Math.cos(a), cy + rad * Math.sin(a));
  }
  g.closePath();
  g.fill();
  g.stroke();
}

export function mountHistoryPanel(host: HTMLElement, ctx: AppContext): PanelHandle {
  host.classList.add("panel", "history-panel");
  host.innerHTML = "";

  let mode: HistoryMode = "loss";
  let logY = false;

  const head = document.createElement("div");
  head.className = "history-head";
  const title = document.createElement("div");
  title.className = "panel-header";
  title.textContent = "History";

  const logLabel = document.createElement("label");
  logLabel.className = "checkbox";
  const logInput = document.createElement("input");
  logInput.type = "checkbox";
  logInput.addEventListener("change", () => {
    logY = logInput.checked;
    draw();
  });
  const logText = document.createElement("span");
  logText.textContent = "log";
  logLabel.append(logInput, logText);

  const modeBtn = document.createElement("button");
  modeBtn.className = "collapse-btn";
  modeBtn.title = "Cycle source: loss → s/epoch → test-set hits/epoch";
  modeBtn.addEventListener("click", () => {
    mode = NEXT_MODE[mode];
    modeBtn.textContent = MODE_LABEL[mode];
    draw();
  });
  modeBtn.textContent = MODE_LABEL[mode];

  head.append(title, logLabel, modeBtn);

  const canvas = document.createElement("canvas");
  canvas.className = "history-canvas";
  host.append(head, canvas);
  const g = canvas.getContext("2d")!;

  const fmt = (v: number): string =>
    v === 0 ? "0" : v >= 100 ? v.toFixed(0) : v >= 0.01 ? v.toFixed(2) : v.toExponential(0);

  function draw(): void {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (w <= 0 || h <= 0) return;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);

    const pad = 4;
    const plotW = w - 2 * pad;
    const plotH = h - 2 * pad;
    g.strokeStyle = "#ccd6e0";
    g.lineWidth = 1;
    g.strokeRect(pad + 0.5, pad + 0.5, plotW - 1, plotH - 1);

    const empty = (msg: string) => {
      g.fillStyle = "#9aa7b4";
      g.font = "10px system-ui, sans-serif";
      g.textAlign = "center";
      g.fillText(msg, w / 2, h / 2);
      g.textAlign = "left";
    };

    // --- collect the active series ---
    type Pt = { x: number; ys: (number | null)[] };
    let pts: Pt[];
    let colors: string[];
    if (mode === "loss") {
      const data = ctx.state.loop.epochHistory;
      if (data.length === 0) return empty("per-epoch loss");
      pts = data.map((p) => ({ x: p.x, ys: [p.trainLoss, p.testLoss] }));
      colors = [TRAIN_COLOR, TEST_COLOR];
    } else if (mode === "secs") {
      const data = ctx.state.loop.epochStats;
      if (data.length === 0) return empty("seconds / epoch");
      pts = data.map((p) => ({ x: p.x, ys: [p.seconds] }));
      colors = [SECS_COLOR];
    } else {
      const data = ctx.state.loop.epochStats;
      if (data.length === 0) return empty("test-set hits / epoch");
      pts = data.map((p) => ({ x: p.x, ys: [p.hits] }));
      colors = [HITS_COLOR];
    }

    // --- y scale (linear, or log when possible) ---
    let maxY = 0;
    let minPos = Infinity;
    for (const p of pts) {
      for (const v of p.ys) {
        if (v === null || v === undefined) continue;
        maxY = Math.max(maxY, v);
        if (v > 0 && v < minPos) minPos = v;
      }
    }
    const useLog = logY && isFinite(minPos) && maxY > 0;
    let yOf: (v: number) => number;
    let yTop: number;
    let yBottom: number;
    if (useLog) {
      const hi = Math.log10(maxY);
      const lo = Math.min(Math.log10(minPos), hi - 1e-6);
      const span = hi - lo;
      yOf = (v) => pad + plotH - ((Math.log10(Math.max(v, minPos)) - lo) / span) * plotH;
      yTop = maxY;
      yBottom = minPos;
    } else {
      const top = maxY > 0 ? maxY * 1.05 : 1;
      yOf = (v) => pad + plotH - (v / top) * plotH;
      yTop = top;
      yBottom = 0;
    }

    const minX = pts[0].x;
    const spanX = Math.max(1, pts[pts.length - 1].x - minX);
    const xOf = (x: number) => pad + ((x - minX) / spanX) * plotW;

    // --- series ---
    if (pts.length === 1) {
      const v = pts[0].ys[0];
      if (v !== null && v !== undefined) {
        g.fillStyle = colors[0];
        g.fillRect(xOf(pts[0].x) - 1.5, yOf(v) - 1.5, 3, 3);
      }
    } else {
      for (let s = 0; s < colors.length; s++) {
        g.strokeStyle = colors[s];
        g.lineWidth = 1.25;
        g.beginPath();
        let started = false;
        for (const p of pts) {
          const v = p.ys[s];
          if (v === null || v === undefined) continue;
          if (!started) {
            g.moveTo(xOf(p.x), yOf(v));
            started = true;
          } else {
            g.lineTo(xOf(p.x), yOf(v));
          }
        }
        g.stroke();
      }
    }

    // --- best-test star (loss mode), matching the main loss plot ---
    if (mode === "loss") {
      const eh = ctx.state.loop.epochHistory;
      let best: { x: number; v: number } | null = null;
      for (const p of eh) {
        if (p.testLoss !== null && (best === null || p.testLoss < best.v)) {
          best = { x: p.x, v: p.testLoss };
        }
      }
      if (best && eh.length > 0) {
        const firstX = eh[0].x;
        const lastX = eh[eh.length - 1].x;
        const frac = lastX > firstX ? (best.x - firstX) / (lastX - firstX) : 1;
        const color = frac < 1 / 3 ? "#e0413a" : frac < 2 / 3 ? "#f5c518" : "#2fbf71";
        drawStar(g, xOf(best.x), yOf(best.v), 10, color);
      }
    }

    // --- y-axis extreme labels (top + bottom, all modes) ---
    g.fillStyle = "#1f2a36";
    g.font = "9px system-ui, sans-serif";
    g.textAlign = "left";
    g.fillText(fmt(yTop), pad + 3, pad + 10);
    g.fillText(fmt(yBottom), pad + 3, pad + plotH - 4);
  }

  function update(): void {
    draw();
  }

  update();
  return { update };
}
