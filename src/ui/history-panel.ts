/**
 * History panel: a small, roughly square, always-full-range plot toggling
 * between per-epoch LOSS (train + test) and TIMING (samples/sec over the run).
 * No zoom, no x labels — a global reminder of overall progress. The button in
 * the header shows (and switches) which series is displayed.
 */

import type { AppContext } from "../state";
import type { PanelHandle } from "./top-panel";

const TRAIN_COLOR = "#2b7cff";
const TEST_COLOR = "#ff6b35";
const TIMING_COLOR = "#5b3fa8";

type HistoryMode = "loss" | "timing";

export function mountHistoryPanel(host: HTMLElement, ctx: AppContext): PanelHandle {
  host.classList.add("panel", "history-panel");
  host.innerHTML = "";

  let mode: HistoryMode = "loss";

  const head = document.createElement("div");
  head.className = "history-head";
  const title = document.createElement("div");
  title.className = "panel-header";
  title.textContent = "History";
  const modeBtn = document.createElement("button");
  modeBtn.className = "collapse-btn";
  modeBtn.title = "Toggle between loss and timing history";
  modeBtn.addEventListener("click", () => {
    mode = mode === "loss" ? "timing" : "loss";
    setModeLabel();
    draw();
  });
  head.append(title, modeBtn);

  const canvas = document.createElement("canvas");
  canvas.className = "history-canvas";
  host.append(head, canvas);
  const g = canvas.getContext("2d")!;

  function setModeLabel(): void {
    modeBtn.textContent = mode === "loss" ? "loss" : "samples/s";
  }
  setModeLabel();

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

    type Pt = { x: number; ys: (number | null)[] };
    let pts: Pt[];
    let colors: string[];
    if (mode === "loss") {
      const data = ctx.state.loop.epochHistory;
      if (data.length === 0) {
        empty("per-epoch loss");
        return;
      }
      pts = data.map((p) => ({ x: p.x, ys: [p.trainLoss, p.testLoss] }));
      colors = [TRAIN_COLOR, TEST_COLOR];
    } else {
      const data = ctx.state.loop.timingHistory;
      if (data.length === 0) {
        empty("samples/sec");
        return;
      }
      pts = data.map((p) => ({ x: p.x, ys: [p.sps] }));
      colors = [TIMING_COLOR];
    }

    let maxY = 0;
    for (const p of pts) for (const v of p.ys) maxY = Math.max(maxY, v ?? 0);
    const top = maxY > 0 ? maxY * 1.05 : 1;
    const minX = pts[0].x;
    const spanX = Math.max(1, pts[pts.length - 1].x - minX);
    const xOf = (x: number) => pad + ((x - minX) / spanX) * plotW;
    const yOf = (v: number) => pad + plotH - (v / top) * plotH;

    if (pts.length === 1) {
      g.fillStyle = colors[0];
      const v = pts[0].ys[0];
      if (v !== null) g.fillRect(xOf(pts[0].x) - 1.5, yOf(v) - 1.5, 3, 3);
      return;
    }

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

  function update(): void {
    draw();
  }

  update();
  return { update };
}
