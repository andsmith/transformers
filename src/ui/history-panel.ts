/**
 * History panel: a small, roughly square, always-full-range plot of the
 * per-epoch loss (train + test). No zoom, no controls, no x labels — a global
 * reminder of overall progress, regardless of how the main loss plot is
 * zoomed or configured.
 */

import type { AppContext } from "../state";
import type { LossPoint } from "../training/loop";
import type { PanelHandle } from "./top-panel";

const TRAIN_COLOR = "#2b7cff";
const TEST_COLOR = "#ff6b35";

export function mountHistoryPanel(host: HTMLElement, ctx: AppContext): PanelHandle {
  host.classList.add("panel", "history-panel");
  host.innerHTML = "";

  const title = document.createElement("div");
  title.className = "panel-header";
  title.textContent = "History";

  const canvas = document.createElement("canvas");
  canvas.className = "history-canvas";
  host.append(title, canvas);
  const g = canvas.getContext("2d")!;

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

    const data = ctx.state.loop.epochHistory;
    if (data.length === 0) {
      g.fillStyle = "#9aa7b4";
      g.font = "10px system-ui, sans-serif";
      g.textAlign = "center";
      g.fillText("per-epoch loss", w / 2, h / 2);
      g.textAlign = "left";
      return;
    }

    let maxLoss = 0;
    for (const p of data) maxLoss = Math.max(maxLoss, p.trainLoss, p.testLoss ?? 0);
    const top = maxLoss > 0 ? maxLoss * 1.05 : 1;
    const minX = data[0].x;
    const spanX = Math.max(1, data[data.length - 1].x - minX);
    const xOf = (x: number) => pad + ((x - minX) / spanX) * plotW;
    const yOf = (v: number) => pad + plotH - (v / top) * plotH;

    const plotLine = (color: string, pick: (p: LossPoint) => number | null) => {
      g.strokeStyle = color;
      g.lineWidth = 1.25;
      g.beginPath();
      let started = false;
      for (const p of data) {
        const v = pick(p);
        if (v === null) continue;
        if (!started) {
          g.moveTo(xOf(p.x), yOf(v));
          started = true;
        } else {
          g.lineTo(xOf(p.x), yOf(v));
        }
      }
      g.stroke();
    };
    // A single point won't draw a line; mark it.
    if (data.length === 1) {
      g.fillStyle = TRAIN_COLOR;
      g.fillRect(xOf(data[0].x) - 1.5, yOf(data[0].trainLoss) - 1.5, 3, 3);
    } else {
      plotLine(TRAIN_COLOR, (p) => p.trainLoss);
      plotLine(TEST_COLOR, (p) => p.testLoss);
    }
  }

  function update(): void {
    draw();
  }

  update();
  return { update };
}
