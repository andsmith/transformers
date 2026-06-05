/**
 * Bottom panel: a growing line chart of training loss (per sample) and test
 * loss, with a per-iteration / per-epoch toggle. Redrawn every animation frame
 * by `main.ts` so it animates as training advances.
 */

import type { AppContext, LossView } from "../state";
import type { LossPoint } from "../training/loop";
import { makeCheckbox, makeRadioGroup, type Checkbox, type RadioGroup } from "./controls";
import type { PanelHandle } from "./top-panel";

const TRAIN_COLOR = "#2b7cff";
const TEST_COLOR = "#ff6b35";

export function mountLossPanel(host: HTMLElement, ctx: AppContext): PanelHandle {
  host.classList.add("panel", "loss-panel");
  host.innerHTML = "";

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
    (v) => ctx.apply({ lossView: v }),
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

  head.append(title, viewRadios.el, logCheck.el, legend);
  host.appendChild(head);

  const canvas = document.createElement("canvas");
  canvas.className = "loss-canvas";
  host.appendChild(canvas);
  const cssCtx = canvas.getContext("2d")!;

  function seriesFor(): LossPoint[] {
    return ctx.state.lossView === "epoch"
      ? ctx.state.loop.epochHistory
      : ctx.state.loop.iterHistory;
  }

  function draw(): void {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || host.clientWidth;
    const h = canvas.clientHeight || 120;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    cssCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    cssCtx.clearRect(0, 0, w, h);

    const pad = { l: 38, r: 8, t: 8, b: 18 };
    const plotW = w - pad.l - pad.r;
    const plotH = h - pad.t - pad.b;

    // axes
    cssCtx.strokeStyle = "#ccd6e0";
    cssCtx.lineWidth = 1;
    cssCtx.beginPath();
    cssCtx.moveTo(pad.l, pad.t);
    cssCtx.lineTo(pad.l, pad.t + plotH);
    cssCtx.lineTo(pad.l + plotW, pad.t + plotH);
    cssCtx.stroke();

    const data = seriesFor();
    if (data.length === 0) {
      cssCtx.fillStyle = "#8a98a8";
      cssCtx.font = "12px system-ui, sans-serif";
      cssCtx.fillText("No data yet — Step or Go to start training.", pad.l + 8, pad.t + 18);
      return;
    }

    let maxLoss = 0;
    let minPos = Infinity; // smallest positive value, for the log scale
    for (const p of data) {
      maxLoss = Math.max(maxLoss, p.trainLoss, p.testLoss ?? 0);
      for (const v of [p.trainLoss, p.testLoss]) {
        if (v !== null && v > 0 && v < minPos) minPos = v;
      }
    }
    const maxX = Math.max(1, data[data.length - 1].x);
    const minX = data[0].x;
    const spanX = Math.max(1, maxX - minX);
    const xOf = (x: number) => pad.l + ((x - minX) / spanX) * plotW;

    const logScale = ctx.state.lossLogScale;
    let yOf: (loss: number) => number;
    let ticks: number[];
    if (logScale) {
      if (!isFinite(minPos)) minPos = 1e-3;
      const hi = Math.log10(maxLoss > 0 ? maxLoss : 1);
      const lo = Math.min(Math.log10(minPos), hi - 1e-6);
      const span = hi - lo;
      yOf = (loss) =>
        pad.t +
        plotH -
        ((Math.log10(Math.max(loss, minPos)) - lo) / span) * plotH;
      ticks = [Math.pow(10, lo), Math.pow(10, (lo + hi) / 2), Math.pow(10, hi)];
    } else {
      const top = maxLoss > 0 ? maxLoss * 1.1 : 1;
      yOf = (loss) => pad.t + plotH - (loss / top) * plotH;
      ticks = [0, top / 2, top];
    }

    // y-axis ticks
    const fmt = (v: number) =>
      v === 0 ? "0" : v >= 0.01 ? v.toFixed(2) : v.toExponential(0);
    cssCtx.fillStyle = "#8a98a8";
    cssCtx.font = "10px system-ui, sans-serif";
    for (const v of ticks) {
      cssCtx.fillText(fmt(v), 4, yOf(v) + 3);
    }

    // Epoch boundaries (per-iteration view): thin dashed gray verticals, plus
    // floating "epoch k" labels when the epochs are wide enough to fit them.
    if (ctx.state.lossView === "iteration") {
      const trainLen = ctx.state.dataset.train.length;
      if (trainLen > 0) {
        const lineShade = "#b6c0cb";
        cssCtx.strokeStyle = lineShade;
        cssCtx.lineWidth = 1;
        cssCtx.setLineDash([3, 3]);
        for (let k = 1; k * trainLen <= maxX; k++) {
          const bx = xOf(k * trainLen);
          if (bx > pad.l && bx < pad.l + plotW) {
            cssCtx.beginPath();
            cssCtx.moveTo(bx, pad.t);
            cssCtx.lineTo(bx, pad.t + plotH);
            cssCtx.stroke();
          }
        }
        cssCtx.setLineDash([]);

        const epochPxW = (trainLen / spanX) * plotW;
        if (epochPxW >= 48) {
          cssCtx.fillStyle = lineShade;
          cssCtx.font = "10px system-ui, sans-serif";
          cssCtx.textAlign = "center";
          for (let k = 0; k * trainLen <= maxX; k++) {
            const cx = xOf((k + 0.5) * trainLen);
            if (cx > pad.l && cx < pad.l + plotW) {
              cssCtx.fillText(`epoch ${k}`, cx, pad.t + 9);
            }
          }
          cssCtx.textAlign = "left";
        }
      }
    }

    const plotLine = (color: string, pick: (p: LossPoint) => number | null) => {
      cssCtx.strokeStyle = color;
      cssCtx.lineWidth = 1.5;
      cssCtx.beginPath();
      let started = false;
      for (const p of data) {
        const v = pick(p);
        if (v === null) continue;
        const x = xOf(p.x);
        const y = yOf(v);
        if (!started) {
          cssCtx.moveTo(x, y);
          started = true;
        } else {
          cssCtx.lineTo(x, y);
        }
      }
      cssCtx.stroke();
    };

    plotLine(TRAIN_COLOR, (p) => p.trainLoss);
    plotLine(TEST_COLOR, (p) => p.testLoss);
  }

  function update(): void {
    viewRadios.set(ctx.state.lossView);
    logCheck.set(ctx.state.lossLogScale);
    draw();
  }

  update();
  return { update };
}
