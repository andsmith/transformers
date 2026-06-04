/**
 * Bottom panel: a growing line chart of training loss (per sample) and test
 * loss, with a per-iteration / per-epoch toggle. Redrawn every animation frame
 * by `main.ts` so it animates as training advances.
 */

import type { AppContext, LossView } from "../state";
import type { LossPoint } from "../training/loop";
import { makeRadioGroup, type RadioGroup } from "./controls";
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

  const legend = document.createElement("span");
  legend.className = "legend";
  legend.innerHTML =
    `<span class="dot" style="background:${TRAIN_COLOR}"></span>train ` +
    `<span class="dot" style="background:${TEST_COLOR}"></span>test`;

  head.append(title, viewRadios.el, legend);
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
    for (const p of data) {
      maxLoss = Math.max(maxLoss, p.trainLoss, p.testLoss ?? 0);
    }
    maxLoss = maxLoss > 0 ? maxLoss * 1.1 : 1;
    const maxX = Math.max(1, data[data.length - 1].x);
    const minX = data[0].x;
    const spanX = Math.max(1, maxX - minX);

    const xOf = (x: number) => pad.l + ((x - minX) / spanX) * plotW;
    const yOf = (loss: number) => pad.t + plotH - (loss / maxLoss) * plotH;

    // y-axis ticks
    cssCtx.fillStyle = "#8a98a8";
    cssCtx.font = "10px system-ui, sans-serif";
    for (let i = 0; i <= 2; i++) {
      const v = (maxLoss * i) / 2;
      const y = yOf(v);
      cssCtx.fillText(v.toFixed(2), 4, y + 3);
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
    draw();
  }

  update();
  return { update };
}
