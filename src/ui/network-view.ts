/**
 * Center panel: the main network visualization area.
 *
 * PLACEHOLDER for this milestone — it draws a framed canvas with a "coming
 * soon" message and the current model shape. This is where the activation
 * color maps / Hinton diagrams and the backprop weight-update animation will
 * render next.
 */

import type { AppContext } from "../state";
import { TASK_SPECS } from "../tasks/types";
import type { PanelHandle } from "./top-panel";

export function mountNetworkView(host: HTMLElement, ctx: AppContext): PanelHandle {
  host.classList.add("panel", "network-view");
  host.innerHTML = "";

  const canvas = document.createElement("canvas");
  canvas.className = "network-canvas";
  host.appendChild(canvas);
  const g = canvas.getContext("2d")!;

  function draw(): void {
    const dpr = window.devicePixelRatio || 1;
    const w = host.clientWidth;
    const h = host.clientHeight;
    if (w === 0 || h === 0) return;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);

    g.fillStyle = "#9aa7b4";
    g.textAlign = "center";
    g.font = "600 18px system-ui, sans-serif";
    g.fillText("Network visualization", w / 2, h / 2 - 14);

    g.font = "13px system-ui, sans-serif";
    g.fillStyle = "#b6c0cb";
    g.fillText("(activations & backprop animation — coming soon)", w / 2, h / 2 + 8);

    const s = ctx.state;
    const spec = TASK_SPECS[s.task];
    const shape =
      `${spec.label} · |V|=${s.numSymbols} · d=${s.embedDim} · ` +
      `${s.peScheme} PE · ${s.numHeads} head · ${model_params(ctx)} params`;
    g.font = "12px ui-monospace, monospace";
    g.fillStyle = "#8a98a8";
    g.fillText(shape, w / 2, h / 2 + 30);
    g.textAlign = "left";
  }

  function update(): void {
    draw();
  }

  update();
  return { update };
}

function model_params(ctx: AppContext): number {
  return ctx.state.model.store.all().length;
}
