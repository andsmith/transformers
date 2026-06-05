/**
 * Status panel (bottom-left): title carries the progress ("Status - epoch e"
 * when running by epochs, "... - iter i" otherwise), and below it two columns
 * of live text — timing (left: samples/s averaged at ≤1 Hz, seconds/epoch) and
 * loss (right: most recent sample + epoch means, prominent).
 */

import type { AppContext } from "../state";
import type { PanelHandle } from "./top-panel";

export function mountStatusPanel(host: HTMLElement, ctx: AppContext): PanelHandle {
  host.classList.add("panel", "status-panel");
  host.innerHTML = "";

  const title = document.createElement("div");
  title.className = "panel-header";

  const grid = document.createElement("div");
  grid.className = "status-grid";
  host.append(title, grid);

  const section = (name: string): HTMLDivElement => {
    const cell = document.createElement("div");
    cell.className = "status-cell";
    const cap = document.createElement("div");
    cap.className = "status-cell-title";
    cap.textContent = name;
    const body = document.createElement("div");
    body.className = "status-cell-body";
    cell.append(cap, body);
    grid.appendChild(cell);
    return body;
  };

  const timingBody = section("timing");
  const lossBody = section("loss");

  // --- timing accumulators (sampled at most once per second) ---
  let lastT = performance.now();
  let lastIter = 0;
  let sps = 0;
  let lastEpoch = 0;
  let lastEpochT = performance.now();
  let secPerEpoch = 0;

  const fmtLoss = (v: number | null | undefined): string =>
    v === null || v === undefined ? "—" : v >= 0.01 ? v.toFixed(3) : v.toExponential(1);

  function update(): void {
    const s = ctx.state;
    const loop = s.loop;
    const st = loop.staged;

    const byEpoch = s.stepGranularity === "epoch" || s.stepGranularity === "epochs";
    title.textContent = byEpoch
      ? `Status - epoch ${loop.epoch}`
      : `Status - epoch ${loop.epoch} - iter ${loop.iteration}`;

    // loss (prominent numbers)
    const lastIterPt = loop.iterHistory[loop.iterHistory.length - 1];
    const lastEpochPt = loop.epochHistory[loop.epochHistory.length - 1];
    const sampleLoss = st ? st.lossValue.data : lastIterPt?.trainLoss;
    lossBody.innerHTML =
      `sample <span class="status-big">${fmtLoss(sampleLoss)}</span><br>` +
      `epoch <span class="status-big">${fmtLoss(lastEpochPt?.trainLoss)}</span>` +
      `<span class="status-dim"> / test ${fmtLoss(lastEpochPt?.testLoss)}</span>`;

    // timing — refresh the rate at most once per second.
    const now = performance.now();
    if (now - lastT >= 1000) {
      sps = ((loop.iteration - lastIter) * 1000) / (now - lastT);
      lastIter = loop.iteration;
      lastT = now;
    }
    if (loop.epoch !== lastEpoch) {
      secPerEpoch = (now - lastEpochT) / 1000 / Math.max(1, loop.epoch - lastEpoch);
      lastEpoch = loop.epoch;
      lastEpochT = now;
    }
    const spsText = sps > 0 ? sps.toFixed(sps >= 100 ? 0 : 1) : "—";
    const epochText =
      secPerEpoch > 0
        ? `${secPerEpoch.toFixed(secPerEpoch >= 100 ? 0 : 1)} s`
        : sps > 0
          ? `~${(loop.trainSize / sps).toFixed(1)} s`
          : "—";
    timingBody.innerHTML = `${spsText} samples/s<br>${epochText} / epoch`;
  }

  update();
  return { update };
}
