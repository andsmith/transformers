/**
 * Status panel (bottom-left): a 2×2 grid of live sections —
 *   model  (embedding dim, positional encoding, FF layers, heads)
 *   dataset (max input/output dims, train & test sizes)
 *   loss   (most recent sample + epoch means, prominent)
 *   timing (samples/second averaged at ≤1 Hz, seconds/epoch)
 * plus a compact progress/phase line under the title.
 */

import type { AppContext } from "../state";
import { PIPELINE_STAGES } from "../training/loop";
import { isClassification } from "../tasks/types";
import type { PanelHandle } from "./top-panel";

export function mountStatusPanel(host: HTMLElement, ctx: AppContext): PanelHandle {
  host.classList.add("panel", "status-panel");
  host.innerHTML = "";

  const title = document.createElement("div");
  title.className = "panel-header";
  title.textContent = "Status";

  const phaseLine = document.createElement("div");
  phaseLine.className = "status-line status-phase";

  const grid = document.createElement("div");
  grid.className = "status-grid";
  host.append(title, phaseLine, grid);

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

  const modelBody = section("model");
  const dataBody = section("dataset");
  const lossBody = section("loss");
  const timingBody = section("timing");

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

    // Progress / phase line.
    const progress =
      `epoch ${loop.epoch} · sample ${Math.min(loop.cursorPos + 1, loop.trainSize)}` +
      `/${loop.trainSize}`;
    if (!st) {
      phaseLine.textContent = `${progress} · idle`;
      phaseLine.className = "status-line status-phase idle";
    } else if (st.phase === "forward") {
      phaseLine.textContent = `${progress} · Forward — ${PIPELINE_STAGES[st.stage]?.title ?? ""}`;
      phaseLine.className = "status-line status-phase forward";
    } else if (st.phase === "backward") {
      phaseLine.textContent = `${progress} · Backprop — ${PIPELINE_STAGES[st.stage]?.title ?? ""}`;
      phaseLine.className = "status-line status-phase backward";
    } else {
      phaseLine.textContent = `${progress} · sample #${st.sample.index}${s.running ? " · training…" : ""}`;
      phaseLine.className = "status-line status-phase complete";
    }

    // model
    const pe = s.peScheme === "sinusoidal" ? "positional" : "learned";
    modelBody.innerHTML =
      `D<sub>embed</sub> = ${s.embedDim}<br>` +
      `P<sub>embed</sub>: ${pe}<br>` +
      `FF layers: ${s.numOutputLayers} · heads: ${s.numHeads}`;

    // dataset
    const dOut = isClassification(s.task) ? 1 : s.maxSeqLen;
    dataBody.innerHTML =
      `d<sub>input</sub>(max) = ${s.maxSeqLen} · d<sub>output</sub>(max) = ${dOut}<br>` +
      `train: ${s.dataset.train.length} · test: ${s.dataset.test.length}`;

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
