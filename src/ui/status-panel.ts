/**
 * Status panel (bottom-left). Title carries the progress ("Status - epoch e",
 * plus "- iter i" outside epoch modes). Below it, stacked sections whose
 * titles (and underline) span the full panel width:
 *   loss   — single column: last sample / last epoch train / last epoch test /
 *            best epoch test
 *   timing — two lines: "Samples/s: x · s/epoch: y" and
 *            "Test-set hits/epoch: n" (rejected training draws)
 */

import type { AppContext } from "../state";
import type { PanelHandle } from "./top-panel";

export function mountStatusPanel(host: HTMLElement, ctx: AppContext): PanelHandle {
  host.classList.add("panel", "status-panel");
  host.innerHTML = "";

  const title = document.createElement("div");
  title.className = "panel-header";
  host.appendChild(title);

  const section = (name: string, twoCol: boolean): HTMLDivElement => {
    const wrap = document.createElement("div");
    wrap.className = "status-section";
    const cap = document.createElement("div");
    cap.className = "status-section-title";
    cap.textContent = name;
    const body = document.createElement("div");
    body.className = `status-items${twoCol ? " two-col" : ""}`;
    wrap.append(cap, body);
    host.appendChild(wrap);
    return body;
  };

  const lossBody = section("loss", false);
  const timingBody = section("timing", false);

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

    // Epoch-only title while actively running an epoch mode; otherwise (and
    // always when paused) include the iteration.
    const byEpoch = s.stepGranularity === "epoch" || s.stepGranularity === "epochs";
    title.textContent =
      byEpoch && s.running
        ? `Status - epoch ${loop.epoch}`
        : `Status - epoch ${loop.epoch} - iteration ${loop.iteration}`;

    // --- loss (single column, prominent numbers) ---
    const lastIterPt = loop.iterHistory[loop.iterHistory.length - 1];
    const lastEpochPt = loop.epochHistory[loop.epochHistory.length - 1];
    const sampleLoss = st ? st.lossValue : lastIterPt?.trainLoss;
    let bestTest: { x: number; v: number } | null = null;
    for (const p of loop.epochHistory) {
      if (p.testLoss !== null && (bestTest === null || p.testLoss < bestTest.v)) {
        bestTest = { x: p.x, v: p.testLoss };
      }
    }
    lossBody.innerHTML =
      `<div>Last sample: <span class="status-big">${fmtLoss(sampleLoss)}</span></div>` +
      `<div>Last epoch (train): <span class="status-big">${fmtLoss(lastEpochPt?.trainLoss)}</span></div>` +
      `<div>Last epoch (test): <span class="status-big">${fmtLoss(lastEpochPt?.testLoss)}</span></div>` +
      `<div>Best epoch (test): <span class="status-big">${bestTest ? fmtLoss(bestTest.v) : "—"}</span>` +
      `${bestTest ? `<span class="status-dim"> @ ep ${bestTest.x}</span>` : ""}</div>`;

    // --- timing (two columns) ---
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
        ? secPerEpoch.toFixed(secPerEpoch >= 100 ? 0 : 1)
        : sps > 0
          ? `~${(loop.trainSize / sps).toFixed(1)}`
          : "—";
    const hits =
      loop.lastEpochRejections >= 0
        ? String(loop.lastEpochRejections)
        : `${loop.rejectionsThisEpoch}…`;
    timingBody.innerHTML =
      `<div>Samples/s: <span class="status-big">${spsText}</span>` +
      `<span class="status-dim"> · </span>s/epoch: <span class="status-big">${epochText}</span></div>` +
      `<div title="Training draws rejected for colliding with the test set">` +
      `Test-set hits/epoch: <span class="status-big">${hits}</span></div>`;
  }

  update();
  return { update };
}
