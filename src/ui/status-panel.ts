/**
 * Status panel (bottom-left): live text summary of the experiment and
 * training progress — model definition, task/dataset, epoch, sample and
 * progress through the epoch, pipeline stage/pass, latest loss.
 */

import type { AppContext } from "../state";
import { modelSummary } from "../state";
import { PIPELINE_STAGES } from "../training/loop";
import { TASK_SPECS } from "../tasks/types";
import type { PanelHandle } from "./top-panel";

export function mountStatusPanel(host: HTMLElement, ctx: AppContext): PanelHandle {
  host.classList.add("panel", "status-panel");
  host.innerHTML = "";

  const title = document.createElement("div");
  title.className = "panel-header";
  title.textContent = "Status";

  const body = document.createElement("div");
  body.className = "status-body";
  host.append(title, body);

  const line = (cls = ""): HTMLDivElement => {
    const el = document.createElement("div");
    el.className = `status-line ${cls}`;
    body.appendChild(el);
    return el;
  };

  const modelLine = line();
  const dataLine = line();
  const progressLine = line();
  const phaseLine = line("status-phase");
  const lossLine = line();

  function update(): void {
    const s = ctx.state;
    const loop = s.loop;
    const st = loop.staged;

    modelLine.textContent = modelSummary(s);
    dataLine.textContent =
      `${TASK_SPECS[s.task].kind} · ${s.dataset.train.length} train / ` +
      `${s.dataset.test.length} test · seed ${s.seed}${s.randomSeed ? " (random)" : ""}`;
    progressLine.textContent =
      `epoch ${loop.epoch} · sample ${Math.min(loop.cursorPos + 1, loop.trainSize)}` +
      `/${loop.trainSize} · iter ${loop.iteration}`;

    if (!st) {
      phaseLine.textContent = "idle — press Step to begin";
      phaseLine.className = "status-line status-phase idle";
      lossLine.textContent = "";
      return;
    }
    const stageName = PIPELINE_STAGES[st.stage]?.title ?? "";
    if (st.phase === "forward") {
      phaseLine.textContent = `Forward Pass — ${stageName}`;
      phaseLine.className = "status-line status-phase forward";
    } else if (st.phase === "backward") {
      phaseLine.textContent = `Backpropagation Pass — ${stageName}`;
      phaseLine.className = "status-line status-phase backward";
    } else {
      phaseLine.textContent = s.running ? "training…" : "iteration complete";
      phaseLine.className = "status-line status-phase complete";
    }
    lossLine.textContent = `sample #${st.sample.index} · loss ${st.lossValue.data.toFixed(4)}`;
  }

  update();
  return { update };
}
