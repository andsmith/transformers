/**
 * Top panel: title, task selection, and the model + learning hyperparameters.
 * (The Step/Go run controls live as an overlay on the network view — see
 * run-controls.ts.)
 */

import type { AppContext } from "../state";
import { VERSION } from "../version";
import type { Task } from "../tasks/types";
import { ALL_TASKS, TASK_SPECS } from "../tasks/types";
import type { PEScheme } from "../model/embeddings";
import { ACT_CMAPS, WEIGHT_CMAPS } from "../viz/colormaps";
import {
  makeDropdown,
  makeFieldset,
  makeRadioGroup,
  makeSlider,
  type Dropdown,
  type RadioGroup,
  type Slider,
} from "./controls";

export interface PanelHandle {
  update(): void;
}

export function mountTopPanel(host: HTMLElement, ctx: AppContext): PanelHandle {
  host.classList.add("panel", "top-panel");
  host.innerHTML = "";

  // --- title row with collapse toggle ---
  const titleRow = document.createElement("div");
  titleRow.className = "title-row";
  const title = document.createElement("h1");
  title.className = "app-title";
  title.textContent = `Transformer Playground - Version ${VERSION}`;
  const collapseBtn = document.createElement("button");
  collapseBtn.className = "collapse-btn";
  collapseBtn.addEventListener("click", () =>
    ctx.apply({ topCollapsed: !ctx.state.topCollapsed }),
  );
  titleRow.append(title, collapseBtn);
  host.appendChild(titleRow);

  const row = document.createElement("div");
  row.className = "control-row";
  host.appendChild(row);

  // --- task ---
  // Compact text-only buttons (descriptions live in the tooltips) so all the
  // control fieldsets fit on a single row.
  const taskBox = makeFieldset("Task");
  const taskRadios: RadioGroup<Task> = makeRadioGroup(
    ALL_TASKS.map((t) => ({
      value: t,
      label: TASK_SPECS[t].label,
      title: TASK_SPECS[t].description,
    })),
    ctx.state.task,
    (t) => ctx.apply({ task: t }),
  );
  taskBox.append(taskRadios.el);
  row.appendChild(taskBox);

  // --- model ---
  const modelBox = makeFieldset("Model");
  const embedSlider: Slider = makeSlider({
    label: "Embedding dim",
    min: 2,
    max: 16,
    step: 2,
    value: ctx.state.embedDim,
    onInput: (v) => ctx.apply({ embedDim: v }),
  });
  const peRadios: RadioGroup<PEScheme> = makeRadioGroup(
    [
      { value: "sinusoidal", label: "Sinusoidal", title: "Fixed encoding (Vaswani 2017)" },
      { value: "learned", label: "Learned", title: "Trainable positional weights" },
    ],
    ctx.state.peScheme,
    (s) => ctx.apply({ peScheme: s }),
  );
  const layerRadios: RadioGroup<"1" | "2"> = makeRadioGroup(
    [
      { value: "1", label: "1 layer" },
      { value: "2", label: "2 layers" },
    ],
    String(ctx.state.numOutputLayers) as "1" | "2",
    (s) => ctx.apply({ numOutputLayers: Number(s) as 1 | 2 }),
  );
  const headRadios: RadioGroup<"1" | "2"> = makeRadioGroup(
    [
      { value: "1", label: "1 head" },
      { value: "2", label: "2 heads", disabled: true, title: "Multi-head: coming soon" },
    ],
    "1",
    () => {},
  );
  // Lay the model items out in a wrapping horizontal row to keep the panel
  // short.
  const modelRow = document.createElement("div");
  modelRow.className = "fieldset-row";
  modelRow.append(
    embedSlider.el,
    label("Positional", peRadios.el),
    label("Output", layerRadios.el),
    label("Attention", headRadios.el),
  );
  modelBox.append(modelRow);
  row.appendChild(modelBox);

  // --- learning ---
  const learnBox = makeFieldset("Learning");
  const lrSlider: Slider = makeSlider({
    label: "Learning rate",
    min: -3,
    max: 0,
    step: 0.1,
    value: Math.log10(ctx.state.learningRate),
    format: (v) => Math.pow(10, v).toPrecision(2),
    onInput: (v) => ctx.apply({ learningRate: Math.pow(10, v) }),
  });
  learnBox.append(lrSlider.el);
  row.appendChild(learnBox);

  // --- colormaps ---
  const cmapBox = makeFieldset("Colormaps");
  const cmapOptions = (names: string[]) =>
    names.map((n) => ({ value: n, label: n }));
  const weightsCmapDd: Dropdown<string> = makeDropdown(
    cmapOptions(Object.keys(WEIGHT_CMAPS)),
    ctx.state.weightsCmap,
    (n) => ctx.apply({ weightsCmap: n }),
  );
  const actsCmapDd: Dropdown<string> = makeDropdown(
    cmapOptions(Object.keys(ACT_CMAPS)),
    ctx.state.actsCmap,
    (n) => ctx.apply({ actsCmap: n }),
  );
  cmapBox.append(
    label("Weights", weightsCmapDd.el),
    label("Activations", actsCmapDd.el),
  );
  row.appendChild(cmapBox);

  function update(): void {
    const s = ctx.state;
    host.classList.toggle("collapsed", s.topCollapsed);
    row.style.display = s.topCollapsed ? "none" : "";
    collapseBtn.textContent = s.topCollapsed ? "▼ controls" : "▲ hide";
    collapseBtn.title = s.topCollapsed ? "Restore the controls" : "Collapse the controls";
    taskRadios.set(s.task);
    embedSlider.set(s.embedDim);
    peRadios.set(s.peScheme);
    layerRadios.set(String(s.numOutputLayers) as "1" | "2");
    lrSlider.set(Math.log10(s.learningRate));
    weightsCmapDd.set(s.weightsCmap);
    actsCmapDd.set(s.actsCmap);
  }

  update();
  return { update };
}

/** Wrap a control with a small caption above it. */
function label(text: string, el: HTMLElement): HTMLElement {
  const wrap = document.createElement("div");
  wrap.className = "labeled";
  const cap = document.createElement("div");
  cap.className = "caption";
  cap.textContent = text;
  wrap.append(cap, el);
  return wrap;
}
