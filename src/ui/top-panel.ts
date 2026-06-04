/**
 * Top panel: title, task selection, model + learning hyperparameters, and the
 * Step/Go control with its granularity dropdown.
 */

import type { AppContext } from "../state";
import type { Task } from "../tasks/types";
import { ALL_TASKS, TASK_SPECS } from "../tasks/types";
import type { PEScheme } from "../model/embeddings";
import type { StepGranularity } from "../training/loop";
import { MAX_VOCAB } from "../tasks/grammar";
import {
  makeButton,
  makeDropdown,
  makeFieldset,
  makeRadioCards,
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

  // --- title ---
  const title = document.createElement("h1");
  title.className = "app-title";
  title.textContent = "Transformer Playground";
  host.appendChild(title);

  const row = document.createElement("div");
  row.className = "control-row";
  host.appendChild(row);

  // --- task ---
  const taskBox = makeFieldset("Task");
  const taskRadios: RadioGroup<Task> = makeRadioCards(
    ALL_TASKS.map((t) => ({
      value: t,
      title: TASK_SPECS[t].label,
      description: TASK_SPECS[t].description,
    })),
    ctx.state.task,
    (t) => ctx.apply({ task: t }),
  );
  taskBox.append(taskRadios.el);
  row.appendChild(taskBox);

  // --- model ---
  const modelBox = makeFieldset("Model");
  const symbolsSlider: Slider = makeSlider({
    label: "Symbols (|V|)",
    min: 2,
    max: MAX_VOCAB,
    step: 1,
    value: ctx.state.numSymbols,
    onInput: (v) => ctx.apply({ numSymbols: v }),
  });
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
  modelBox.append(symbolsSlider.el, embedSlider.el, label("Positional", peRadios.el), label("Output", layerRadios.el), label("Attention", headRadios.el));
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

  // --- run ---
  const runBox = makeFieldset("Run");
  const granDropdown: Dropdown<StepGranularity> = makeDropdown(
    [
      { value: "layer", label: "1 layer" },
      { value: "iteration", label: "1 iteration" },
      { value: "epoch", label: "1 epoch" },
      { value: "run", label: "Run continuously" },
    ],
    ctx.state.stepGranularity,
    (g) => ctx.apply({ stepGranularity: g }),
  );
  const stepBtn = makeButton("Step", () => ctx.step());
  const counters = document.createElement("div");
  counters.className = "hint";
  runBox.append(granDropdown.el, stepBtn, counters);
  row.appendChild(runBox);

  function update(): void {
    const s = ctx.state;
    taskRadios.set(s.task);
    symbolsSlider.set(s.numSymbols);
    embedSlider.set(s.embedDim);
    peRadios.set(s.peScheme);
    layerRadios.set(String(s.numOutputLayers) as "1" | "2");
    lrSlider.set(Math.log10(s.learningRate));
    granDropdown.set(s.stepGranularity);
    stepBtn.textContent =
      s.stepGranularity === "run" ? (s.running ? "Stop" : "Go") : "Step";
    counters.textContent = `iter ${s.loop.iteration} · epoch ${s.loop.epoch}`;
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
