/**
 * Right-hand dataset panel: display mode, vocabulary/size/split sliders, a
 * regenerate button, and a rendering of the generated examples (input/output
 * pairs for transduction tasks, or two labelled groups for classification).
 */

import type { AppContext, DisplayMode } from "../state";
import type { Example } from "../tasks/types";
import { isClassification } from "../tasks/types";
import { datasetSummary, MAX_SEQ_LEN_LIMIT } from "../tasks/datasets";
import { tokenChar, tokenColor, MAX_VOCAB } from "../tasks/grammar";
import {
  makeButton,
  makeCheckbox,
  makeRadioGroup,
  makeSlider,
  type Checkbox,
  type RadioGroup,
  type Slider,
} from "./controls";
import type { PanelHandle } from "./top-panel";

/** Cap on how many examples we draw, to keep the DOM light. */
const MAX_SHOWN = 60;

export function mountDatasetPanel(host: HTMLElement, ctx: AppContext): PanelHandle {
  host.classList.add("panel", "dataset-panel");
  host.innerHTML = "";

  const header = document.createElement("div");
  header.className = "panel-header";
  header.textContent = "Dataset";
  host.appendChild(header);

  const summary = document.createElement("div");
  summary.className = "hint";
  host.appendChild(summary);

  // --- controls ---
  const displayRadios: RadioGroup<DisplayMode> = makeRadioGroup(
    [
      { value: "chars", label: "Characters" },
      { value: "squares", label: "Colored squares" },
    ],
    ctx.state.display,
    (d) => ctx.apply({ display: d }),
  );

  const vocabSlider: Slider = makeSlider({
    label: "Symbols (|V|)",
    min: 2,
    max: MAX_VOCAB,
    step: 1,
    value: ctx.state.numSymbols,
    onInput: (v) => ctx.apply({ numSymbols: v }),
  });
  const maxLenSlider: Slider = makeSlider({
    label: "Max sequence length",
    min: 2,
    max: MAX_SEQ_LEN_LIMIT,
    step: 1,
    value: ctx.state.maxSeqLen,
    onInput: (v) => ctx.apply({ maxSeqLen: v }),
  });
  const fixedLenCheck: Checkbox = makeCheckbox(
    "All sequences max length",
    ctx.state.fixedLength,
    (c) => ctx.apply({ fixedLength: c }),
  );
  const countSlider: Slider = makeSlider({
    label: "Examples",
    min: 10,
    max: 1000,
    step: 10,
    value: ctx.state.numExamples,
    onInput: (v) => ctx.apply({ numExamples: v }),
  });
  const splitSlider: Slider = makeSlider({
    label: "Train / test split",
    min: 0,
    max: 50,
    step: 1,
    value: Math.round(ctx.state.trainTestSplit * 100),
    format: (v) => `${v}% test`,
    onInput: (v) => ctx.apply({ trainTestSplit: v / 100 }),
  });
  const regenBtn = makeButton("Regenerate", () => ctx.regenerate());

  const controls = document.createElement("div");
  controls.className = "dataset-controls";
  controls.append(
    displayRadios.el,
    vocabSlider.el,
    maxLenSlider.el,
    fixedLenCheck.el,
    countSlider.el,
    splitSlider.el,
    regenBtn,
  );
  host.appendChild(controls);

  // --- examples ---
  const examplesEl = document.createElement("div");
  examplesEl.className = "examples";
  host.appendChild(examplesEl);

  function renderToken(id: number): HTMLElement {
    const s = ctx.state;
    if (s.display === "squares") {
      const sq = document.createElement("span");
      sq.className = "tok square";
      sq.style.backgroundColor = tokenColor(id, s.numSymbols);
      sq.title = tokenChar(s.task, id, s.numSymbols);
      return sq;
    }
    const span = document.createElement("span");
    span.className = "tok char";
    span.textContent = tokenChar(s.task, id, s.numSymbols);
    return span;
  }

  function renderSeq(ids: number[]): HTMLElement {
    const row = document.createElement("span");
    row.className = "seq";
    for (const id of ids) row.appendChild(renderToken(id));
    return row;
  }

  function renderTransduction(list: Example[]): void {
    for (const ex of list.slice(0, MAX_SHOWN)) {
      const row = document.createElement("div");
      row.className = "example-row";
      const arrow = document.createElement("span");
      arrow.className = "arrow";
      arrow.textContent = "→";
      row.append(renderSeq(ex.input), arrow, renderSeq(ex.output));
      examplesEl.appendChild(row);
    }
  }

  function renderClassification(list: Example[]): void {
    const balanced = list.filter((e) => e.output[0] === 1);
    const unbalanced = list.filter((e) => e.output[0] === 0);
    const half = Math.floor(MAX_SHOWN / 2);

    const group = (titleText: string, items: Example[]): HTMLElement => {
      const col = document.createElement("div");
      col.className = "class-group";
      const t = document.createElement("div");
      t.className = "caption";
      t.textContent = `${titleText} (${items.length})`;
      col.appendChild(t);
      for (const ex of items.slice(0, half)) {
        const row = document.createElement("div");
        row.className = "example-row";
        row.appendChild(renderSeq(ex.input));
        col.appendChild(row);
      }
      return col;
    };

    const cols = document.createElement("div");
    cols.className = "class-columns";
    cols.append(group("Balanced", balanced), group("Unbalanced", unbalanced));
    examplesEl.appendChild(cols);
  }

  function update(): void {
    const s = ctx.state;
    summary.textContent = datasetSummary(s.dataset);
    displayRadios.set(s.display);
    vocabSlider.set(s.numSymbols);
    maxLenSlider.set(s.maxSeqLen);
    fixedLenCheck.set(s.fixedLength);
    countSlider.set(s.numExamples);
    splitSlider.set(Math.round(s.trainTestSplit * 100));

    examplesEl.innerHTML = "";
    if (isClassification(s.task)) renderClassification(s.dataset.examples);
    else renderTransduction(s.dataset.examples);
  }

  update();
  return { update };
}
