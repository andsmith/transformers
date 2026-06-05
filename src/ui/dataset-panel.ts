/**
 * Dataset panel. "Dataset Generation" (display, vocabulary, sequence length —
 * changes reset training) and "Dataset Size" (train samples/epoch, test set
 * size, regenerate/seed — changes keep the model training). Training samples
 * are drawn on the fly; the list below shows the FIXED test set (the only
 * samples with stable ids).
 */

import type { AppContext, DisplayMode } from "../state";
import { testSetMax, TRAIN_PER_EPOCH_MAX } from "../state";
import type { Example } from "../tasks/types";
import { isClassification } from "../tasks/types";
import type { TestEval } from "../training/loop";
import { MAX_SEQ_LEN_LIMIT, sampleSpaceSize, SPACE_HUGE } from "../tasks/datasets";
import { tokenChar, tokenColor, MAX_VOCAB } from "../tasks/grammar";
import {
  makeButton,
  makeCheckbox,
  makeFieldset,
  makeNumberInput,
  makeRadioGroup,
  makeRangeSlider,
  makeSlider,
  type Checkbox,
  type NumberInput,
  type RadioGroup,
  type RangeSlider,
  type Slider,
} from "./controls";
import type { PanelHandle } from "./top-panel";

export function mountDatasetPanel(host: HTMLElement, ctx: AppContext): PanelHandle {
  host.classList.add("panel", "dataset-panel");
  host.innerHTML = "";

  const header = document.createElement("div");
  header.className = "panel-header";
  header.textContent = "Dataset";
  host.appendChild(header);

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
    label: "Vocabulary size",
    min: 2,
    max: MAX_VOCAB,
    step: 1,
    value: ctx.state.numSymbols,
    onInput: (v) => ctx.apply({ numSymbols: v }),
  });

  // Live preview of the vocabulary (chars or colored squares).
  const vocabPreview = document.createElement("div");
  vocabPreview.className = "labeled";
  const vocabCap = document.createElement("div");
  vocabCap.className = "caption";
  vocabCap.textContent = "Vocabulary";
  const vocabTokens = document.createElement("div");
  vocabTokens.className = "vocab-tokens";
  vocabPreview.append(vocabCap, vocabTokens);

  const vocabRow = document.createElement("div");
  vocabRow.className = "control-pair";
  vocabRow.append(vocabSlider.el, vocabPreview);

  // Dual slider: min and max sequence length on one axis.
  const lenSlider: RangeSlider = makeRangeSlider({
    label: "Sequence length",
    min: 2,
    max: MAX_SEQ_LEN_LIMIT,
    step: 1,
    lo: ctx.state.minSeqLen,
    hi: ctx.state.maxSeqLen,
    onInput: (lo, hi) => ctx.apply({ minSeqLen: lo, maxSeqLen: hi }),
  });
  const fixedLenCheck: Checkbox = makeCheckbox(
    "All max length",
    ctx.state.fixedLength,
    (c) => ctx.apply({ fixedLength: c }),
  );
  fixedLenCheck.el.title = "Generate every sequence at exactly the maximum length";

  const lenRow = document.createElement("div");
  lenRow.className = "control-pair";
  lenRow.append(lenSlider.el, fixedLenCheck.el);
  // On-the-fly training: how many fresh samples make up one epoch.
  const trainSlider: Slider = makeSlider({
    label: "Train samples/epoch",
    min: 10,
    max: TRAIN_PER_EPOCH_MAX,
    step: 1,
    inline: true,
    value: ctx.state.trainPerEpoch,
    onInput: (v) => ctx.apply({ trainPerEpoch: v }),
  });
  // Fixed held-out test set (max dynamically clamped to 20% of the space).
  const testSlider: Slider = makeSlider({
    label: "Test set size",
    min: 0,
    max: 500,
    step: 1,
    inline: true,
    value: ctx.state.testSetSize,
    onInput: (v) => ctx.apply({ testSetSize: v }),
  });
  const spaceHint = document.createElement("div");
  spaceHint.className = "control-note";
  // Regenerate (compact) + seed entry + random-seed checkbox, one row.
  const regenBtn = makeButton("Regenerate", () => ctx.regenerate());
  regenBtn.classList.add("btn-compact");
  const seedInput: NumberInput = makeNumberInput(
    ctx.state.seed,
    (v) => ctx.apply({ seed: v }),
    "Random seed (reproducibility) — applied on Regenerate",
  );
  const randomCheck: Checkbox = makeCheckbox(
    "random",
    ctx.state.randomSeed,
    (c) => ctx.apply({ randomSeed: c }),
  );
  randomCheck.el.title = "Draw a fresh random seed on every Regenerate";
  const regenRow = document.createElement("div");
  regenRow.className = "regen-row";
  regenRow.append(regenBtn, seedInput.el, randomCheck.el);

  // --- two sub-boxes: Generation (resets training) | Size (does not) ---
  const genBox = makeFieldset("Dataset Generation");
  genBox.classList.add("sub-box");
  genBox.append(displayRadios.el, vocabRow, lenRow);

  const sizeBox = makeFieldset("Dataset Size");
  sizeBox.classList.add("sub-box");
  sizeBox.append(trainSlider.el, testSlider.el, spaceHint, regenRow);

  // --- Test Set box (its own sub-box, fills the remaining panel height) ---
  // Sort state (panel-local). null = index order.
  type SortKey = "wrong" | "first";
  let sortKey: SortKey | null = null;
  let sortDir: 1 | -1 = -1; // -1 = descending
  let outMode: "y" | "yhat" = "y";

  const sortRow = document.createElement("div");
  sortRow.className = "sort-row";
  const wrongBtn = makeButton("# wrong", () => cycleSort("wrong"));
  const firstBtn = makeButton("1st error", () => cycleSort("first"));
  wrongBtn.classList.add("sort-btn");
  firstBtn.classList.add("sort-btn");
  const evalHint = document.createElement("span");
  evalHint.className = "hint eval-hint";
  const outBtn = makeButton("out: y", () => {
    outMode = outMode === "y" ? "yhat" : "y";
    invalidateList();
  });
  outBtn.classList.add("sort-btn", "out-btn");
  sortRow.append(wrongBtn, firstBtn, evalHint, outBtn);

  function cycleSort(key: SortKey): void {
    if (sortKey !== key) {
      sortKey = key;
      sortDir = -1; // first activation: descending
    } else {
      sortDir = (sortDir === -1 ? 1 : -1) as 1 | -1;
    }
    invalidateList();
  }

  const testBox = makeFieldset("Test Set");
  testBox.classList.add("sub-box", "test-set-box");
  const examplesEl = document.createElement("div");
  examplesEl.className = "examples";
  testBox.append(sortRow, examplesEl);

  const controls = document.createElement("div");
  controls.className = "dataset-controls";
  controls.append(genBox, sizeBox, testBox);
  host.appendChild(controls);

  /** Force the (memoized) example list to rebuild next update(). */
  function invalidateList(): void {
    lastListKey = "";
  }

  function renderToken(id: number, wrong = false, tip = ""): HTMLElement {
    const s = ctx.state;
    const el =
      s.display === "squares"
        ? (() => {
            const sq = document.createElement("span");
            sq.className = "tok square";
            sq.style.backgroundColor = tokenColor(id, s.numSymbols);
            return sq;
          })()
        : (() => {
            const span = document.createElement("span");
            span.className = "tok char";
            span.textContent = tokenChar(s.task, id, s.numSymbols);
            return span;
          })();
    if (wrong) el.classList.add("wrong");
    el.title = tip || tokenChar(s.task, id, s.numSymbols);
    return el;
  }

  /** Render a token sequence; `marks[i]` true outlines position i as wrong. */
  function renderSeq(ids: number[], marks?: boolean[], pTrue?: number[]): HTMLElement {
    const row = document.createElement("span");
    row.className = "seq";
    ids.forEach((id, i) => {
      const tip = pTrue ? `p(true)=${pTrue[i].toFixed(2)}` : "";
      row.appendChild(renderToken(id, marks?.[i] ?? false, tip));
    });
    return row;
  }

  /** Re-render the vocabulary preview (all token ids, current display mode). */
  function renderVocab(): void {
    vocabTokens.innerHTML = "";
    for (let id = 0; id < ctx.state.numSymbols; id++) {
      vocabTokens.appendChild(renderToken(id));
    }
  }

  /** Render the test set, marking last-epoch correctness when available. */
  function renderList(list: Example[]): void {
    const classification = isClassification(ctx.state.task);
    const frag = document.createDocumentFragment();
    const evalMap = new Map<number, TestEval>();
    for (const e of ctx.state.loop.lastTestEval ?? []) evalMap.set(e.index, e);
    const haveEval = evalMap.size > 0;
    const showHat = outMode === "yhat" && haveEval;

    // Order rows: by index unless an eval-based sort is active.
    const sorted = [...list];
    if (sortKey && haveEval) {
      const keyOf = (ex: Example) => {
        const ev = evalMap.get(ex.index);
        if (!ev) return { primary: -1, conf: 0 };
        const primary = sortKey === "wrong" ? ev.wrong : ev.firstWrong === Infinity ? 1e9 : ev.firstWrong;
        return { primary, conf: ev.meanPTrue };
      };
      sorted.sort((a, b) => {
        const ka = keyOf(a);
        const kb = keyOf(b);
        if (ka.primary !== kb.primary) return (ka.primary - kb.primary) * sortDir;
        return (ka.conf - kb.conf) * sortDir; // tie-break by confidence
      });
    } else {
      sorted.sort((a, b) => a.index - b.index);
    }

    // Fixed input-column width so every arrow lines up vertically.
    const maxInLen = sorted.reduce((m, e) => Math.max(m, e.input.length), 1);
    const perTok = ctx.state.display === "squares" ? 16 : 14;
    const inColW = `${maxInLen * perTok}px`;

    // Column header: "Input → Output".
    const head = document.createElement("div");
    head.className = "examples-header";
    const headIdx = document.createElement("span");
    headIdx.className = "ex-index";
    const headIn = document.createElement("span");
    headIn.textContent = "Input";
    headIn.style.minWidth = inColW;
    head.append(headIdx, headIn);
    const headOut = document.createElement("span");
    headOut.textContent = showHat ? "ŷ" : "Output";
    if (classification) {
      headOut.className = "head-right";
      head.append(headOut);
    } else {
      const headArrow = document.createElement("span");
      headArrow.className = "arrow";
      headArrow.textContent = "→";
      head.append(headArrow, headOut);
    }
    frag.appendChild(head);

    for (const ex of sorted) {
      const ev = evalMap.get(ex.index);
      const rowEl = document.createElement("div");
      rowEl.className = "example-row";

      const idx = document.createElement("span");
      idx.className = "ex-index";
      if (ev) idx.classList.add(ev.wrong === 0 ? "all-right" : "has-wrong");
      idx.textContent = `#${ex.index}`;
      const inSeq = renderSeq(ex.input);
      inSeq.style.minWidth = inColW;
      rowEl.append(idx, inSeq);

      if (classification) {
        const trueLabel = ex.output[0] === 1;
        const showLabel = showHat && ev ? ev.pred[0] === 1 : trueLabel;
        const tag = document.createElement("span");
        const ok = ev ? ev.correct[0] : true;
        tag.className =
          `class-tag ${showLabel ? "balanced" : "unbalanced"}` +
          (ev ? (ok ? " ok" : " bad") : "");
        tag.textContent =
          (showLabel ? "balanced" : "unbalanced") + (ev ? (ok ? " ✓" : " ✗") : "");
        if (ev) tag.title = `p(true)=${ev.pTrue[0].toFixed(2)}`;
        rowEl.append(tag);
      } else {
        const arrow = document.createElement("span");
        arrow.className = "arrow";
        arrow.textContent = "→";
        const outIds = showHat && ev ? ev.pred : ex.output;
        const marks = ev ? ev.correct.map((c) => !c) : undefined;
        rowEl.append(arrow, renderSeq(outIds, marks, ev?.pTrue));
      }
      frag.appendChild(rowEl);
    }
    examplesEl.innerHTML = "";
    examplesEl.appendChild(frag);
  }

  // The example list can be large (up to 5000 rows), so only rebuild it when
  // something that affects it actually changes.
  let lastDataset = ctx.state.dataset;
  let lastListKey = "";

  function update(): void {
    const s = ctx.state;
    displayRadios.set(s.display);
    vocabSlider.set(s.numSymbols);
    lenSlider.set(s.minSeqLen, s.maxSeqLen);
    fixedLenCheck.set(s.fixedLength);
    trainSlider.set(s.trainPerEpoch);

    // Dynamic test-set cap: 20% of the theoretical sample space (≤500).
    const maxTest = testSetMax(s);
    testSlider.setMax(maxTest);
    testSlider.set(s.testSetSize);
    const space = sampleSpaceSize(s.numSymbols, s.minSeqLen, s.maxSeqLen, s.fixedLength);
    const fmtSpace =
      space >= SPACE_HUGE
        ? "≥10¹⁵"
        : space >= 1e6
          ? space.toExponential(1)
          : String(Math.round(space));
    // What share of the whole space the test set occupies.
    const pct = (s.dataset.test.length / space) * 100;
    const fmtPct =
      pct >= 0.01 || pct === 0 ? `${pct.toFixed(2)}%` : `${pct.toExponential(1)}%`;
    spaceHint.textContent =
      `sample space ≈ ${fmtSpace}, test set ${fmtPct}` +
      (maxTest < 500 ? ` (capped at ${maxTest})` : "");

    seedInput.set(s.seed);
    seedInput.el.disabled = s.randomSeed;
    randomCheck.set(s.randomSeed);
    renderVocab();

    // Sort controls: enabled only once an epoch eval exists.
    const haveEval = !!s.loop.lastTestEval;
    wrongBtn.disabled = !haveEval;
    firstBtn.disabled = !haveEval;
    outBtn.disabled = !haveEval;
    const arrow = (k: typeof sortKey) =>
      sortKey === k ? (sortDir === -1 ? " ▼" : " ▲") : "";
    wrongBtn.textContent = `# wrong${arrow("wrong")}`;
    wrongBtn.classList.toggle("active", sortKey === "wrong");
    firstBtn.textContent = `1st error${arrow("first")}`;
    firstBtn.classList.toggle("active", sortKey === "first");
    outBtn.textContent = outMode === "yhat" ? "out: ŷ" : "out: y";
    evalHint.textContent = haveEval ? `eval @ epoch ${s.loop.lastTestEvalEpoch}` : "no eval yet";

    const listKey = `${s.display}|${sortKey ?? ""}|${sortDir}|${outMode}|${s.loop.lastTestEvalEpoch}`;
    if (s.dataset !== lastDataset || listKey !== lastListKey) {
      renderList(s.dataset.test);
      lastDataset = s.dataset;
      lastListKey = listKey;
    }
  }

  update();
  return { update };
}
