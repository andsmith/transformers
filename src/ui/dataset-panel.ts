/**
 * Two left-column frames driven by one mount:
 * - the Dataset frame (`genHost`): "Sequence options" + "Test Set Options".
 * - the Test Set frame (`testHost`): grok filters, sort controls, and the
 *   indexed list of the FIXED held-out test set with per-epoch eval marks.
 * Training samples are drawn on the fly (no stable ids); only test samples are
 * listed.
 */

import type { AppContext, DisplayMode } from "../state";
import { maxNestingDepth, testSetMax, TRAIN_PER_EPOCH_MAX } from "../state";
import type { Example } from "../tasks/types";
import { isClassification, TASK_SPECS } from "../tasks/types";
import type { TestEval } from "../training/loop";
import { MAX_SEQ_LEN_LIMIT, sampleSpaceSize, SPACE_HUGE } from "../tasks/datasets";
import { compileFilters, randomRegexes } from "../tasks/grok";
import { prepareDemos, type PreparedDemo } from "../tasks/demos";
import { tokenChar, tokenColor, MAX_VOCAB, maxDelims } from "../tasks/grammar";
import {
  makeButton,
  makeCheckbox,
  makeFieldset,
  makeNumberInput,
  makeRadioGroup,
  makeRangeSlider,
  makeSlider,
  makeTextInput,
  type Checkbox,
  type NumberInput,
  type RadioGroup,
  type RangeSlider,
  type Slider,
  type TextInput,
} from "./controls";
import type { PanelHandle } from "./top-panel";

export function mountDatasetPanel(
  genHost: HTMLElement,
  testHost: HTMLElement,
  ctx: AppContext,
): PanelHandle {
  genHost.classList.add("panel", "dataset-panel");
  genHost.innerHTML = "";
  testHost.classList.add("panel", "test-set-panel");
  testHost.innerHTML = "";

  const header = document.createElement("div");
  header.className = "panel-header";
  header.textContent = "Dataset";
  genHost.appendChild(header);

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

  const uniformLenCheck: Checkbox = makeCheckbox(
    "Uniform length prior",
    ctx.state.uniformLen,
    (c) => ctx.apply({ uniformLen: c }),
  );
  uniformLenCheck.el.title =
    "On: every length equally likely (short sequences over-represented). " +
    "Off: length ∝ |V|^L — uniform over the whole sample space.";

  const demoCheck: Checkbox = makeCheckbox(
    "Demo Examples",
    ctx.state.demoExamples,
    (c) => ctx.apply({ demoExamples: c }),
  );
  demoCheck.el.title =
    "Swap the test set for a curated set of demonstration examples for this " +
    "task — train on real data first, then step through these to see how the " +
    "trained model works.";

  // --- task-dependent options (only parens for now) ---
  const delimSlider: Slider = makeSlider({
    label: "Delimiter kinds",
    min: 1,
    max: maxDelims(ctx.state.numSymbols),
    step: 1,
    inline: true,
    value: ctx.state.parensDelims,
    onInput: (v) => ctx.apply({ parensDelims: v }),
  });
  delimSlider.el.title =
    "Number of distinct matched delimiter pairs (the rest of the vocabulary " +
    "becomes distractors). Up to 5, limited by ⌊|V|/2⌋.";
  delimSlider.el.classList.add("narrow");
  const depthSlider: Slider = makeSlider({
    label: "Max nesting depth",
    min: 1,
    max: maxNestingDepth(ctx.state.maxSeqLen),
    step: 1,
    inline: true,
    value: ctx.state.parensMaxDepth,
    onInput: (v) => ctx.apply({ parensMaxDepth: v }),
  });
  depthSlider.el.classList.add("narrow");
  const noMixedCheck: Checkbox = makeCheckbox(
    "No mixed nesting",
    ctx.state.parensNoMixedNesting,
    (c) => ctx.apply({ parensNoMixedNesting: c }),
  );
  noMixedCheck.el.title =
    "Forbid mixing delimiter types within a nest (e.g. no '[()]', but '(())[[]]' is fine)";
  const parensOptions = makeFieldset("Parens Options");
  parensOptions.classList.add("task-options");
  // Stacked: delimiter kinds, then nesting depth, then no-mixed under it.
  parensOptions.append(delimSlider.el, depthSlider.el, noMixedCheck.el);

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

  // --- grokking: held-out subset by regex over the vocab glyphs ---
  const grokInput: TextInput = makeTextInput(
    ctx.state.grokFilters,
    "regexes, comma-separated (e.g. aa, ^a.*b$)",
    (v) => ctx.apply({ grokFilters: v }),
  );
  grokInput.el.classList.add("grok-input");
  const grokClearBtn = makeButton("✕", () => {
    grokInput.set("");
    ctx.apply({ grokFilters: "" });
  });
  grokClearBtn.classList.add("btn-compact", "grok-clear-btn");
  grokClearBtn.title = "Clear the grok filters";
  const grokRandomBtn = makeButton("🎲", () => {
    const r = randomRegexes(ctx.state.task, ctx.state.numSymbols, ctx.state.parensDelims);
    grokInput.set(r);
    ctx.apply({ grokFilters: r });
  });
  grokRandomBtn.classList.add("btn-compact", "grok-rand-btn");
  grokRandomBtn.title = "Generate a random set of filters";
  const grokRow = document.createElement("div");
  grokRow.className = "labeled";
  const grokCap = document.createElement("div");
  grokCap.className = "caption";
  grokCap.textContent = "Grok Filters (held out from training)";
  const grokInputRow = document.createElement("div");
  grokInputRow.className = "grok-row";
  grokInputRow.append(grokClearBtn, grokInput.el, grokRandomBtn);
  grokRow.append(grokCap, grokInputRow);

  const grokStatus = document.createElement("div");
  grokStatus.className = "control-note grok-status";

  // --- Dataset frame: two sub-boxes ---
  const genBox = makeFieldset("Sequence options");
  genBox.classList.add("sub-box");
  genBox.append(
    displayRadios.el,
    vocabRow,
    lenRow,
    uniformLenCheck.el,
    demoCheck.el,
    parensOptions,
  );

  // --- demonstration controls: replace the "Test Set Options" body in demo
  //     mode (selector, cycle buttons, name + description, Apply setup). ---
  // Selected demo + prepared list are recomputed each update(); handlers read
  // these panel-local snapshots.
  let curPrepared: PreparedDemo[] = [];
  let curSelIdx = 0;

  function cycleDemo(dir: number): void {
    const n = curPrepared.length;
    if (n === 0) return;
    ctx.apply({ demoIndex: (((curSelIdx + dir) % n) + n) % n });
  }

  const demoSelect = document.createElement("select");
  demoSelect.className = "dropdown demo-select";
  demoSelect.addEventListener("change", () =>
    ctx.apply({ demoIndex: Number(demoSelect.value) }),
  );
  const demoPrev = makeButton("◀", () => cycleDemo(-1));
  const demoNext = makeButton("▶", () => cycleDemo(1));
  demoPrev.classList.add("btn-compact", "demo-cycle");
  demoNext.classList.add("btn-compact", "demo-cycle");
  const demoSelRow = document.createElement("div");
  demoSelRow.className = "demo-select-row";
  demoSelRow.append(demoPrev, demoSelect, demoNext);

  const demoName = document.createElement("div");
  demoName.className = "demo-name";
  const demoDesc = document.createElement("div");
  demoDesc.className = "demo-desc";
  const demoHint = document.createElement("div");
  demoHint.className = "demo-hint";

  const applySetupBtn = makeButton("Apply setup", () => {
    const sel = curPrepared[curSelIdx];
    if (sel?.setup) ctx.apply({ ...sel.setup });
  });
  applySetupBtn.classList.add("btn-compact", "apply-setup-btn");
  applySetupBtn.title =
    "Reconfigure the model to the settings this demo is designed for " +
    "(resets training — then train and step through).";
  const demoUpdateCheck: Checkbox = makeCheckbox(
    "Update weights",
    ctx.state.demoUpdateWeights,
    (c) => ctx.apply({ demoUpdateWeights: c }),
  );
  demoUpdateCheck.el.title =
    "Off: stepping shows the forward + gradient sweep but leaves the trained " +
    "weights unchanged (inspection). On: the demos are used as training data.";
  const demoActions = document.createElement("div");
  demoActions.className = "demo-actions";
  demoActions.append(applySetupBtn, demoUpdateCheck.el);

  const demoControls = document.createElement("div");
  demoControls.className = "demo-controls";
  demoControls.append(demoSelRow, demoName, demoDesc, demoHint, demoActions);

  const normalControls = document.createElement("div");
  normalControls.className = "size-controls";
  normalControls.append(trainSlider.el, testSlider.el, spaceHint, regenRow);

  const sizeBox = makeFieldset("Test Set Options");
  sizeBox.classList.add("sub-box");
  const sizeTitle = sizeBox.querySelector(".fieldset-title") as HTMLElement;
  sizeBox.append(normalControls, demoControls);

  const controls = document.createElement("div");
  controls.className = "dataset-controls";
  controls.append(genBox, sizeBox);
  genHost.appendChild(controls);

  // --- Test Set frame (its own panel): grok controls, sorting, the list ---
  // View state (panel-local).
  type SortKey = "input" | "wrong" | "first";
  let sortKey: SortKey | null = null; // null = index order
  let sortDir: 1 | -1 = -1; // -1 = descending
  let outMode: "y" | "yhat" = "y";
  // The eval snapshot the list is currently rendered against. While "frozen"
  // (after the user modifies the view) it stays pinned so training can
  // continue without the rows shifting; the refresh button pulls the latest.
  let viewEval: TestEval[] | null = null;
  let viewEvalEpoch = -1;
  let frozen = false;

  /** Freeze on first user modification, snapshotting the latest eval. */
  function freezeView(): void {
    if (!frozen) {
      frozen = true;
      viewEval = ctx.state.loop.lastTestEval;
      viewEvalEpoch = ctx.state.loop.lastTestEvalEpoch;
    }
  }

  const sortRow = document.createElement("div");
  sortRow.className = "sort-row";
  const inputBtn = makeButton("input", () => cycleSort("input"));
  const wrongBtn = makeButton("# wrong", () => cycleSort("wrong"));
  const firstBtn = makeButton("1st error", () => cycleSort("first"));
  for (const b of [inputBtn, wrongBtn, firstBtn]) b.classList.add("sort-btn");
  const outBtn = makeButton("out: y", () => {
    freezeView();
    outMode = outMode === "y" ? "yhat" : "y";
    update();
  });
  outBtn.classList.add("sort-btn", "out-btn");
  sortRow.append(inputBtn, wrongBtn, firstBtn, outBtn);

  function cycleSort(key: SortKey): void {
    freezeView();
    if (sortKey !== key) {
      sortKey = key;
      sortDir = -1; // first activation: descending
    } else {
      sortDir = (sortDir === -1 ? 1 : -1) as 1 | -1;
    }
    update();
  }

  // Frame header: dynamic title text + a refresh button (active when stale).
  const testTitle = document.createElement("div");
  testTitle.className = "panel-header test-title-row";
  const titleText = document.createElement("span");
  const refreshBtn = makeButton("⟳", () => {
    // Pull the latest eval into the (still-frozen) view.
    viewEval = ctx.state.loop.lastTestEval;
    viewEvalEpoch = ctx.state.loop.lastTestEvalEpoch;
    update();
  });
  refreshBtn.classList.add("refresh-btn");
  refreshBtn.title = "Refresh the test-set view to the latest evaluation";
  testTitle.append(titleText, refreshBtn);

  const examplesEl = document.createElement("div");
  examplesEl.className = "examples";

  // Grok controls sit at the top of the Test Set frame, above the sort row.
  testHost.append(testTitle, grokRow, grokStatus, sortRow, examplesEl);

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
            span.textContent = tokenChar(s.task, id, s.numSymbols, s.parensDelims);
            return span;
          })();
    if (wrong) el.classList.add("wrong");
    el.title = tip || tokenChar(s.task, id, s.numSymbols, s.parensDelims);
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

  /** Re-render the vocabulary preview; tokens are click-to-copy (so the
   *  bracket glyphs can be pasted into a grok filter). */
  function renderVocab(): void {
    const s = ctx.state;
    vocabTokens.innerHTML = "";
    for (let id = 0; id < s.numSymbols; id++) {
      const el = renderToken(id);
      const ch = tokenChar(s.task, id, s.numSymbols, s.parensDelims);
      el.classList.add("copyable");
      el.title = `Click to copy "${ch}"`;
      el.addEventListener("click", () => {
        navigator.clipboard?.writeText(ch);
        el.classList.add("copied");
        window.setTimeout(() => el.classList.remove("copied"), 500);
      });
      vocabTokens.appendChild(el);
    }
  }

  /** Lexical comparison of two token sequences. */
  function lexCmp(a: number[], b: number[]): number {
    const n = Math.min(a.length, b.length);
    for (let i = 0; i < n; i++) if (a[i] !== b[i]) return a[i] - b[i];
    return a.length - b.length;
  }

  /** Render the test set, marking the displayed eval's correctness. */
  function renderList(list: Example[]): void {
    const classification = isClassification(ctx.state.task);
    const frag = document.createDocumentFragment();
    const evalMap = new Map<number, TestEval>();
    for (const e of viewEval ?? []) evalMap.set(e.index, e);
    const haveEval = evalMap.size > 0;
    const showHat = outMode === "yhat" && haveEval;

    // Order rows. "input" needs no eval; "wrong"/"first" do.
    const sorted = [...list];
    if (sortKey === "input") {
      sorted.sort((a, b) => lexCmp(a.input, b.input) * sortDir);
    } else if (sortKey && haveEval) {
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

  /** Render the demo set in authored order: all demos (valid + greyed invalid),
   *  the selected one highlighted, valid ones carrying the latest eval marks. */
  function renderDemoList(list: PreparedDemo[], selIdx: number): void {
    const classification = isClassification(ctx.state.task);
    const frag = document.createDocumentFragment();
    const evalMap = new Map<number, TestEval>();
    for (const e of viewEval ?? []) evalMap.set(e.index, e);
    const haveEval = evalMap.size > 0;
    const showHat = outMode === "yhat" && haveEval;

    const maxInLen = list.reduce((m, d) => Math.max(m, d.input.length), 1);
    const perTok = ctx.state.display === "squares" ? 16 : 14;
    const inColW = `${maxInLen * perTok}px`;

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

    list.forEach((d, i) => {
      const rowEl = document.createElement("div");
      rowEl.className = "example-row demo-row";
      if (i === selIdx) rowEl.classList.add("selected");
      if (!d.valid) rowEl.classList.add("invalid");
      rowEl.title = d.valid ? d.name : `${d.name} — ${d.hint}`;
      rowEl.addEventListener("click", () => ctx.apply({ demoIndex: d.index }));

      const ev = d.valid ? evalMap.get(d.index) : undefined;
      const idx = document.createElement("span");
      idx.className = "ex-index";
      if (ev) idx.classList.add(ev.wrong === 0 ? "all-right" : "has-wrong");
      idx.textContent = `#${d.index}`;
      const inSeq = renderSeq(d.input);
      inSeq.style.minWidth = inColW;
      rowEl.append(idx, inSeq);

      if (!d.valid) {
        const tag = document.createElement("span");
        tag.className = "demo-invalid-tag";
        tag.textContent = d.hint;
        rowEl.append(tag);
      } else if (classification) {
        const trueLabel = d.output[0] === 1;
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
        const outIds = showHat && ev ? ev.pred : d.output;
        const marks = ev ? ev.correct.map((c) => !c) : undefined;
        rowEl.append(arrow, renderSeq(outIds, marks, ev?.pTrue));
      }
      frag.appendChild(rowEl);
    });
    examplesEl.innerHTML = "";
    examplesEl.appendChild(frag);
  }

  // The example list can be large (up to 5000 rows), so only rebuild it when
  // something that affects it actually changes.
  let lastDataset = ctx.state.dataset;
  let lastListKey = "";
  let lastDemoOptKey = "";

  function update(): void {
    const s = ctx.state;
    displayRadios.set(s.display);
    vocabSlider.set(s.numSymbols);
    lenSlider.set(s.minSeqLen, s.maxSeqLen);
    fixedLenCheck.set(s.fixedLength);
    uniformLenCheck.set(s.uniformLen);
    demoCheck.set(s.demoExamples);
    demoUpdateCheck.set(s.demoUpdateWeights);
    trainSlider.set(s.trainPerEpoch);

    // Parens options: visible only for the parens task; maxes follow vocab/length.
    parensOptions.style.display = s.task === "parens" ? "" : "none";
    delimSlider.setMax(maxDelims(s.numSymbols));
    delimSlider.set(s.parensDelims);
    depthSlider.setMax(maxNestingDepth(s.maxSeqLen));
    depthSlider.set(s.parensMaxDepth);
    noMixedCheck.set(s.parensNoMixedNesting);

    // Grok controls + status.
    grokInput.set(s.grokFilters);
    const { errors } = compileFilters(s.grokFilters);
    const mi = s.dataset.matchInfo;
    if (errors.length > 0) {
      grokStatus.textContent = `regex error: ${errors[0]}`;
      grokStatus.classList.add("error");
    } else {
      grokStatus.classList.remove("error");
      if (!s.grokFilters.trim()) {
        grokStatus.textContent = "";
      } else if (mi?.mode === "enumerated") {
        grokStatus.textContent = `${mi.count} matching samples (enumerated)`;
      } else if (mi?.mode === "sampled") {
        grokStatus.textContent = `~${mi.count} matching (sampled — space too large to enumerate)`;
      } else {
        grokStatus.textContent = "";
      }
    }

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

    // --- demonstration mode: retitle + swap the Test-Set-Options body ---
    const demoMode = s.demoExamples;
    sizeTitle.textContent = demoMode
      ? `DEMONSTRATION – ${TASK_SPECS[s.task].label}`
      : "Test Set Options";
    normalControls.style.display = demoMode ? "none" : "";
    demoControls.style.display = demoMode ? "" : "none";
    grokRow.style.display = demoMode ? "none" : "";
    grokStatus.style.display = demoMode ? "none" : "";

    if (demoMode) {
      curPrepared = prepareDemos(s.task, {
        numSymbols: s.numSymbols,
        parensDelims: s.parensDelims,
        maxSeqLen: s.maxSeqLen,
      });
      curSelIdx = Math.min(Math.max(0, s.demoIndex), Math.max(0, curPrepared.length - 1));
      // Rebuild the <select> options when the task / validity set changes.
      const optKey = `${s.task}|${curPrepared.map((d) => (d.valid ? "1" : "0")).join("")}`;
      if (optKey !== lastDemoOptKey) {
        demoSelect.innerHTML = "";
        for (const d of curPrepared) {
          const o = document.createElement("option");
          o.value = String(d.index);
          o.textContent = d.valid ? d.name : `⚠ ${d.name}`;
          demoSelect.appendChild(o);
        }
        lastDemoOptKey = optKey;
      }
      const sel = curPrepared[curSelIdx];
      if (sel) {
        if (document.activeElement !== demoSelect) demoSelect.value = String(sel.index);
        demoName.textContent = sel.name;
        demoDesc.textContent = sel.description;
        demoHint.textContent = sel.valid ? "" : sel.hint;
        demoHint.style.display = sel.valid ? "none" : "";
        applySetupBtn.style.display = sel.setup ? "" : "none";
      }
    } else {
      curPrepared = [];
      curSelIdx = 0;
    }

    // Dataset rebuilt (new test set) → reset the view to live/index order.
    if (s.dataset !== lastDataset) {
      frozen = false;
      viewEval = null;
      viewEvalEpoch = -1;
    }
    // Auto mode tracks the latest eval; frozen mode keeps the pinned snapshot.
    if (!frozen) {
      viewEval = s.loop.lastTestEval;
      viewEvalEpoch = s.loop.lastTestEvalEpoch;
    }

    const haveEval = !!viewEval;
    const liveHaveEval = !!s.loop.lastTestEval;
    // input sort needs no eval; wrong/first/out do. Demos render in authored
    // order (no sorting), so only the y/ŷ toggle stays live in demo mode.
    inputBtn.disabled = demoMode;
    wrongBtn.disabled = demoMode || !haveEval;
    firstBtn.disabled = demoMode || !haveEval;
    outBtn.disabled = !haveEval;
    const arrow = (k: typeof sortKey) =>
      sortKey === k ? (sortDir === -1 ? " ▼" : " ▲") : "";
    inputBtn.textContent = `input${arrow("input")}`;
    inputBtn.classList.toggle("active", sortKey === "input");
    wrongBtn.textContent = `# wrong${arrow("wrong")}`;
    wrongBtn.classList.toggle("active", sortKey === "wrong");
    firstBtn.textContent = `1st error${arrow("first")}`;
    firstBtn.classList.toggle("active", sortKey === "first");
    outBtn.textContent = outMode === "yhat" ? "out: ŷ" : "out: y";

    // Title: TEST SET (N% correct) - eval @ epoch E  /  - no eval yet
    let titleStr = demoMode ? "Demonstration Set" : "Test Set";
    if (haveEval && viewEval) {
      const correct = viewEval.filter((e) => e.wrong === 0).length;
      const pctCorrect = Math.round((correct / viewEval.length) * 100);
      titleStr += ` (${pctCorrect}% correct) - eval @ epoch ${viewEvalEpoch}`;
    } else {
      titleStr += " - no eval yet";
    }
    titleText.textContent = titleStr;

    // Refresh button active only when a newer eval exists than what's shown.
    const stale = liveHaveEval && s.loop.lastTestEvalEpoch > viewEvalEpoch;
    refreshBtn.disabled = !stale;
    refreshBtn.classList.toggle("stale", stale);

    if (demoMode) {
      const demoKey = `demo|${s.display}|${s.task}|${s.numSymbols}|${s.parensDelims}|${s.maxSeqLen}|${curSelIdx}|${outMode}|${viewEvalEpoch}`;
      if (s.dataset !== lastDataset || demoKey !== lastListKey) {
        renderDemoList(curPrepared, curSelIdx);
        lastDataset = s.dataset;
        lastListKey = demoKey;
      }
    } else {
      const listKey = `${s.display}|${sortKey ?? ""}|${sortDir}|${outMode}|${viewEvalEpoch}`;
      if (s.dataset !== lastDataset || listKey !== lastListKey) {
        renderList(s.dataset.test);
        lastDataset = s.dataset;
        lastListKey = listKey;
      }
    }
  }

  update();
  return { update };
}
