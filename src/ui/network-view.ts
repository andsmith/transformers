/**
 * Center panel: the network visualization.
 *
 * Renders the transformer as 8 pipeline stage-columns processed left to right
 * (titles above each), driven by the training loop's staged sample:
 *
 *   Input | Embeddings | Embed Sum | Q·K·V | Attention | Weighted Sum |
 *   Residual | Output
 *
 * The forward pass reveals columns left→right; the backpropagation pass sweeps
 * right→left switching each column to gradient heatmaps (∇ badges). A bar of
 * per-stage rectangles at the bottom marks the active stage (green). Weight
 * matrices and activations are visually distinct (frame colors + W/A badges,
 * separate user-selectable colormaps).
 */

import type { AppContext } from "../state";
import type { PanelHandle } from "./top-panel";
import { PIPELINE_STAGES, type StagedSample } from "../training/loop";
import {
  ACT_CMAPS,
  WEIGHT_CMAPS,
  DEFAULT_ACT_CMAP,
  DEFAULT_WEIGHT_CMAP,
  type Stops,
} from "../viz/colormaps";
import { drawGlyph, drawMatrix } from "../viz/draw";
import { tokenChar, tokenColor } from "../tasks/grammar";
import { isClassification } from "../tasks/types";

const HEADER_H = 46; // space reserved for the DOM header
const TITLE_ROW_H = 18;
const INDICATOR_H = 13; // ~one text line
const MARGIN = 10;
const COL_GAP = 16;
const MAT_FIXED_H = 22; // badge row + title row per matrix (see viz/draw.ts)
const GLYPH_ROW_H = 13; // "+" / "=" rows between stacked matrices
const ROW_GAP = 9; // vertical gap between sub-rows (QKV, embeddings)

const ACTIVE_GREEN = "#2fbf71";
const IDLE_GRAY = "#c9d2dc";

interface ColumnSpec {
  /** Natural size: cells scale with the global cell size; fixed parts don't. */
  wCells: number;
  fixedW: number;
  hCells: number;
  fixedH: number;
  draw(g: CanvasRenderingContext2D, x: number, y: number, cell: number, grad: boolean): void;
}

export function mountNetworkView(host: HTMLElement, ctx: AppContext): PanelHandle {
  host.classList.add("panel", "network-view");
  host.innerHTML = "";

  // --- DOM header (top-left; the Run overlay occupies the top-right) ---
  const header = document.createElement("div");
  header.className = "nv-header";
  const titleEl = document.createElement("div");
  titleEl.className = "nv-title";
  const passEl = document.createElement("div");
  passEl.className = "nv-pass";
  header.append(titleEl, passEl);
  host.appendChild(header);

  const canvas = document.createElement("canvas");
  canvas.className = "network-canvas";
  host.appendChild(canvas);
  const g = canvas.getContext("2d")!;

  function softmaxNums(row: number[]): number[] {
    const m = Math.max(...row);
    const exps = row.map((v) => Math.exp(v - m));
    const s = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / s);
  }

  /** Build the 8 column renderers for the current staged sample. */
  function buildColumns(st: StagedSample): ColumnSpec[] {
    const s = ctx.state;
    const model = s.model;
    const trace = st.trace;
    const wCmap: Stops = WEIGHT_CMAPS[s.weightsCmap] ?? WEIGHT_CMAPS[DEFAULT_WEIGHT_CMAP];
    const aCmap: Stops = ACT_CMAPS[s.actsCmap] ?? ACT_CMAPS[DEFAULT_ACT_CMAP];

    const n = st.sample.input.length;
    const V = s.numSymbols;
    const d = s.embedDim;
    const L = model.embeddings.posTable.length;
    const classification = isClassification(s.task);
    const outUnits = model.outputUnits;

    const tokTable = model.embeddings.tokenTable;
    const posTable = model.embeddings.posTable;
    const usedTokenRows = [...new Set(st.sample.input)];
    const usedPosRows = Array.from({ length: n }, (_, i) => i);

    // 1. Input: token glyphs + one-hot matrix, rows aligned per position.
    const input: ColumnSpec = {
      wCells: 1 + V,
      fixedW: 8,
      hCells: n,
      fixedH: MAT_FIXED_H,
      draw(gc, x, y, cell, _grad) {
        // Tokens (one-hot has no gradient; this column always shows values).
        const top = y + 11; // align with the one-hot's cell rows (badge offset)
        for (let i = 0; i < n; i++) {
          const id = st.sample.input[i];
          if (s.display === "squares") {
            gc.fillStyle = tokenColor(id, V);
            gc.fillRect(x, top + i * cell + 1, cell - 1, cell - 2);
          } else {
            gc.fillStyle = "#1f2a36";
            gc.font = `${Math.max(8, cell - 3)}px ui-monospace, monospace`;
            gc.textAlign = "center";
            gc.textBaseline = "middle";
            gc.fillText(tokenChar(s.task, id, V), x + cell / 2, top + i * cell + cell / 2);
            gc.textAlign = "left";
            gc.textBaseline = "alphabetic";
          }
        }
        drawMatrix(gc, x + cell + 8, y, cell, trace.oneHot, {
          cmap: aCmap,
          kind: "acts",
          title: `one-hot (${n}×${V})`,
        });
      },
    };

    // 2. Embeddings: token + position weight tables, stacked.
    const embeds: ColumnSpec = {
      wCells: d,
      fixedW: 0,
      hCells: V + L,
      fixedH: 2 * MAT_FIXED_H + ROW_GAP,
      draw(gc, x, y, cell, grad) {
        const a = drawMatrix(gc, x, y, cell, tokTable, {
          cmap: wCmap,
          kind: "weights",
          grad,
          title: `token table (${V}×${d})`,
          outlineRows: usedTokenRows,
        });
        drawMatrix(gc, x, y + a.h + ROW_GAP, cell, posTable, {
          cmap: wCmap,
          kind: "weights",
          grad,
          title: `position table (${L}×${d})`,
          outlineRows: usedPosRows,
        });
      },
    };

    // Helper for "m1 + m2 = m3" stacked activation columns.
    const stacked3 = (
      m1: () => Parameters<typeof drawMatrix>[4],
      t1: string,
      m2: () => Parameters<typeof drawMatrix>[4],
      t2: string,
      m3: () => Parameters<typeof drawMatrix>[4],
      t3: string,
    ): ColumnSpec => ({
      wCells: d,
      fixedW: 0,
      hCells: 3 * n,
      fixedH: 3 * MAT_FIXED_H + 2 * GLYPH_ROW_H,
      draw(gc, x, y, cell, grad) {
        const w = d * cell;
        let yy = y;
        const s1 = drawMatrix(gc, x, yy, cell, m1(), { cmap: aCmap, kind: "acts", grad, title: t1 });
        yy += s1.h;
        drawGlyph(gc, x + w / 2, yy + GLYPH_ROW_H / 2, "+");
        yy += GLYPH_ROW_H;
        const s2 = drawMatrix(gc, x, yy, cell, m2(), { cmap: aCmap, kind: "acts", grad, title: t2 });
        yy += s2.h;
        drawGlyph(gc, x + w / 2, yy + GLYPH_ROW_H / 2, "=");
        yy += GLYPH_ROW_H;
        drawMatrix(gc, x, yy, cell, m3(), { cmap: aCmap, kind: "acts", grad, title: t3 });
      },
    });

    // 3. Embed Sum: tok + pos = x.
    const embedSum = stacked3(
      () => trace.tok,
      "tok",
      () => trace.pos,
      "pos",
      () => trace.x,
      "x = tok + pos",
    );

    // 4. Q K V: three rows of W (weights) -> projection (activations).
    const attn = model.attention;
    const qkvRows: Array<[string, typeof attn.wq, typeof trace.attention.q]> = [
      ["Q", attn.wq, trace.attention.q],
      ["K", attn.wk, trace.attention.k],
      ["V", attn.wv, trace.attention.v],
    ];
    const qkv: ColumnSpec = {
      wCells: 2 * d,
      fixedW: 8,
      hCells: 3 * Math.max(d, n),
      fixedH: 3 * MAT_FIXED_H + 2 * ROW_GAP,
      draw(gc, x, y, cell, grad) {
        let yy = y;
        for (const [name, w, act] of qkvRows) {
          const rowCells = Math.max(d, n);
          const a = drawMatrix(gc, x, yy, cell, w, {
            cmap: wCmap,
            kind: "weights",
            grad,
            title: `W_${name} (${d}×${d})`,
          });
          drawMatrix(gc, x + a.w + 8, yy, cell, act, {
            cmap: aCmap,
            kind: "acts",
            grad,
            title: name,
          });
          yy += rowCells * cell + MAT_FIXED_H + ROW_GAP;
        }
      },
    };

    // 5. Attention: scaled scores + row softmax.
    const attnScores: ColumnSpec = {
      wCells: n,
      fixedW: 0,
      hCells: 2 * n,
      fixedH: 2 * MAT_FIXED_H + ROW_GAP,
      draw(gc, x, y, cell, grad) {
        const a = drawMatrix(gc, x, y, cell, trace.attention.scores, {
          cmap: aCmap,
          kind: "acts",
          grad,
          title: "QKᵀ/√d",
        });
        drawMatrix(gc, x, y + a.h + ROW_GAP, cell, trace.attention.attnW, {
          cmap: aCmap,
          kind: "acts",
          grad,
          title: "softmax rows",
        });
      },
    };

    // 6. Weighted sum (the lookup's output).
    const weighted: ColumnSpec = {
      wCells: d,
      fixedW: 0,
      hCells: n,
      fixedH: MAT_FIXED_H,
      draw(gc, x, y, cell, grad) {
        drawMatrix(gc, x, y, cell, trace.attention.out, {
          cmap: aCmap,
          kind: "acts",
          grad,
          title: "A·V",
        });
      },
    };

    // 7. Residual: x + attnOut = y.
    const residual = stacked3(
      () => trace.x,
      "x",
      () => trace.attention.out,
      "attn out",
      () => trace.y,
      "y = x + attn",
    );

    // 8. Output: W_out weights + output-unit activations.
    const outActRows = classification ? 1 : n;
    const outActCols = classification ? 1 : V;
    const output: ColumnSpec = {
      wCells: Math.max(d, outActCols),
      fixedW: 0,
      hCells: outUnits + outActRows,
      fixedH: 2 * MAT_FIXED_H + ROW_GAP,
      draw(gc, x, y, cell, grad) {
        const a = drawMatrix(gc, x, y, cell, model.wOut, {
          cmap: wCmap,
          kind: "weights",
          grad,
          title: `W_out (${outUnits}×${d})`,
        });
        const yy = y + a.h + ROW_GAP;
        if (grad) {
          // Gradient of the raw logits — the signal backprop actually uses.
          drawMatrix(gc, x, yy, cell, trace.logits, {
            cmap: aCmap,
            kind: "acts",
            grad: true,
            title: "∇logits",
          });
        } else if (classification) {
          const p = 1 / (1 + Math.exp(-trace.logits[0][0].data));
          drawMatrix(gc, x, yy, cell, [[p]], {
            cmap: aCmap,
            kind: "acts",
            title: `σ(logit) = ${p.toFixed(2)}`,
          });
        } else {
          const probs = trace.logits.map((row) => softmaxNums(row.map((v) => v.data)));
          drawMatrix(gc, x, yy, cell, probs, {
            cmap: aCmap,
            kind: "acts",
            title: "softmax(logits)",
          });
        }
      },
    };

    return [input, embeds, embedSum, qkv, attnScores, weighted, residual, output];
  }

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

    const loop = ctx.state.loop;
    const st = loop.staged;

    // Header text.
    titleEl.textContent = st
      ? `Network Visualization - Epoch ${loop.epoch} - Sample #${st.sample.index}`
      : "Network Visualization";
    passEl.className = "nv-pass " + (st ? st.phase : "idle");
    if (!st) {
      passEl.textContent = "Press Step to begin";
    } else if (st.phase === "forward") {
      passEl.textContent = "Forward Pass";
    } else if (st.phase === "backward") {
      passEl.textContent = "Backpropagation Pass";
    } else {
      const loss = st.lossValue.data.toFixed(3);
      passEl.textContent = ctx.state.running
        ? `Training continuously · loss ${loss}`
        : `Iteration complete · loss ${loss}`;
    }

    if (!st) {
      g.fillStyle = "#9aa7b4";
      g.font = "13px system-ui, sans-serif";
      g.textAlign = "center";
      g.fillText(
        "Step advances one pipeline stage: forward left→right, then backprop right←left.",
        w / 2,
        h / 2,
      );
      g.textAlign = "left";
      return;
    }

    const cols = buildColumns(st);

    // Global cell size: fit total width and every column's height.
    const availW = w - 2 * MARGIN - (cols.length - 1) * COL_GAP - cols.reduce((a, c) => a + c.fixedW, 0);
    const totalWCells = cols.reduce((a, c) => a + c.wCells, 0);
    const contentTop = HEADER_H + TITLE_ROW_H;
    const contentH = h - contentTop - INDICATOR_H - 2 * MARGIN;
    let cell = availW / Math.max(1, totalWCells);
    for (const c of cols) {
      cell = Math.min(cell, (contentH - c.fixedH) / Math.max(1, c.hCells));
    }
    cell = Math.max(2, Math.min(14, Math.floor(cell)));

    const phase = st.phase;
    // "complete" shows the whole pipeline with no active stage.
    const active = phase === "complete" ? -1 : st.stage;

    let x = MARGIN;
    for (let i = 0; i < cols.length; i++) {
      const c = cols[i];
      const colW = c.wCells * cell + c.fixedW;
      const isActive = i === active;

      // Stage title.
      g.fillStyle = isActive ? ACTIVE_GREEN : "#5a6675";
      g.font = `${isActive ? "bold " : ""}11px system-ui, sans-serif`;
      g.textAlign = "center";
      g.fillText(PIPELINE_STAGES[i].title, x + colW / 2, HEADER_H + 12, colW + COL_GAP - 4);
      g.textAlign = "left";

      // Column content. Forward: reveal columns up to the active one.
      // Backward: all visible; columns the sweep has reached (>= active) show
      // gradients. Complete: everything visible as values.
      const visible = phase !== "forward" || i <= active;
      if (visible) {
        const grad = phase === "backward" && i >= active;
        c.draw(g, x, contentTop + 4, cell, grad);
      }

      // Active-stage indicator rect (exactly as wide as the column above it).
      g.fillStyle = isActive ? ACTIVE_GREEN : IDLE_GRAY;
      g.fillRect(x, h - MARGIN - INDICATOR_H, colW, INDICATOR_H);

      x += colW + COL_GAP;
    }
  }

  function update(): void {
    draw();
  }

  update();
  return { update };
}
