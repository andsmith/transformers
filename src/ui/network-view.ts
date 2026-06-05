/**
 * Center panel: the network visualization.
 *
 * Renders the transformer as 8 pipeline stage-columns processed left to right
 * (titles above each), driven by the training loop's staged sample:
 *
 *   Input | Embeddings | Embed Sum | Q·K·V | Attention | Weighted Sum |
 *   Residual | Output
 *
 * Nothing is ever erased while stepping: weight matrices are always drawn
 * (values normally, ∇ heatmaps while the backward sweep covers them) and
 * activations of stages the forward pass hasn't reached yet are drawn as flat
 * light-gray placeholders. A bar of per-stage rectangles at the bottom marks
 * the active stage (green), with draggable "<>" handles between the bars to
 * resize column widths.
 *
 * Column widths are always laid out for the maximum sequence length, so the
 * layout (and any user resizing) stays stable across samples of different
 * lengths; the "constant-size" option additionally stretches activations
 * vertically/horizontally to fill that footprint.
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
const MIN_COL_W = 30; // resize floor
const HANDLE_HIT = 10; // px hit radius around a <> handle

const ACTIVE_GREEN = "#2fbf71";
const IDLE_GRAY = "#c9d2dc";

interface DrawMode {
  /** Show gradients instead of values (backward sweep covers this column). */
  grad: boolean;
  /** Stage not yet reached: activations as light-gray placeholders. */
  ghost: boolean;
}

interface ColumnSpec {
  /** Natural size: cells scale with the global cell size; fixed parts don't. */
  wCells: number;
  fixedW: number;
  hCells: number;
  fixedH: number;
  draw(
    g: CanvasRenderingContext2D,
    x: number,
    y: number,
    cellW: number,
    cellH: number,
    mode: DrawMode,
  ): void;
}

export function mountNetworkView(host: HTMLElement, ctx: AppContext): PanelHandle {
  host.classList.add("panel", "network-view");
  host.innerHTML = "";

  // --- DOM header (top-left) ---
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

  // --- column resizing state ---
  /** Per-column width adjustment in px (boundary drags transfer width). */
  const colDelta: number[] = PIPELINE_STAGES.map(() => 0);
  /** Last drawn layout, for handle hit-testing. */
  let lastLayout: { x: number; w: number }[] = [];
  let indicatorTop = 0;
  let resizeDrag: {
    boundary: number; // between columns [boundary] and [boundary+1]
    startX: number;
    dl: number;
    dr: number;
    wl: number;
    wr: number;
  } | null = null;

  function boundaryAt(px: number, py: number): number | null {
    if (lastLayout.length === 0) return null;
    if (py < indicatorTop - 8 || py > indicatorTop + INDICATOR_H + 8) return null;
    for (let i = 0; i < lastLayout.length - 1; i++) {
      const bx = lastLayout[i].x + lastLayout[i].w + COL_GAP / 2;
      if (Math.abs(px - bx) <= HANDLE_HIT) return i;
    }
    return null;
  }

  canvas.addEventListener("pointerdown", (e) => {
    const b = boundaryAt(e.offsetX, e.offsetY);
    if (b === null) return;
    resizeDrag = {
      boundary: b,
      startX: e.offsetX,
      dl: colDelta[b],
      dr: colDelta[b + 1],
      wl: lastLayout[b].w,
      wr: lastLayout[b + 1].w,
    };
    canvas.setPointerCapture(e.pointerId);
  });

  canvas.addEventListener("pointermove", (e) => {
    if (resizeDrag) {
      const d = resizeDrag;
      const dx = Math.max(
        -(d.wl - MIN_COL_W),
        Math.min(d.wr - MIN_COL_W, e.offsetX - d.startX),
      );
      colDelta[d.boundary] = d.dl + dx;
      colDelta[d.boundary + 1] = d.dr - dx;
      return;
    }
    canvas.style.cursor = boundaryAt(e.offsetX, e.offsetY) !== null ? "col-resize" : "default";
  });

  const endResize = (e: PointerEvent) => {
    if (!resizeDrag) return;
    resizeDrag = null;
    canvas.releasePointerCapture?.(e.pointerId);
  };
  canvas.addEventListener("pointerup", endResize);
  canvas.addEventListener("pointercancel", endResize);

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
    const twoLayer = model.wFF !== null;

    // Width layout is ALWAYS sized for the maximum sequence length, so the
    // columns (and the user's resize adjustments) stay put across samples.
    const NW = Math.max(n, s.maxSeqLen);
    // Heights stretch to the max footprint only with constant-size on.
    const NH = s.vizConstantSize ? NW : n;
    const rowH = (ch: number) => (ch * NH) / n;
    // Horizontal stretch for seq-wide matrices (attention): fill the NW-wide
    // slot when constant-size is on, else draw at natural width.
    const colW = (cw: number) => (s.vizConstantSize ? (cw * NW) / n : cw);

    const tokTable = model.embeddings.tokenTable;
    const posTable = model.embeddings.posTable;
    const usedTokenRows = [...new Set(st.sample.input)];
    const usedPosRows = Array.from({ length: n }, (_, i) => i);

    // 1. Input: token glyphs + one-hot matrix, rows aligned per position.
    const input: ColumnSpec = {
      wCells: 1 + V,
      fixedW: 8,
      hCells: NH,
      fixedH: MAT_FIXED_H,
      draw(gc, x, y, cw, ch, mode) {
        const rh = rowH(ch);
        const top = y + 11; // align with the one-hot's cell rows (badge offset)
        for (let i = 0; i < n; i++) {
          const id = st.sample.input[i];
          if (s.display === "squares") {
            gc.fillStyle = tokenColor(id, V);
            gc.fillRect(x, top + i * rh + 1, cw - 1, rh - 2);
          } else {
            gc.fillStyle = "#1f2a36";
            gc.font = `${Math.max(8, Math.min(cw, ch) - 3)}px ui-monospace, monospace`;
            gc.textAlign = "center";
            gc.textBaseline = "middle";
            gc.fillText(tokenChar(s.task, id, V), x + cw / 2, top + i * rh + rh / 2);
            gc.textAlign = "left";
            gc.textBaseline = "alphabetic";
          }
        }
        drawMatrix(gc, x + cw + 8, y, cw, rh, trace.oneHot, {
          cmap: aCmap,
          kind: "acts",
          ghost: mode.ghost,
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
      draw(gc, x, y, cw, ch, mode) {
        const a = drawMatrix(gc, x, y, cw, ch, tokTable, {
          cmap: wCmap,
          kind: "weights",
          grad: mode.grad,
          title: `token table (${V}×${d})`,
          outlineRows: usedTokenRows,
        });
        drawMatrix(gc, x, y + a.h + ROW_GAP, cw, ch, posTable, {
          cmap: wCmap,
          kind: "weights",
          grad: mode.grad,
          title: `position table (${L}×${d})`,
          outlineRows: usedPosRows,
        });
      },
    };

    // Helper for "m1 + m2 = m3" stacked activation columns.
    const stacked3 = (
      m1: () => Parameters<typeof drawMatrix>[5],
      t1: string,
      m2: () => Parameters<typeof drawMatrix>[5],
      t2: string,
      m3: () => Parameters<typeof drawMatrix>[5],
      t3: string,
    ): ColumnSpec => ({
      wCells: d,
      fixedW: 0,
      hCells: 3 * NH,
      fixedH: 3 * MAT_FIXED_H + 2 * GLYPH_ROW_H,
      draw(gc, x, y, cw, ch, mode) {
        const w = d * cw;
        const rh = rowH(ch);
        const opts = { cmap: aCmap, kind: "acts" as const, grad: mode.grad, ghost: mode.ghost };
        let yy = y;
        const s1 = drawMatrix(gc, x, yy, cw, rh, m1(), { ...opts, title: t1 });
        yy += s1.h;
        drawGlyph(gc, x + w / 2, yy + GLYPH_ROW_H / 2, "+");
        yy += GLYPH_ROW_H;
        const s2 = drawMatrix(gc, x, yy, cw, rh, m2(), { ...opts, title: t2 });
        yy += s2.h;
        drawGlyph(gc, x + w / 2, yy + GLYPH_ROW_H / 2, "=");
        yy += GLYPH_ROW_H;
        drawMatrix(gc, x, yy, cw, rh, m3(), { ...opts, title: t3 });
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
      hCells: 3 * Math.max(d, NH),
      fixedH: 3 * MAT_FIXED_H + 2 * ROW_GAP,
      draw(gc, x, y, cw, ch, mode) {
        let yy = y;
        for (const [name, w, act] of qkvRows) {
          const rowCells = Math.max(d, NH);
          const a = drawMatrix(gc, x, yy, cw, ch, w, {
            cmap: wCmap,
            kind: "weights",
            grad: mode.grad,
            title: `W_${name} (${d}×${d})`,
          });
          drawMatrix(gc, x + a.w + 8, yy, cw, rowH(ch), act, {
            cmap: aCmap,
            kind: "acts",
            grad: mode.grad,
            ghost: mode.ghost,
            title: name,
          });
          yy += rowCells * ch + MAT_FIXED_H + ROW_GAP;
        }
      },
    };

    // 5. Attention: scaled scores + row softmax (n×n, stretching to the NW
    // footprint when constant-size is on).
    const attnScores: ColumnSpec = {
      wCells: NW,
      fixedW: 0,
      hCells: 2 * NH,
      fixedH: 2 * MAT_FIXED_H + ROW_GAP,
      draw(gc, x, y, cw, ch, mode) {
        const sw = colW(cw);
        const sh = rowH(ch);
        const opts = { cmap: aCmap, kind: "acts" as const, grad: mode.grad, ghost: mode.ghost };
        const a = drawMatrix(gc, x, y, sw, sh, trace.attention.scores, {
          ...opts,
          title: "QKᵀ/√d",
        });
        drawMatrix(gc, x, y + a.h + ROW_GAP, sw, sh, trace.attention.attnW, {
          ...opts,
          title: "softmax rows",
        });
      },
    };

    // 6. Weighted sum (the lookup's output).
    const weighted: ColumnSpec = {
      wCells: d,
      fixedW: 0,
      hCells: NH,
      fixedH: MAT_FIXED_H,
      draw(gc, x, y, cw, ch, mode) {
        drawMatrix(gc, x, y, cw, rowH(ch), trace.attention.out, {
          cmap: aCmap,
          kind: "acts",
          grad: mode.grad,
          ghost: mode.ghost,
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

    // 8. Output: (optional FF hidden layer) + W_out + output activations.
    const outActRows = classification ? 1 : NH;
    const outActCols = classification ? 1 : V;
    const hiddenRows = classification ? 1 : NH;
    const output: ColumnSpec = {
      wCells: Math.max(d, outActCols),
      fixedW: 0,
      hCells: (twoLayer ? d + hiddenRows : 0) + outUnits + outActRows,
      fixedH: (twoLayer ? 4 : 2) * MAT_FIXED_H + (twoLayer ? 3 : 1) * ROW_GAP,
      draw(gc, x, y, cw, ch, mode) {
        const outCellH = classification ? ch : rowH(ch);
        let yy = y;

        if (twoLayer) {
          const f = drawMatrix(gc, x, yy, cw, ch, model.wFF!, {
            cmap: wCmap,
            kind: "weights",
            grad: mode.grad,
            title: `W_ff (${d}×${d})`,
          });
          yy += f.h + ROW_GAP;
          const hh = drawMatrix(
            gc,
            x,
            yy,
            cw,
            classification ? ch : rowH(ch),
            st.trace.hidden!,
            {
              cmap: aCmap,
              kind: "acts",
              grad: mode.grad,
              ghost: mode.ghost,
              title: "h = relu(W_ff·y)",
            },
          );
          yy += hh.h + ROW_GAP;
        }

        const a = drawMatrix(gc, x, yy, cw, ch, model.wOut, {
          cmap: wCmap,
          kind: "weights",
          grad: mode.grad,
          title: `W_out (${outUnits}×${d})`,
        });
        yy += a.h + ROW_GAP;

        if (mode.grad) {
          // Gradient of the raw logits — the signal backprop actually uses.
          drawMatrix(gc, x, yy, cw, outCellH, trace.logits, {
            cmap: aCmap,
            kind: "acts",
            grad: true,
            title: "∇logits",
          });
        } else if (classification) {
          const p = 1 / (1 + Math.exp(-trace.logits[0][0].data));
          drawMatrix(gc, x, yy, cw, outCellH, [[p]], {
            cmap: aCmap,
            kind: "acts",
            ghost: mode.ghost,
            title: `σ(logit) = ${p.toFixed(2)}`,
          });
        } else {
          const probs = trace.logits.map((row) => softmaxNums(row.map((v) => v.data)));
          drawMatrix(gc, x, yy, cw, outCellH, probs, {
            cmap: aCmap,
            kind: "acts",
            ghost: mode.ghost,
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
      lastLayout = [];
      return;
    }

    const cols = buildColumns(st);

    const gaps = (cols.length - 1) * COL_GAP;
    const fixedSum = cols.reduce((a, c) => a + c.fixedW, 0);
    const availContentW = Math.max(50, w - 2 * MARGIN - gaps - fixedSum);
    const totalWCells = cols.reduce((a, c) => a + c.wCells, 0);
    const contentTop = HEADER_H + TITLE_ROW_H;
    const contentH = h - contentTop - INDICATOR_H - 2 * MARGIN;

    // Columns spread across ALL the horizontal space (proportional to their
    // natural cell widths), plus the user's resize deltas.
    const allocW = cols.map((c, i) =>
      Math.max(
        MIN_COL_W,
        c.fixedW + (availContentW * c.wCells) / Math.max(1, totalWCells) + colDelta[i],
      ),
    );

    const phase = st.phase;
    const active = phase === "complete" ? -1 : st.stage;
    indicatorTop = h - MARGIN - INDICATOR_H;

    const layout: { x: number; w: number }[] = [];
    let x = MARGIN;
    for (let i = 0; i < cols.length; i++) {
      const c = cols[i];
      const colWpx = allocW[i];
      const isActive = i === active;
      layout.push({ x, w: colWpx });

      // Square cells: grow to fill the column's width, capped by the vertical
      // fit so badges/titles never push past the panel.
      const wFit = (colWpx - c.fixedW) / Math.max(1, c.wCells);
      const hFit = (contentH - c.fixedH) / Math.max(1, c.hCells);
      const cell = Math.max(2, Math.floor(Math.min(wFit, hFit)));
      // Center the drawn content within the allocated column width.
      const drawnW = c.wCells * cell + c.fixedW;
      const dx = Math.max(0, (colWpx - drawnW) / 2);

      // Stage title.
      g.fillStyle = isActive ? ACTIVE_GREEN : "#5a6675";
      g.font = `${isActive ? "bold " : ""}11px system-ui, sans-serif`;
      g.textAlign = "center";
      g.fillText(PIPELINE_STAGES[i].title, x + colWpx / 2, HEADER_H + 12, colWpx + COL_GAP - 4);
      g.textAlign = "left";

      // Column content — always drawn. Forward: unreached activations ghost.
      // Backward: columns the sweep covers (>= active) show gradients.
      const mode: DrawMode =
        phase === "forward"
          ? { grad: false, ghost: i > active }
          : phase === "backward"
            ? { grad: i >= active, ghost: false }
            : { grad: false, ghost: false };
      cols[i].draw(g, x + dx, contentTop + 4, cell, cell, mode);

      // Active-stage indicator rect (exactly as wide as the column above it).
      g.fillStyle = isActive ? ACTIVE_GREEN : IDLE_GRAY;
      g.fillRect(x, indicatorTop, colWpx, INDICATOR_H);

      x += colWpx + COL_GAP;
    }
    lastLayout = layout;

    // "<>" resize handles between the indicator bars.
    g.fillStyle = "#8a98a8";
    g.font = "bold 9px system-ui, sans-serif";
    g.textAlign = "center";
    g.textBaseline = "middle";
    for (let i = 0; i < layout.length - 1; i++) {
      const bx = layout[i].x + layout[i].w + COL_GAP / 2;
      g.fillText("<>", bx, indicatorTop + INDICATOR_H / 2);
    }
    g.textAlign = "left";
    g.textBaseline = "alphabetic";
  }

  function update(): void {
    draw();
  }

  update();
  return { update };
}
