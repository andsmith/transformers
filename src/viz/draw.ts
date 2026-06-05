/**
 * Canvas drawing helpers for the network visualization.
 *
 * The central primitive is {@link drawMatrix}: it renders a matrix of Values
 * (or plain numbers) as a heatmap with an unmistakable WEIGHTS vs ACTIVATIONS
 * treatment — distinct frame colors plus a corner badge (`W`/`A`, or `∇W`/`∇A`
 * when showing gradients).
 */

import { Value } from "../engine/value";
import {
  cmapColor,
  matrixRange,
  normDiv,
  normSeq,
  type Stops,
} from "./colormaps";

/** Frame/badge colors that distinguish the two kinds of heatmap. */
export const WEIGHTS_FRAME = "#5b3fa8"; // purple = trainable weights
export const ACTS_FRAME = "#1f7a4d"; // green = data flowing through

export interface MatrixOpts {
  cmap: Stops;
  kind: "weights" | "acts";
  /** Read .grad instead of .data (Value matrices only). */
  grad?: boolean;
  /** Draw as a flat light-gray placeholder (stage not yet reached) — the
   *  matrix keeps its footprint but shows no values. */
  ghost?: boolean;
  /** Small caption centered under the matrix. */
  title?: string;
  /** Row indices to outline (e.g. embedding-table rows used by the sample). */
  outlineRows?: number[];
}

export interface DrawnSize {
  w: number;
  h: number;
}

const BADGE_H = 11;
const TITLE_H = 11;

/** Extract plain numbers from a Value or number matrix. */
export function matrixData(
  m: Value[][] | number[][],
  grad: boolean,
): number[][] {
  return m.map((row) =>
    row.map((v) => (typeof v === "number" ? v : grad ? v.grad : v.data)),
  );
}

/**
 * Draw a heatmap at (x, y) with independent cell width/height (non-square
 * cells let activation matrices stretch into a constant footprint regardless
 * of sequence length). The badge row is drawn above the cells, the title (if
 * any) below. Returns the total footprint.
 */
export function drawMatrix(
  g: CanvasRenderingContext2D,
  x: number,
  y: number,
  cellW: number,
  cellH: number,
  m: Value[][] | number[][],
  opts: MatrixOpts,
): DrawnSize {
  const data = matrixData(m, !!opts.grad);
  const rows = data.length;
  const cols = rows > 0 ? data[0].length : 0;
  const w = cols * cellW;
  const h = rows * cellH;
  const top = y + BADGE_H;

  if (opts.ghost) {
    // Flat placeholder: shape without values.
    g.fillStyle = "#e6ebf1";
    g.fillRect(x, top, w, h);
  } else {
    // Normalization: sequential min..max for forward weights; symmetric about
    // 0 for activations and for any gradient view.
    const { min, max } = matrixRange(data);
    const diverging = opts.kind === "acts" || !!opts.grad;
    const absMax = Math.max(Math.abs(min), Math.abs(max));

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = data[r][c];
        const t = diverging ? normDiv(v, absMax) : normSeq(v, min, max);
        g.fillStyle = cmapColor(opts.cmap, t);
        g.fillRect(x + c * cellW, top + r * cellH, cellW, cellH);
      }
    }
  }

  // Frame + badge make the kind unmistakable (muted when ghosted).
  const frame = opts.ghost
    ? "#c4cdd6"
    : opts.kind === "weights"
      ? WEIGHTS_FRAME
      : ACTS_FRAME;
  g.strokeStyle = frame;
  g.lineWidth = 1.5;
  g.strokeRect(x - 0.75, top - 0.75, w + 1.5, h + 1.5);

  const badge = `${opts.grad ? "∇" : ""}${opts.kind === "weights" ? "W" : "A"}`;
  g.fillStyle = frame;
  g.font = "bold 9px system-ui, sans-serif";
  g.textAlign = "left";
  g.textBaseline = "alphabetic";
  g.fillText(badge, x, y + BADGE_H - 3);

  // Row outlines (e.g. which embedding rows this sample uses).
  if (!opts.ghost && opts.outlineRows && opts.outlineRows.length > 0) {
    g.strokeStyle = "#111";
    g.lineWidth = 1;
    for (const r of opts.outlineRows) {
      if (r >= 0 && r < rows) {
        g.strokeRect(x + 0.5, top + r * cellH + 0.5, w - 1, cellH - 1);
      }
    }
  }

  let totalH = BADGE_H + h;
  if (opts.title) {
    g.fillStyle = "#5a6675";
    g.font = "9px system-ui, sans-serif";
    g.textAlign = "center";
    g.fillText(opts.title, x + w / 2, top + h + TITLE_H - 2, Math.max(w, 60));
    g.textAlign = "left";
    totalH += TITLE_H;
  }

  return { w, h: totalH };
}

/** A centered operator glyph ("+", "=") between stacked operands. */
export function drawGlyph(
  g: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  ch: string,
): void {
  g.fillStyle = "#5a6675";
  g.font = "bold 12px system-ui, sans-serif";
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.fillText(ch, cx, cy);
  g.textAlign = "left";
  g.textBaseline = "alphabetic";
}
