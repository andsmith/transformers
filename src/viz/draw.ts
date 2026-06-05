/**
 * Canvas drawing helpers for the network visualization.
 *
 * The central primitive is {@link drawMatrix}. Caption layout around the
 * heatmap:
 *   top-left  — kind label "Wts." / "Act." (∇-prefixed for gradient views)
 *   top-right — the matrix size "rows×cols"
 *   bottom    — description / equation (caller-provided title)
 *   left      — optional formal math name (e.g. W with subscript Q), large+dark
 * All caption text scales with the `font` option (the viz-captions font size).
 */

import { Value } from "../engine/value";
import {
  cmapColor,
  matrixRange,
  normDiv,
  normSeq,
  type Stops,
} from "./colormaps";

/** Frame/label colors that distinguish the two kinds of heatmap. */
export const WEIGHTS_FRAME = "#5b3fa8"; // purple = trainable weights
export const ACTS_FRAME = "#1f7a4d"; // green = data flowing through
const GHOST_FRAME = "#c4cdd6";
const INK = "#1f2a36";
const MUTED = "#5a6675";

export interface MathLabel {
  main: string;
  sub?: string;
}

export interface MatrixOpts {
  cmap: Stops;
  kind: "weights" | "acts";
  /** Read .grad instead of .data (Value matrices only). */
  grad?: boolean;
  /** Draw as a flat light-gray placeholder (stage not yet reached). */
  ghost?: boolean;
  /** Bottom caption: description or equation. */
  title?: string;
  /** Formal name drawn left of the matrix in large dark text. */
  mathLabel?: MathLabel;
  /** Caption font size in px (viz font setting). */
  font?: number;
  /** Row indices to outline (e.g. embedding-table rows used by the sample). */
  outlineRows?: number[];
}

export interface DrawnSize {
  w: number;
  h: number;
}

export const DEFAULT_VIZ_FONT = 9;

/** Height of the caption strip above the cells. */
export function capH(font: number): number {
  return font + 4;
}

/** Height of the description strip below the cells. */
export function titleH(font: number): number {
  return font + 4;
}

/** Fixed vertical overhead of one matrix (caption + description strips). */
export function matFixedH(font: number): number {
  return capH(font) + titleH(font);
}

/** Horizontal gutter reserved for a math label. */
export function mathLabelW(font: number): number {
  return Math.round(font * 2.4) + 6;
}

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
 * Draw a heatmap at (x, y) with independent cell width/height. Returns the
 * total footprint (including math-label gutter and caption strips).
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
  const font = opts.font ?? DEFAULT_VIZ_FONT;
  const data = matrixData(m, !!opts.grad);
  const rows = data.length;
  const cols = rows > 0 ? data[0].length : 0;
  const lw = opts.mathLabel ? mathLabelW(font) : 0;
  const w = cols * cellW;
  const h = rows * cellH;
  const gx = x + lw;
  const gy = y + capH(font);

  // Formal name in the left gutter, vertically centered on the cells.
  if (opts.mathLabel) {
    g.fillStyle = INK;
    g.textBaseline = "middle";
    g.textAlign = "left";
    const mainFont = font + 5;
    g.font = `italic bold ${mainFont}px Georgia, "Times New Roman", serif`;
    const mainW = g.measureText(opts.mathLabel.main).width;
    const my = gy + h / 2;
    g.fillText(opts.mathLabel.main, x, my);
    if (opts.mathLabel.sub) {
      g.font = `italic ${font}px Georgia, "Times New Roman", serif`;
      g.fillText(opts.mathLabel.sub, x + mainW + 1, my + mainFont * 0.35);
    }
    g.textBaseline = "alphabetic";
  }

  if (opts.ghost) {
    // Flat placeholder: shape without values.
    g.fillStyle = "#e6ebf1";
    g.fillRect(gx, gy, w, h);
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
        g.fillRect(gx + c * cellW, gy + r * cellH, cellW, cellH);
      }
    }
  }

  // Frame makes the kind unmistakable (muted when ghosted).
  const frame = opts.ghost
    ? GHOST_FRAME
    : opts.kind === "weights"
      ? WEIGHTS_FRAME
      : ACTS_FRAME;
  g.strokeStyle = frame;
  g.lineWidth = 1.5;
  g.strokeRect(gx - 0.75, gy - 0.75, w + 1.5, h + 1.5);

  // Top-left: kind label. Top-right: size.
  const kindText = `${opts.grad ? "∇ " : ""}${opts.kind === "weights" ? "Wts." : "Act."}`;
  g.fillStyle = frame;
  g.font = `bold ${font}px system-ui, sans-serif`;
  g.textAlign = "left";
  g.fillText(kindText, gx, y + capH(font) - 3);
  g.fillStyle = MUTED;
  g.font = `${Math.max(7, font - 1)}px system-ui, sans-serif`;
  g.textAlign = "right";
  g.fillText(`${rows}×${cols}`, gx + w, y + capH(font) - 3);
  g.textAlign = "left";

  // Row outlines (e.g. which embedding rows this sample uses).
  if (!opts.ghost && opts.outlineRows && opts.outlineRows.length > 0) {
    g.strokeStyle = "#111";
    g.lineWidth = 1;
    for (const r of opts.outlineRows) {
      if (r >= 0 && r < rows) {
        g.strokeRect(gx + 0.5, gy + r * cellH + 0.5, w - 1, cellH - 1);
      }
    }
  }

  let totalH = capH(font) + h;
  if (opts.title) {
    g.fillStyle = opts.ghost ? GHOST_FRAME : MUTED;
    g.font = `${font}px system-ui, sans-serif`;
    g.textAlign = "center";
    g.fillText(opts.title, gx + w / 2, gy + h + titleH(font) - 3, Math.max(w, 70));
    g.textAlign = "left";
    totalH += titleH(font);
  }

  return { w: lw + w, h: totalH };
}

/** A centered operator glyph ("+", "=") between stacked operands. */
export function drawGlyph(
  g: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  ch: string,
): void {
  g.fillStyle = MUTED;
  g.font = "bold 12px system-ui, sans-serif";
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.fillText(ch, cx, cy);
  g.textAlign = "left";
  g.textBaseline = "alphabetic";
}
