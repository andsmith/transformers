/**
 * Colormaps for the network visualization.
 *
 * Sequential maps (viridis family) color WEIGHT matrices via linear min..max
 * normalization. Diverging maps (bwr family) color ACTIVATIONS via symmetric
 * normalization about zero, so 0 always lands on the white/neutral midpoint.
 * Gradients use symmetric normalization with either family.
 */

export type RGB = [number, number, number];
/** Evenly spaced color stops over t in [0, 1]. */
export type Stops = RGB[];

// --- sequential (weights) ---

const VIRIDIS: Stops = [
  [68, 1, 84],
  [71, 44, 122],
  [59, 81, 139],
  [44, 113, 142],
  [33, 144, 141],
  [39, 173, 129],
  [92, 200, 99],
  [170, 220, 50],
  [253, 231, 37],
];

const PLASMA: Stops = [
  [13, 8, 135],
  [75, 3, 161],
  [125, 3, 168],
  [168, 34, 150],
  [203, 70, 121],
  [229, 107, 93],
  [248, 148, 65],
  [253, 195, 40],
  [240, 249, 33],
];

const INFERNO: Stops = [
  [0, 0, 4],
  [40, 11, 84],
  [101, 21, 110],
  [159, 42, 99],
  [212, 72, 66],
  [245, 125, 21],
  [250, 193, 39],
  [252, 255, 164],
];

const MAGMA: Stops = [
  [0, 0, 4],
  [28, 16, 68],
  [79, 18, 123],
  [129, 37, 129],
  [181, 54, 122],
  [229, 80, 100],
  [251, 135, 97],
  [254, 194, 135],
  [252, 253, 191],
];

const GRAY: Stops = [
  [0, 0, 0],
  [255, 255, 255],
];

// --- diverging (activations / gradients) ---

const BWR: Stops = [
  [0, 0, 255],
  [255, 255, 255],
  [255, 0, 0],
];

const COOLWARM: Stops = [
  [59, 76, 192],
  [144, 178, 254],
  [221, 221, 221],
  [246, 153, 122],
  [180, 4, 38],
];

const SEISMIC: Stops = [
  [0, 0, 76],
  [0, 0, 255],
  [255, 255, 255],
  [255, 0, 0],
  [128, 0, 0],
];

export const WEIGHT_CMAPS: Record<string, Stops> = {
  viridis: VIRIDIS,
  plasma: PLASMA,
  inferno: INFERNO,
  magma: MAGMA,
  gray: GRAY,
};

export const ACT_CMAPS: Record<string, Stops> = {
  bwr: BWR,
  coolwarm: COOLWARM,
  seismic: SEISMIC,
};

export const DEFAULT_WEIGHT_CMAP = "viridis";
export const DEFAULT_ACT_CMAP = "bwr";

/** Interpolate a colormap at t in [0,1] and return a CSS color. */
export function cmapColor(stops: Stops, t: number): string {
  const tt = Math.max(0, Math.min(1, t));
  const pos = tt * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(pos));
  const f = pos - i;
  const a = stops[i];
  const b = stops[i + 1];
  const r = Math.round(a[0] + (b[0] - a[0]) * f);
  const g = Math.round(a[1] + (b[1] - a[1]) * f);
  const bl = Math.round(a[2] + (b[2] - a[2]) * f);
  return `rgb(${r},${g},${bl})`;
}

/** Min/max over a numeric matrix. */
export function matrixRange(m: number[][]): { min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;
  for (const row of m) {
    for (const v of row) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  if (!isFinite(min)) {
    min = 0;
    max = 1;
  }
  return { min, max };
}

/** Linear min..max normalization (sequential maps). */
export function normSeq(v: number, min: number, max: number): number {
  const span = max - min;
  return span > 1e-12 ? (v - min) / span : 0.5;
}

/** Symmetric-about-zero normalization (diverging maps): 0 maps to 0.5. */
export function normDiv(v: number, absMax: number): number {
  return absMax > 1e-12 ? 0.5 + v / (2 * absMax) : 0.5;
}
