/**
 * Typed-array transformer (the app's training/visualization engine).
 *
 * Same architecture as the scalar `Value` model in ./transformer.ts —
 * token+positional embedding → single attention head → residual → optional FF
 * hidden layer → output projection — but the forward and backward are written
 * by hand over plain `number[][]` matrices (no per-scalar autograd graph). This
 * is ~10–100× faster for these tiny models; the scalar model is retained as a
 * gradient-parity oracle in the smoke test.
 *
 * `forward` records every intermediate's values; `backward` records every
 * intermediate's gradients (and accumulates weight gradients) — so the network
 * visualization draws exactly what it did before, reading `.v` for the forward
 * pass and `.g` for the backprop sweep.
 */

import { mulberry32 } from "../tasks/datasets";
import { isClassification, type Task } from "../tasks/types";
import type { PEScheme } from "./embeddings";

export interface ModelConfig {
  task: Task;
  vocabSize: number;
  embedDim: number;
  /** Fixed identity token embedding (pedagogical; d_tok = vocabSize). */
  tokenOneHot: boolean;
  peScheme: PEScheme;
  numOutputLayers: 1 | 2;
  maxLen: number;
}

/** Token-content dims (one-hot forces |V|). */
export function tokenDims(cfg: { tokenOneHot: boolean; vocabSize: number; embedDim: number }): number {
  return cfg.tokenOneHot ? cfg.vocabSize : cfg.embedDim;
}

/** Full model width: token dims + a dedicated block for one-hot positions. */
export function modelDims(cfg: {
  tokenOneHot: boolean;
  vocabSize: number;
  embedDim: number;
  peScheme: PEScheme;
  maxLen: number;
}): number {
  return tokenDims(cfg) + (cfg.peScheme === "onehot" ? cfg.maxLen : 0);
}

/** A trainable weight matrix: values `w` and accumulated gradients `g`. */
export interface Param {
  name: string;
  rows: number;
  cols: number;
  w: number[][];
  g: number[][];
}

/** An activation matrix: forward values `v` and (post-backward) gradients `g`. */
export interface Mat {
  v: number[][];
  g: number[][];
}

export interface FastTrace {
  oneHot: number[][];
  tok: Mat;
  pos: Mat;
  x: Mat;
  q: Mat;
  k: Mat;
  v: Mat;
  scores: Mat;
  attnW: Mat;
  out: Mat;
  y: Mat;
  pooled: Mat | null;
  hidden: Mat | null;
  logits: Mat;
}

// --- small matrix helpers (row-major number[][]) ---
const zeros = (r: number, c: number): number[][] =>
  Array.from({ length: r }, () => new Array<number>(c).fill(0));

function matVec(m: number[][], x: number[]): number[] {
  const out = new Array<number>(m.length);
  for (let r = 0; r < m.length; r++) {
    let acc = 0;
    const row = m[r];
    for (let c = 0; c < x.length; c++) acc += row[c] * x[c];
    out[r] = acc;
  }
  return out;
}

function sinusoidalTable(maxLen: number, dim: number): number[][] {
  const t = zeros(maxLen, dim);
  for (let pos = 0; pos < maxLen; pos++) {
    for (let i = 0; i < dim; i++) {
      const k = Math.floor(i / 2);
      const angle = pos / Math.pow(10000, (2 * k) / dim);
      t[pos][i] = i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
    }
  }
  return t;
}

export class FastModel {
  readonly cfg: ModelConfig;
  readonly outputUnits: number;
  /** Full embedding width (token dims + one-hot position block when used). */
  readonly dim: number;
  /** Trainable params (excludes any fixed table: sinusoidal/one-hot/zero). */
  readonly params: Param[] = [];

  /** Token table — a Param when learned, else a fixed identity block with a
   *  grad buffer kept only for the visualization. */
  readonly tokTable: Param;
  /** Positional table — a Param when learned, else fixed (sinusoidal /
   *  one-hot identity block / zeros) with a display-only grad buffer. */
  readonly posTable: Param;

  readonly wq: Param;
  readonly wk: Param;
  readonly wv: Param;
  readonly wFF: Param | null;
  readonly wOut: Param;

  constructor(cfg: ModelConfig, rng: () => number) {
    this.cfg = cfg;
    this.outputUnits = isClassification(cfg.task) ? 1 : cfg.vocabSize;
    const dTok = tokenDims(cfg);
    const d = modelDims(cfg);
    this.dim = d;

    // He-uniform initializer (matches scalar model: (rng*2-1)/sqrt(fanIn)).
    const he = (fanIn: number) => {
      const s = 1 / Math.sqrt(Math.max(1, fanIn));
      return () => (rng() * 2 - 1) * s;
    };
    // Draw order MUST match ParamStore in the scalar model so the same seed
    // produces the same initial weights.
    const make = (name: string, rows: number, cols: number, init: () => number, trainable = true): Param => {
      const w = zeros(rows, cols);
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) w[r][c] = init();
      const p: Param = { name, rows, cols, w, g: zeros(rows, cols) };
      if (trainable) this.params.push(p);
      return p;
    };
    const fixed = (name: string, rows: number, cols: number, at: (r: number, c: number) => number): Param => {
      const w = zeros(rows, cols);
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) w[r][c] = at(r, c);
      return { name, rows, cols, w, g: zeros(rows, cols) };
    };

    this.tokTable = cfg.tokenOneHot
      ? fixed("tok_emb", cfg.vocabSize, d, (r, c) => (c === r ? 1 : 0))
      : make("tok_emb", cfg.vocabSize, d, he(d));

    switch (cfg.peScheme) {
      case "learned":
        this.posTable = make("pos_emb", cfg.maxLen, d, he(d));
        break;
      case "sinusoidal":
        this.posTable = {
          name: "pos_emb",
          rows: cfg.maxLen,
          cols: d,
          w: sinusoidalTable(cfg.maxLen, d),
          g: zeros(cfg.maxLen, d),
        };
        break;
      case "onehot":
        // Identity in a dedicated trailing block: position i flags coordinate
        // dTok + i — "what" and "where" live on separate wires.
        this.posTable = fixed("pos_emb", cfg.maxLen, d, (r, c) => (c === dTok + r ? 1 : 0));
        break;
      case "none":
        this.posTable = fixed("pos_emb", cfg.maxLen, d, () => 0);
        break;
    }

    this.wq = make("attn_Wq", d, d, he(d));
    this.wk = make("attn_Wk", d, d, he(d));
    this.wv = make("attn_Wv", d, d, he(d));
    this.wFF = cfg.numOutputLayers === 2 ? make("ff_W1", d, d, he(d)) : null;
    this.wOut = make("out_proj", this.outputUnits, d, he(d));
  }

  zeroGrad(): void {
    for (const p of this.params) for (const row of p.g) row.fill(0);
    // Fixed tables aren't in params but their ∇ is shown — keep it per-sample.
    for (const row of this.posTable.g) row.fill(0);
    for (const row of this.tokTable.g) row.fill(0);
  }

  step(lr: number): void {
    for (const p of this.params) {
      for (let r = 0; r < p.rows; r++) {
        const w = p.w[r];
        const g = p.g[r];
        for (let c = 0; c < p.cols; c++) w[c] -= lr * g[c];
      }
    }
  }

  /** Forward pass; records every intermediate's values into the trace. */
  forward(ids: number[]): FastTrace {
    const n = ids.length;
    const d = this.dim;
    const V = this.cfg.vocabSize;
    const invSqrtD = 1 / Math.sqrt(d);
    const classify = isClassification(this.cfg.task);

    const oneHot = ids.map((id) => {
      const row = new Array<number>(V).fill(0);
      row[id] = 1;
      return row;
    });

    const tok = ids.map((id) => this.tokTable.w[id].slice());
    const pos = ids.map((_, i) => this.posTable.w[i].slice());
    const x = tok.map((ti, i) => ti.map((val, c) => val + pos[i][c]));

    const q = x.map((xi) => matVec(this.wq.w, xi));
    const k = x.map((xi) => matVec(this.wk.w, xi));
    const v = x.map((xi) => matVec(this.wv.w, xi));

    const scores = zeros(n, n);
    const attnW = zeros(n, n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let dotqk = 0;
        for (let c = 0; c < d; c++) dotqk += q[i][c] * k[j][c];
        scores[i][j] = dotqk * invSqrtD;
      }
      // softmax (numerically stable) over row i
      let mx = -Infinity;
      for (let j = 0; j < n; j++) mx = Math.max(mx, scores[i][j]);
      let sum = 0;
      for (let j = 0; j < n; j++) {
        const e = Math.exp(scores[i][j] - mx);
        attnW[i][j] = e;
        sum += e;
      }
      for (let j = 0; j < n; j++) attnW[i][j] /= sum;
    }

    const out = zeros(n, d);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        const a = attnW[i][j];
        for (let c = 0; c < d; c++) out[i][c] += a * v[j][c];
      }

    const y = x.map((xi, i) => xi.map((val, c) => val + out[i][c]));

    const relu = (vec: number[]) => vec.map((z) => (z > 0 ? z : 0));

    let pooled: Mat | null = null;
    let hidden: Mat | null = null;
    let logits: number[][];
    if (classify) {
      const p = new Array<number>(d).fill(0);
      for (let i = 0; i < n; i++) for (let c = 0; c < d; c++) p[c] += y[i][c] / n;
      pooled = { v: [p], g: [new Array<number>(d).fill(0)] };
      let pre = p;
      if (this.wFF) {
        pre = relu(matVec(this.wFF.w, p));
        hidden = { v: [pre], g: [new Array<number>(d).fill(0)] };
      }
      logits = [matVec(this.wOut.w, pre)];
    } else {
      let pre = y;
      if (this.wFF) {
        const h = y.map((yi) => relu(matVec(this.wFF!.w, yi)));
        hidden = { v: h, g: zeros(n, d) };
        pre = h;
      }
      logits = pre.map((row) => matVec(this.wOut.w, row));
    }

    return {
      oneHot,
      tok: { v: tok, g: zeros(n, d) },
      pos: { v: pos, g: zeros(n, d) },
      x: { v: x, g: zeros(n, d) },
      q: { v: q, g: zeros(n, d) },
      k: { v: k, g: zeros(n, d) },
      v: { v, g: zeros(n, d) },
      scores: { v: scores, g: zeros(n, n) },
      attnW: { v: attnW, g: zeros(n, n) },
      out: { v: out, g: zeros(n, d) },
      y: { v: y, g: zeros(n, d) },
      pooled,
      hidden,
      logits: { v: logits, g: zeros(logits.length, this.outputUnits) },
    };
  }

  /**
   * Loss for the trace's logits; also seeds `logits.g` with ∂loss/∂logits.
   * Transduction: mean per-position softmax cross-entropy. Classification:
   * sigmoid binary cross-entropy on the single pooled logit.
   */
  computeLoss(trace: FastTrace, ex: { output: number[] }): number {
    if (isClassification(this.cfg.task)) {
      const logit = trace.logits.v[0][0];
      const p = 1 / (1 + Math.exp(-logit));
      const y = ex.output[0];
      trace.logits.g[0][0] = p - y; // ∂BCE/∂logit
      const pt = y === 1 ? p : 1 - p;
      return -Math.log(Math.max(pt, 1e-12));
    }
    const nPos = trace.logits.v.length;
    let loss = 0;
    for (let i = 0; i < nPos; i++) {
      const row = trace.logits.v[i];
      let mx = -Infinity;
      for (const z of row) mx = Math.max(mx, z);
      let sum = 0;
      const probs = row.map((z) => {
        const e = Math.exp(z - mx);
        sum += e;
        return e;
      });
      for (let c = 0; c < probs.length; c++) probs[c] /= sum;
      const t = ex.output[i];
      loss += -Math.log(Math.max(probs[t], 1e-12)) / nPos;
      for (let c = 0; c < probs.length; c++) {
        trace.logits.g[i][c] = (probs[c] - (c === t ? 1 : 0)) / nPos;
      }
    }
    return loss;
  }

  /** Backward pass: fills every intermediate `.g` and accumulates weight `.g`. */
  backward(trace: FastTrace): void {
    const n = trace.x.v.length;
    const d = this.dim;
    const classify = isClassification(this.cfg.task);

    const dLogits = trace.logits.g;
    // pre = input to the output projection (per output row).
    const preRows = classify
      ? this.wFF
        ? trace.hidden!.v
        : trace.pooled!.v
      : this.wFF
        ? trace.hidden!.v
        : trace.y.v;
    const dPre = zeros(preRows.length, d);

    // Output projection: logits = wOut · pre.
    for (let i = 0; i < dLogits.length; i++) {
      for (let o = 0; o < this.outputUnits; o++) {
        const dl = dLogits[i][o];
        if (dl === 0) continue;
        const wRow = this.wOut.w[o];
        const gRow = this.wOut.g[o];
        const pre = preRows[i];
        for (let c = 0; c < d; c++) {
          gRow[c] += dl * pre[c];
          dPre[i][c] += dl * wRow[c];
        }
      }
    }

    // FF backward (optional): pre = relu(wFF · preInput).
    let dPreInput = dPre;
    if (this.wFF) {
      const hv = trace.hidden!.v;
      const dHidden = trace.hidden!.g; // grad wrt the relu output (for the viz)
      const dPreIn = zeros(preRows.length, d);
      for (let i = 0; i < hv.length; i++) {
        for (let a = 0; a < d; a++) {
          dHidden[i][a] = dPre[i][a];
          const dz = hv[i][a] > 0 ? dPre[i][a] : 0; // relu'
          if (dz === 0) continue;
          const gRow = this.wFF.g[a];
          const wRow = this.wFF.w[a];
          const inRow = classify ? trace.pooled!.v[i] : trace.y.v[i];
          for (let b = 0; b < d; b++) {
            gRow[b] += dz * inRow[b];
            dPreIn[i][b] += dz * wRow[b];
          }
        }
      }
      dPreInput = dPreIn;
    }

    // dY: residual source. Classification routes the single pooled grad to all
    // positions; transduction is per position.
    const dY = zeros(n, d);
    if (classify) {
      const dPooled = trace.pooled!.g[0];
      for (let c = 0; c < d; c++) dPooled[c] = dPreInput[0][c];
      for (let i = 0; i < n; i++) for (let c = 0; c < d; c++) dY[i][c] += dPooled[c] / n;
    } else {
      for (let i = 0; i < n; i++) for (let c = 0; c < d; c++) dY[i][c] += dPreInput[i][c];
    }

    // Residual y = x + out.
    const dX = zeros(n, d);
    const dOut = zeros(n, d);
    for (let i = 0; i < n; i++)
      for (let c = 0; c < d; c++) {
        dX[i][c] += dY[i][c];
        dOut[i][c] += dY[i][c];
      }

    // out[i] = Σ_j attnW[i][j] · v[j].
    const dAttnW = zeros(n, n);
    const dV = zeros(n, d);
    const vv = trace.v.v;
    const aw = trace.attnW.v;
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        let da = 0;
        const a = aw[i][j];
        for (let c = 0; c < d; c++) {
          da += dOut[i][c] * vv[j][c];
          dV[j][c] += dOut[i][c] * a;
        }
        dAttnW[i][j] = da;
      }

    // softmax backward per row: dScores = aw * (dAttnW - Σ(dAttnW*aw)).
    const dScores = zeros(n, n);
    for (let i = 0; i < n; i++) {
      let dotsum = 0;
      for (let j = 0; j < n; j++) dotsum += dAttnW[i][j] * aw[i][j];
      for (let j = 0; j < n; j++) dScores[i][j] = aw[i][j] * (dAttnW[i][j] - dotsum);
    }

    // scores[i][j] = invSqrtD · q[i]·k[j].
    const invSqrtD = 1 / Math.sqrt(d);
    const dQ = zeros(n, d);
    const dK = zeros(n, d);
    const qv = trace.q.v;
    const kv = trace.k.v;
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        const s = dScores[i][j] * invSqrtD;
        if (s === 0) continue;
        for (let c = 0; c < d; c++) {
          dQ[i][c] += s * kv[j][c];
          dK[j][c] += s * qv[i][c];
        }
      }

    // q/k/v = W · x ; accumulate weight grads and dX.
    const proj = (dProj: number[][], W: Param) => {
      for (let i = 0; i < n; i++) {
        const xi = trace.x.v[i];
        for (let a = 0; a < d; a++) {
          const dp = dProj[i][a];
          if (dp === 0) continue;
          const gRow = W.g[a];
          const wRow = W.w[a];
          for (let b = 0; b < d; b++) {
            gRow[b] += dp * xi[b];
            dX[i][b] += dp * wRow[b];
          }
        }
      }
    };
    proj(dQ, this.wq);
    proj(dK, this.wk);
    proj(dV, this.wv);

    // x = tok + pos → embeddings.
    for (let i = 0; i < n; i++) {
      const id = trace.oneHot[i].indexOf(1);
      const tg = this.tokTable.g[id];
      const pg = this.posTable.g[i];
      for (let c = 0; c < d; c++) {
        tg[c] += dX[i][c];
        pg[c] += dX[i][c];
      }
    }

    // Record activation grads for the visualization.
    trace.tok.g = dX.map((r) => r.slice());
    trace.pos.g = dX.map((r) => r.slice());
    trace.x.g = dX;
    trace.q.g = dQ;
    trace.k.g = dK;
    trace.v.g = dV;
    trace.scores.g = dScores;
    trace.attnW.g = dAttnW;
    trace.out.g = dOut;
    trace.y.g = dY;
  }
}

/** Deterministic model RNG from the app seed (matches scalar model offset). */
export function modelRng(seed: number): () => number {
  return mulberry32(seed ^ 0x9e3779b9);
}
