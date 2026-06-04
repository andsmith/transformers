/**
 * Scalar reverse-mode autodiff (micrograd-style).
 *
 * Every number that flows through the model is a `Value` node: it stores its
 * forward `data`, its accumulated `grad`, the children it was computed from,
 * and a local `_backward` closure that pushes gradient onto those children.
 *
 * This per-scalar graph is the seam the visualization reads from — once
 * `backward()` runs, every activation and every weight has a `.grad` we can
 * color in a Hinton diagram.
 */

let nextId = 0;

export class Value {
  /** Forward value. */
  data: number;
  /** Accumulated gradient of the loss w.r.t. this node (after backward()). */
  grad = 0;
  /** Optional human-readable label, handy for the visualization. */
  label: string;
  /** The op that produced this node (e.g. "+", "*", "tanh"), "" for leaves. */
  readonly op: string;
  /** Unique id, used for de-duping in the topological sort and for the viz. */
  readonly id: number;

  /** Parents in the compute graph. */
  private readonly _prev: Value[];
  /** Local backward: distribute this.grad onto _prev. No-op for leaves. */
  private _backward: () => void = () => {};

  constructor(data: number, children: Value[] = [], op = "", label = "") {
    this.data = data;
    this._prev = children;
    this.op = op;
    this.label = label;
    this.id = nextId++;
  }

  private static wrap(x: Value | number): Value {
    return x instanceof Value ? x : new Value(x);
  }

  add(other: Value | number): Value {
    const o = Value.wrap(other);
    const out = new Value(this.data + o.data, [this, o], "+");
    out._backward = () => {
      this.grad += out.grad;
      o.grad += out.grad;
    };
    return out;
  }

  sub(other: Value | number): Value {
    return this.add(Value.wrap(other).neg());
  }

  mul(other: Value | number): Value {
    const o = Value.wrap(other);
    const out = new Value(this.data * o.data, [this, o], "*");
    out._backward = () => {
      this.grad += o.data * out.grad;
      o.grad += this.data * out.grad;
    };
    return out;
  }

  div(other: Value | number): Value {
    const o = Value.wrap(other);
    return this.mul(o.pow(-1));
  }

  neg(): Value {
    return this.mul(-1);
  }

  /** Raise to a constant power. */
  pow(exponent: number): Value {
    const out = new Value(Math.pow(this.data, exponent), [this], `**${exponent}`);
    out._backward = () => {
      this.grad += exponent * Math.pow(this.data, exponent - 1) * out.grad;
    };
    return out;
  }

  exp(): Value {
    const e = Math.exp(this.data);
    const out = new Value(e, [this], "exp");
    out._backward = () => {
      this.grad += e * out.grad;
    };
    return out;
  }

  log(): Value {
    const out = new Value(Math.log(this.data), [this], "log");
    out._backward = () => {
      this.grad += (1 / this.data) * out.grad;
    };
    return out;
  }

  tanh(): Value {
    const t = Math.tanh(this.data);
    const out = new Value(t, [this], "tanh");
    out._backward = () => {
      this.grad += (1 - t * t) * out.grad;
    };
    return out;
  }

  relu(): Value {
    const out = new Value(this.data > 0 ? this.data : 0, [this], "relu");
    out._backward = () => {
      this.grad += (out.data > 0 ? 1 : 0) * out.grad;
    };
    return out;
  }

  /** Children of this node (read-only view for the visualization). */
  children(): readonly Value[] {
    return this._prev;
  }

  /**
   * Reverse-mode backprop from this node. Seeds this.grad = 1, then walks the
   * graph in reverse topological order invoking each node's local _backward.
   */
  backward(): void {
    const topo: Value[] = [];
    const visited = new Set<number>();
    const build = (v: Value) => {
      if (visited.has(v.id)) return;
      visited.add(v.id);
      for (const child of v._prev) build(child);
      topo.push(v);
    };
    build(this);

    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
  }
}

/** Convenience constructor. */
export function val(data: number, label = ""): Value {
  return new Value(data, [], "", label);
}
