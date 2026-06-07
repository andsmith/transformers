/** The four toy tasks the playground supports. */
export type Task = "copy" | "reverse" | "sort" | "parens";

export type TaskKind = "transduction" | "classification";

export interface TaskSpec {
  id: Task;
  label: string;
  kind: TaskKind;
  /** One-line description shown in the UI. */
  description: string;
}

/**
 * A single example.
 * - Transduction tasks (copy/reverse/sort): `output` is an aligned token
 *   sequence the same length as `input`.
 * - Classification task (parens): `output` is a single-element array, `[0]`
 *   (unbalanced) or `[1]` (balanced).
 */
export interface Example {
  /** Test samples: stable id 0..N-1. On-the-fly training samples: the
   *  iteration count at which they were drawn (display only). */
  index: number;
  input: number[];
  output: number[];
}

/**
 * The dataset is a FIXED, deduplicated test set plus the generation rules.
 * Training samples are drawn on the fly (see datasets.generateTrainExample),
 * rejected against `testKeys` so the test set stays truly held out.
 */
export interface Dataset {
  task: Task;
  vocabSize: number;
  minLen: number;
  maxLen: number;
  /** Length prior: true = each length equally likely; false = length ∝ V^L
   *  (uniform over the whole sample space). */
  uniformLen: boolean;
  /** Parens: max nesting depth; ignored by other tasks. */
  parensMaxDepth: number;
  /** Parens: forbid mixing delimiter types within a nest. */
  parensNoMixedNesting: boolean;
  /** Parens: number of distinct delimiter pair kinds. */
  parensDelims: number;
  /** Grok filters (compiled); empty = no held-out subset. A training draw is
   *  rejected if it matches any of these. */
  filters: RegExp[];
  test: Example[];
  /** sampleKey(input) of every test example, for rejection sampling. */
  testKeys: Set<string>;
  /** Grok match statistics (only when filters are set). */
  matchInfo?: { count: number; mode: "enumerated" | "sampled" };
}

export const TASK_SPECS: Record<Task, TaskSpec> = {
  copy: {
    id: "copy",
    label: "Copy",
    kind: "transduction",
    description: "Reproduce the input sequence unchanged.",
  },
  reverse: {
    id: "reverse",
    label: "Reverse",
    kind: "transduction",
    description: "Output the input sequence in reverse order.",
  },
  sort: {
    id: "sort",
    label: "Sort",
    kind: "transduction",
    description: "Output the input tokens sorted ascending.",
  },
  parens: {
    id: "parens",
    label: "Parens",
    kind: "classification",
    description:
      "Decide whether the delimiters are balanced (distractor symbols are ignored).",
  },
};

export const ALL_TASKS: Task[] = ["copy", "reverse", "sort", "parens"];

export function isClassification(task: Task): boolean {
  return TASK_SPECS[task].kind === "classification";
}
