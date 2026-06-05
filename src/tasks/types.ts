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
  test: Example[];
  /** sampleKey(input) of every test example, for rejection sampling. */
  testKeys: Set<string>;
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
