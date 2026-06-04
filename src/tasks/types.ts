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
 * A single training example.
 * - Transduction tasks (copy/reverse/sort): `output` is an aligned token
 *   sequence the same length as `input`.
 * - Classification task (parens): `output` is a single-element array, `[0]`
 *   (unbalanced) or `[1]` (balanced).
 */
export interface Example {
  /** Stable global id (0..count-1, generation order). Survives the train/test
   *  shuffle so a sample can be referenced from anywhere. */
  index: number;
  input: number[];
  output: number[];
}

export interface Dataset {
  task: Task;
  vocabSize: number;
  /** All generated examples (train + test, in generation order). */
  examples: Example[];
  train: Example[];
  test: Example[];
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
