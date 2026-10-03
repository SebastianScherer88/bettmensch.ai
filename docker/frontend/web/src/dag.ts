import type { DagStructure, DagTask } from "./api";

/**
 * Defensively normalizes a `dag_structure` dict into a real `DagStructure`.
 *
 * An assembly's `dag_structure` (from `serialize_assembled_pipeline`) is
 * always well-formed. A registration's is not: the store deliberately
 * leaves it an unshaped, arbitrary dict (no Layer 2 compiler exists yet to
 * populate it consistently - see design-decisions.md), so older/example
 * registrations may hold e.g. `{"tasks": ["add"], "edges": []}` - plain
 * task name strings, no ranks or IO bindings. This fills in sane defaults
 * for whatever's missing rather than letting `DagView` crash on it.
 */
export function normalizeDagStructure(raw: unknown): DagStructure {
  if (!raw || typeof raw !== "object") {
    return { tasks: [], edges: [], output: null };
  }

  const obj = raw as Record<string, unknown>;
  const rawTasks = Array.isArray(obj.tasks) ? obj.tasks : [];

  const tasks: DagTask[] = rawTasks.map((t, i) => {
    if (typeof t === "string") {
      return {
        name: t,
        rank: 0,
        inputs: [],
        outputs: [],
        output_materializers: {},
        resource_requirements: { cpu: null, memory: null, gpu: null },
        uv_requirements: { packages: [], python: null },
        source: null,
        compute_backend: { name: "local", config: {} },
      };
    }
    const partial = (t ?? {}) as Partial<DagTask>;
    return {
      name: partial.name ?? `task-${i}`,
      rank: partial.rank ?? 0,
      inputs: partial.inputs ?? [],
      outputs: partial.outputs ?? [],
      output_materializers: partial.output_materializers ?? {},
      resource_requirements: partial.resource_requirements ?? {
        cpu: null,
        memory: null,
        gpu: null,
      },
      uv_requirements: partial.uv_requirements ?? { packages: [], python: null },
      source: partial.source ?? null,
      compute_backend: partial.compute_backend ?? { name: "local", config: {} },
    };
  });

  const rawEdges = Array.isArray(obj.edges) ? obj.edges : [];
  const edges: [string, string][] = rawEdges
    .filter((e): e is [unknown, unknown] => Array.isArray(e) && e.length === 2)
    .map(([from, to]) => [String(from), String(to)]);

  const output = (obj.output as DagStructure["output"]) ?? null;

  return { tasks, edges, output };
}
