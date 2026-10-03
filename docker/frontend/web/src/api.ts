export type RunStatus = "running" | "succeeded" | "failed";

export type IOSource =
  | { kind: "static"; value: unknown }
  | { kind: "pipeline_input"; name: string }
  | { kind: "task_output"; task: string; output: string };

export interface DagTaskInput {
  name: string;
  source: IOSource;
}

export interface ResourceRequirements {
  cpu: string | number | null;
  memory: string | null;
  gpu: number | null;
}

export interface UvRequirements {
  packages: string[];
  python: string | null;
}

export interface ComputeBackend {
  name: string;
  config: Record<string, unknown>;
}

export interface DagTask {
  name: string;
  rank: number;
  inputs: DagTaskInput[];
  outputs: string[];
  output_materializers: Record<string, string>;
  resource_requirements: ResourceRequirements;
  uv_requirements: UvRequirements;
  source: string | null;
  compute_backend: ComputeBackend;
}

export interface DagOutput {
  name: string;
  source: IOSource;
}

export interface DagStructure {
  tasks: DagTask[];
  edges: [string, string][];
  output: DagOutput | null;
}

export interface PipelineInputSpec {
  required: boolean;
  default: unknown;
  materializer: string;
}

export interface PipelineAssembly {
  pipeline_assembly_id: string;
  pipeline_name: string;
  dag_structure: DagStructure;
  pipeline_inputs: Record<string, PipelineInputSpec>;
  assembled_at: string;
}

export interface PipelineRun {
  pipeline_run_id: string;
  pipeline_name: string;
  status: RunStatus;
  started_at: string;
  ended_at: string | null;
  pipeline_assembly_id: string | null;
}

export interface TaskRun {
  pipeline_run_id: string;
  task_name: string;
  status: RunStatus;
  started_at: string;
  ended_at: string | null;
  logs: string | null;
}

export interface TaskOutput {
  pipeline_run_id: string;
  task_name: string;
  output_name: string;
  artifact_key: string;
}

export interface ArtifactPreview {
  artifact_key: string;
  materializer: string | null;
  value_type: string | null;
  previewable: boolean;
  value: unknown;
  error: string | null;
}

export interface ArtifactSummary {
  pipeline_name: string;
  pipeline_run_id: string;
  run_started_at: string;
  run_status: RunStatus;
  task_name: string;
  output_name: string;
  artifact_key: string;
  materializer: string | null;
  value_type: string | null;
}

export interface PipelineRegistration {
  pipeline_registration_id: string;
  pipeline_name: string;
  backend: string;
  dag_structure: DagStructure;
  pipeline_inputs: Record<string, PipelineInputSpec>;
  backend_metadata: Record<string, unknown>;
  registered_at: string;
  deregistered_at: string | null;
  is_active: boolean;
}

export interface Trigger {
  trigger_id: string;
  pipeline_registration_id: string;
  trigger_type: string;
  trigger_config: Record<string, unknown>;
  registered_at: string;
  deregistered_at: string | null;
  is_active: boolean;
}

export interface PipelineSummary {
  pipeline_name: string;
  is_assembled: boolean;
  last_assembled_at: string | null;
  is_registered: boolean;
  backend: string | null;
  last_registered_at: string | null;
  run_count: number;
  last_run_status: RunStatus | null;
  last_run_at: string | null;
}

export interface PipelineDetail {
  pipeline_name: string;
  latest_assembly: PipelineAssembly | null;
  registrations: PipelineRegistration[];
  run_count: number;
  last_run: PipelineRun | null;
}

const BASE = "/api";

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // response wasn't JSON - keep statusText
    }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  listPipelines() {
    return getJson<PipelineSummary[]>(`/pipelines`);
  },
  getPipeline(pipelineName: string) {
    return getJson<PipelineDetail>(`/pipelines/${encodeURIComponent(pipelineName)}`);
  },
  getPipelineAssembly(assemblyId: string) {
    return getJson<PipelineAssembly>(`/pipeline-assemblies/${assemblyId}`);
  },
  listPipelineRuns(params: { pipelineName?: string; status?: string } = {}) {
    const q = new URLSearchParams();
    if (params.pipelineName) q.set("pipeline_name", params.pipelineName);
    if (params.status) q.set("status", params.status);
    return getJson<PipelineRun[]>(`/pipeline-runs?${q}`);
  },
  getPipelineRun(runId: string) {
    return getJson<PipelineRun>(`/pipeline-runs/${runId}`);
  },
  listTaskRuns(runId: string) {
    return getJson<TaskRun[]>(`/pipeline-runs/${runId}/task-runs`);
  },
  listTaskOutputs(runId: string, taskName: string) {
    return getJson<TaskOutput[]>(
      `/pipeline-runs/${runId}/task-runs/${encodeURIComponent(taskName)}/outputs`,
    );
  },
  listArtifacts(params: { pipelineName?: string; since?: string; until?: string } = {}) {
    const q = new URLSearchParams();
    if (params.pipelineName) q.set("pipeline_name", params.pipelineName);
    if (params.since) q.set("since", params.since);
    if (params.until) q.set("until", params.until);
    return getJson<ArtifactSummary[]>(`/artifacts?${q}`);
  },
  previewArtifact(key: string) {
    return getJson<ArtifactPreview>(`/artifacts/preview?key=${encodeURIComponent(key)}`);
  },
  listRegistrations(params: { pipelineName?: string; activeOnly?: boolean } = {}) {
    const q = new URLSearchParams();
    if (params.pipelineName) q.set("pipeline_name", params.pipelineName);
    if (params.activeOnly) q.set("active_only", "true");
    return getJson<PipelineRegistration[]>(`/pipeline-registrations?${q}`);
  },
  getRegistration(id: string) {
    return getJson<PipelineRegistration>(`/pipeline-registrations/${id}`);
  },
  listTriggers(id: string) {
    return getJson<Trigger[]>(`/pipeline-registrations/${id}/triggers`);
  },
};
