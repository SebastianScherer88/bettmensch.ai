import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ChevronDown, ChevronLeft, ChevronRight } from "lucide-react";
import {
  api,
  type DagTask,
  type PipelineAssembly,
  type PipelineRun,
  type RunStatus,
  type TaskOutput,
  type TaskRun,
} from "../api";
import { StatusBadge } from "../components/StatusBadge";
import { DagView } from "../components/DagView";
import { TaskPanel, type TaskRuntime } from "../components/TaskPanel";
import { ArtifactPreviewCard } from "../components/ArtifactPreviewCard";
import { normalizeDagStructure } from "../dag";

function InfoCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-4">
      <div className="text-xs uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mt-1 text-sm font-medium text-slate-100">{value}</div>
    </div>
  );
}

// Fallback for runs recorded before assembly-snapshotting existed (no
// pipeline_assembly_id) - a flat, expandable task list instead of the DAG.
function FallbackTaskRow({ runId, taskRun }: { runId: string; taskRun: TaskRun }) {
  const [open, setOpen] = useState(false);
  const [outputs, setOutputs] = useState<TaskOutput[] | null>(null);

  const toggle = () => {
    setOpen(!open);
    if (!outputs) {
      api.listTaskOutputs(runId, taskRun.task_name).then(setOutputs);
    }
  };

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/30">
      <button onClick={toggle} className="flex w-full items-center justify-between px-4 py-3 text-left">
        <div className="flex items-center gap-3">
          {open ? (
            <ChevronDown size={16} className="text-slate-500" />
          ) : (
            <ChevronRight size={16} className="text-slate-500" />
          )}
          <span className="font-medium text-slate-100">{taskRun.task_name}</span>
        </div>
        <StatusBadge status={taskRun.status} />
      </button>
      {open && (
        <div className="border-t border-slate-800 px-4 py-3">
          {!outputs && <p className="text-sm text-slate-500">Loading outputs...</p>}
          {outputs && outputs.length === 0 && (
            <p className="text-sm text-slate-500">No outputs recorded.</p>
          )}
          <div className="space-y-3">
            {outputs?.map((o) => (
              <ArtifactPreviewCard
                key={o.output_name}
                outputName={o.output_name}
                artifactKey={o.artifact_key}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export function RunDetail() {
  const { runId } = useParams<{ runId: string }>();
  const [run, setRun] = useState<PipelineRun | null>(null);
  const [taskRuns, setTaskRuns] = useState<TaskRun[]>([]);
  const [assembly, setAssembly] = useState<PipelineAssembly | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<{ task: DagTask; runtime: TaskRuntime } | null>(null);

  useEffect(() => {
    if (!runId) return;
    api
      .getPipelineRun(runId)
      .then((r) => {
        setRun(r);
        if (r.pipeline_assembly_id) {
          api.getPipelineAssembly(r.pipeline_assembly_id).then(setAssembly);
        }
      })
      .catch((e) => setError(String(e.message ?? e)));
    api.listTaskRuns(runId).then(setTaskRuns).catch((e) => setError(String(e.message ?? e)));
  }, [runId]);

  if (error) return <p className="text-sm text-rose-400">{error}</p>;
  if (!run) return <p className="text-sm text-slate-500">Loading...</p>;

  const dag = assembly ? normalizeDagStructure(assembly.dag_structure) : null;
  const statusByTask: Record<string, RunStatus> = Object.fromEntries(
    taskRuns.map((t) => [t.task_name, t.status]),
  );

  const handleTaskClick = async (taskName: string) => {
    const task = dag?.tasks.find((t) => t.name === taskName);
    const taskRun = taskRuns.find((t) => t.task_name === taskName);
    if (!task || !taskRun || !runId) return;

    const outputs = await api.listTaskOutputs(runId, taskName);
    setSelected({
      task,
      runtime: {
        status: taskRun.status,
        startedAt: taskRun.started_at,
        endedAt: taskRun.ended_at,
        outputs,
        logs: taskRun.logs,
      },
    });
  };

  return (
    <div>
      <Link
        to="/runs"
        className="mb-4 inline-flex items-center gap-1 text-sm text-slate-400 hover:text-slate-200"
      >
        <ChevronLeft size={16} /> All runs
      </Link>

      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-white">{run.pipeline_name}</h1>
          <p className="mt-1 font-mono text-xs text-slate-500">{run.pipeline_run_id}</p>
        </div>
        <StatusBadge status={run.status} />
      </div>

      <div className="mb-8 grid grid-cols-3 gap-4">
        <InfoCard label="Started" value={new Date(run.started_at).toLocaleString()} />
        <InfoCard
          label="Ended"
          value={run.ended_at ? new Date(run.ended_at).toLocaleString() : "—"}
        />
        <InfoCard label="Tasks" value={String(taskRuns.length)} />
      </div>

      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">Tasks</h2>

      {dag ? (
        <DagView
          tasks={dag.tasks}
          edges={dag.edges}
          statusByTask={statusByTask}
          selectedTask={selected?.task.name}
          onTaskClick={handleTaskClick}
        />
      ) : (
        <div className="space-y-2">
          {taskRuns.map((t) => (
            <FallbackTaskRow key={t.task_name} runId={runId!} taskRun={t} />
          ))}
        </div>
      )}

      {selected && (
        <TaskPanel
          task={selected.task}
          runtime={selected.runtime}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  );
}
