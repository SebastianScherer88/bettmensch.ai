import { X, Box, Cpu, HardDrive, Zap, Package, Terminal, Cloud, MonitorSmartphone } from "lucide-react";
import type { DagTask, RunStatus, TaskOutput } from "../api";
import { StatusBadge } from "./StatusBadge";
import { ArtifactPreviewCard } from "./ArtifactPreviewCard";
import { SourceVisual } from "./SourceVisual";
import { CodeBlock } from "./CodeBlock";
import { JsonTree } from "./JsonTree";

export interface TaskRuntime {
  status: RunStatus;
  startedAt: string;
  endedAt: string | null;
  outputs: TaskOutput[];
  logs: string | null;
}

interface TaskPanelProps {
  task: DagTask;
  onClose: () => void;
  runtime?: TaskRuntime;
}

function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-500">
      {children}
    </h3>
  );
}

const BACKEND_LABELS: Record<string, string> = {
  local: "Local",
  aws_batch: "AWS Batch",
  aws_lambda: "AWS Lambda",
};

export function TaskPanel({ task, onClose, runtime }: TaskPanelProps) {
  const hasResources =
    task.resource_requirements.cpu != null ||
    task.resource_requirements.memory != null ||
    task.resource_requirements.gpu != null;
  const hasUv = task.uv_requirements.packages.length > 0 || task.uv_requirements.python != null;
  const isLocalBackend = task.compute_backend.name === "local";
  const hasBackendConfig = Object.keys(task.compute_backend.config).length > 0;

  return (
    <div className="fixed inset-0 z-40 flex justify-end">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="relative z-10 flex h-full w-full max-w-2xl flex-col overflow-y-auto border-l border-slate-800 bg-slate-950 shadow-2xl">
        <div className="sticky top-0 z-10 flex items-center justify-between border-b border-slate-800 bg-slate-950/95 px-8 py-5 backdrop-blur">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-sky-500/10">
              <Box size={18} className="text-sky-400" />
            </div>
            <h2 className="font-mono text-xl font-semibold text-white">{task.name}</h2>
            <span
              className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium ${
                isLocalBackend
                  ? "border-slate-700 bg-slate-800/60 text-slate-400"
                  : "border-violet-500/30 bg-violet-500/10 text-violet-300"
              }`}
            >
              {isLocalBackend ? <MonitorSmartphone size={12} /> : <Cloud size={12} />}
              {BACKEND_LABELS[task.compute_backend.name] ?? task.compute_backend.name}
            </span>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-slate-500 hover:bg-slate-900 hover:text-slate-200"
          >
            <X size={20} />
          </button>
        </div>

        <div className="space-y-8 px-8 py-6">
          {runtime && (
            <div className="flex items-center gap-4 rounded-xl border border-slate-800 bg-slate-900/50 p-4">
              <StatusBadge status={runtime.status} />
              <span className="text-sm text-slate-400">
                {new Date(runtime.startedAt).toLocaleString()}
                {runtime.endedAt && ` → ${new Date(runtime.endedAt).toLocaleTimeString()}`}
              </span>
            </div>
          )}

          <section>
            <SectionHeading>Inputs</SectionHeading>
            {task.inputs.length === 0 && (
              <p className="text-sm text-slate-500">None declared.</p>
            )}
            <div className="space-y-3">
              {task.inputs.map((input) => (
                <div
                  key={input.name}
                  className="flex flex-wrap items-center gap-3 rounded-xl border border-slate-800 bg-slate-900/40 p-4"
                >
                  <span className="font-mono text-base font-medium text-slate-100">
                    {input.name}
                  </span>
                  <span className="text-slate-600">sourced from</span>
                  <SourceVisual source={input.source} />
                </div>
              ))}
            </div>
          </section>

          <section>
            <SectionHeading>Outputs</SectionHeading>
            <div className="space-y-3">
              {task.outputs.map((outputName) => {
                const runtimeOutput = runtime?.outputs.find((o) => o.output_name === outputName);
                if (runtimeOutput) {
                  return (
                    <ArtifactPreviewCard
                      key={outputName}
                      outputName={outputName}
                      artifactKey={runtimeOutput.artifact_key}
                    />
                  );
                }
                return (
                  <div
                    key={outputName}
                    className="flex items-center justify-between rounded-xl border border-slate-800 bg-slate-900/40 p-4"
                  >
                    <span className="font-mono text-base text-slate-200">{outputName}</span>
                    <span className="rounded-full bg-slate-800 px-3 py-1 text-xs text-slate-400">
                      {task.output_materializers[outputName] ?? "unknown"}
                    </span>
                  </div>
                );
              })}
            </div>
          </section>

          {(hasResources || hasUv) && (
            <section>
              <SectionHeading>Runtime requirements</SectionHeading>
              <div className="flex flex-wrap gap-2">
                {task.resource_requirements.cpu != null && (
                  <span className="inline-flex items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-1.5 text-sm text-slate-300">
                    <Cpu size={13} className="text-slate-500" /> cpu: {task.resource_requirements.cpu}
                  </span>
                )}
                {task.resource_requirements.memory != null && (
                  <span className="inline-flex items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-1.5 text-sm text-slate-300">
                    <HardDrive size={13} className="text-slate-500" /> memory:{" "}
                    {task.resource_requirements.memory}
                  </span>
                )}
                {task.resource_requirements.gpu != null && (
                  <span className="inline-flex items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-1.5 text-sm text-slate-300">
                    <Zap size={13} className="text-slate-500" /> gpu: {task.resource_requirements.gpu}
                  </span>
                )}
                {task.uv_requirements.python != null && (
                  <span className="inline-flex items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-1.5 text-sm text-slate-300">
                    <Package size={13} className="text-slate-500" /> python{" "}
                    {task.uv_requirements.python}
                  </span>
                )}
                {task.uv_requirements.packages.map((pkg) => (
                  <span
                    key={pkg}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-emerald-500/20 bg-emerald-500/5 px-3 py-1.5 font-mono text-sm text-emerald-300"
                  >
                    {pkg}
                  </span>
                ))}
              </div>
            </section>
          )}

          {!isLocalBackend && hasBackendConfig && (
            <section>
              <SectionHeading>
                {BACKEND_LABELS[task.compute_backend.name] ?? task.compute_backend.name} configuration
              </SectionHeading>
              <JsonTree value={task.compute_backend.config} />
            </section>
          )}

          {task.source && (
            <section>
              <SectionHeading>Source</SectionHeading>
              <CodeBlock code={task.source} />
            </section>
          )}

          {runtime && (
            <section>
              <SectionHeading>Logs</SectionHeading>
              <div className="overflow-hidden rounded-xl border border-slate-800 bg-black">
                <div className="flex items-center gap-2 border-b border-slate-800 bg-slate-900/60 px-4 py-2">
                  <Terminal size={13} className="text-slate-500" />
                  <span className="font-mono text-xs text-slate-500">stdout / stderr</span>
                </div>
                <pre className="max-h-80 overflow-auto px-4 py-3 font-mono text-[13px] leading-relaxed text-slate-300">
                  {runtime.logs || "No output captured."}
                </pre>
              </div>
            </section>
          )}
        </div>
      </div>
    </div>
  );
}
