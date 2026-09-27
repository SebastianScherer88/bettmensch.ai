import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ChevronLeft, Cloud, Workflow, Zap } from "lucide-react";
import { api, type DagTask, type PipelineDetail as PipelineDetailData, type Trigger } from "../api";
import { DagView } from "../components/DagView";
import { TaskPanel } from "../components/TaskPanel";
import { JsonTree } from "../components/JsonTree";
import { normalizeDagStructure } from "../dag";

function InfoCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-4">
      <div className="text-xs uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mt-1 text-sm font-medium text-slate-100">{value}</div>
    </div>
  );
}

export function PipelineDetail() {
  const { pipelineName } = useParams<{ pipelineName: string }>();
  const [detail, setDetail] = useState<PipelineDetailData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<string>("assembled");
  const [selectedTask, setSelectedTask] = useState<DagTask | null>(null);
  const [triggers, setTriggers] = useState<Trigger[]>([]);

  useEffect(() => {
    if (!pipelineName) return;
    api
      .getPipeline(pipelineName)
      .then((d) => {
        setDetail(d);
        setActiveTab(d.latest_assembly ? "assembled" : d.registrations[0]?.pipeline_registration_id ?? "");
      })
      .catch((e) => setError(String(e.message ?? e)));
  }, [pipelineName]);

  useEffect(() => {
    const registration = detail?.registrations.find(
      (r) => r.pipeline_registration_id === activeTab,
    );
    if (!registration) {
      setTriggers([]);
      return;
    }
    api.listTriggers(registration.pipeline_registration_id).then(setTriggers);
  }, [detail, activeTab]);

  const activeDag = useMemo(() => {
    if (!detail) return null;
    if (activeTab === "assembled" && detail.latest_assembly) {
      return normalizeDagStructure(detail.latest_assembly.dag_structure);
    }
    const registration = detail.registrations.find(
      (r) => r.pipeline_registration_id === activeTab,
    );
    return registration ? normalizeDagStructure(registration.dag_structure) : null;
  }, [detail, activeTab]);

  const activeRegistration = detail?.registrations.find(
    (r) => r.pipeline_registration_id === activeTab,
  );

  if (error) return <p className="text-sm text-rose-400">{error}</p>;
  if (!detail) return <p className="text-sm text-slate-500">Loading...</p>;

  return (
    <div>
      <Link
        to="/pipelines"
        className="mb-4 inline-flex items-center gap-1 text-sm text-slate-400 hover:text-slate-200"
      >
        <ChevronLeft size={16} /> All pipelines
      </Link>

      <h1 className="mb-6 text-2xl font-semibold text-white">{detail.pipeline_name}</h1>

      <div className="mb-8 grid grid-cols-3 gap-4">
        <InfoCard
          label="Last assembled"
          value={
            detail.latest_assembly
              ? new Date(detail.latest_assembly.assembled_at).toLocaleString()
              : "never"
          }
        />
        <InfoCard
          label="Registrations"
          value={detail.registrations.length ? String(detail.registrations.length) : "none"}
        />
        <InfoCard
          label="Runs"
          value={
            detail.run_count > 0 && detail.last_run
              ? `${detail.run_count} (last ${detail.last_run.status})`
              : String(detail.run_count)
          }
        />
      </div>

      <div className="mb-4 flex gap-1 rounded-lg border border-slate-800 bg-slate-900 p-1 w-fit">
        {detail.latest_assembly && (
          <button
            onClick={() => setActiveTab("assembled")}
            className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
              activeTab === "assembled"
                ? "bg-slate-700 text-white"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Workflow size={13} /> Assembled
          </button>
        )}
        {detail.registrations.map((r) => (
          <button
            key={r.pipeline_registration_id}
            onClick={() => setActiveTab(r.pipeline_registration_id)}
            className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
              activeTab === r.pipeline_registration_id
                ? "bg-slate-700 text-white"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Cloud size={13} /> {r.backend}
            {!r.is_active && <span className="text-slate-600">(retired)</span>}
          </button>
        ))}
      </div>

      {activeRegistration && Object.keys(activeRegistration.backend_metadata).length > 0 && (
        <div className="mb-4">
          <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
            Backend metadata
          </h2>
          <JsonTree value={activeRegistration.backend_metadata} />
        </div>
      )}

      {activeRegistration && triggers.length > 0 && (
        <div className="mb-4">
          <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
            Triggers
          </h2>
          <div className="space-y-2">
            {triggers.map((t) => (
              <div
                key={t.trigger_id}
                className="rounded-lg border border-slate-800 bg-slate-900/30 p-3"
              >
                <div className="mb-1 flex items-center justify-between">
                  <span className="flex items-center gap-1.5 font-medium text-slate-100">
                    <Zap size={13} className="text-amber-400" /> {t.trigger_type}
                  </span>
                  <span
                    className={`rounded-full px-2 py-0.5 text-xs ${
                      t.is_active
                        ? "bg-emerald-500/15 text-emerald-400"
                        : "bg-slate-700/50 text-slate-400"
                    }`}
                  >
                    {t.is_active ? "active" : "inactive"}
                  </span>
                </div>
                <JsonTree value={t.trigger_config} />
              </div>
            ))}
          </div>
        </div>
      )}

      {activeDag && (
        <DagView
          tasks={activeDag.tasks}
          edges={activeDag.edges}
          selectedTask={selectedTask?.name}
          onTaskClick={(name) => {
            const task = activeDag.tasks.find((t) => t.name === name) ?? null;
            setSelectedTask(task);
          }}
        />
      )}

      {selectedTask && <TaskPanel task={selectedTask} onClose={() => setSelectedTask(null)} />}
    </div>
  );
}
