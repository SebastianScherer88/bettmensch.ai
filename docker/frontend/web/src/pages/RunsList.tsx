import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api, type PipelineRun } from "../api";
import { StatusBadge } from "../components/StatusBadge";

const STATUS_TABS = ["all", "running", "succeeded", "failed"] as const;

function duration(start: string, end: string | null): string {
  if (!end) return "—";
  const ms = new Date(end).getTime() - new Date(start).getTime();
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

export function RunsList() {
  const [runs, setRuns] = useState<PipelineRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [nameFilter, setNameFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState<(typeof STATUS_TABS)[number]>("all");

  useEffect(() => {
    api
      .listPipelineRuns()
      .then(setRuns)
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setLoading(false));
  }, []);

  const filtered = useMemo(
    () =>
      runs.filter(
        (r) =>
          (statusFilter === "all" || r.status === statusFilter) &&
          (!nameFilter || r.pipeline_name.toLowerCase().includes(nameFilter.toLowerCase())),
      ),
    [runs, statusFilter, nameFilter],
  );

  return (
    <div>
      <h1 className="mb-1 text-2xl font-semibold text-white">Pipeline Runs</h1>
      <p className="mb-6 text-sm text-slate-400">
        Every run recorded in the metadata store, most recent first.
      </p>

      <div className="mb-4 flex items-center gap-3">
        <input
          value={nameFilter}
          onChange={(e) => setNameFilter(e.target.value)}
          placeholder="Filter by pipeline name..."
          className="w-64 rounded-lg border border-slate-800 bg-slate-900 px-3 py-1.5 text-sm text-slate-100 placeholder:text-slate-500 focus:border-sky-500 focus:outline-none"
        />
        <div className="flex gap-1 rounded-lg border border-slate-800 bg-slate-900 p-1">
          {STATUS_TABS.map((s) => (
            <button
              key={s}
              onClick={() => setStatusFilter(s)}
              className={`rounded-md px-3 py-1 text-xs font-medium capitalize transition-colors ${
                statusFilter === s
                  ? "bg-slate-700 text-white"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {loading && <p className="text-sm text-slate-500">Loading...</p>}
      {error && (
        <p className="text-sm text-rose-400">Couldn&apos;t reach the metadata store: {error}</p>
      )}

      {!loading && !error && filtered.length === 0 && (
        <div className="rounded-lg border border-dashed border-slate-800 p-8 text-center text-sm text-slate-500">
          No pipeline runs match. Point a <code className="text-slate-400">LocalRunner</code> at
          this store to create some.
        </div>
      )}

      {filtered.length > 0 && (
        <div className="overflow-hidden rounded-lg border border-slate-800">
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-900 text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-4 py-2.5 font-medium">Pipeline</th>
                <th className="px-4 py-2.5 font-medium">Run ID</th>
                <th className="px-4 py-2.5 font-medium">Status</th>
                <th className="px-4 py-2.5 font-medium">Started</th>
                <th className="px-4 py-2.5 font-medium">Duration</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {filtered.map((run) => (
                <tr key={run.pipeline_run_id} className="hover:bg-slate-900/60">
                  <td className="px-4 py-3">
                    <Link
                      to={`/runs/${run.pipeline_run_id}`}
                      className="font-medium text-slate-100 hover:text-sky-400"
                    >
                      {run.pipeline_name}
                    </Link>
                  </td>
                  <td className="px-4 py-3 font-mono text-xs text-slate-500">
                    {run.pipeline_run_id.slice(0, 8)}
                  </td>
                  <td className="px-4 py-3">
                    <StatusBadge status={run.status} />
                  </td>
                  <td className="px-4 py-3 text-slate-400">
                    {new Date(run.started_at).toLocaleString()}
                  </td>
                  <td className="px-4 py-3 text-slate-400">
                    {duration(run.started_at, run.ended_at)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
