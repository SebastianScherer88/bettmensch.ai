import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { CheckCircle2, CircleDashed, Cloud, CloudOff } from "lucide-react";
import { api, type PipelineSummary } from "../api";
import { StatusBadge } from "../components/StatusBadge";

const FILTER_TABS = ["all", "assembled", "registered"] as const;

export function PipelinesList() {
  const [pipelines, setPipelines] = useState<PipelineSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [nameFilter, setNameFilter] = useState("");
  const [filter, setFilter] = useState<(typeof FILTER_TABS)[number]>("all");

  useEffect(() => {
    api
      .listPipelines()
      .then(setPipelines)
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setLoading(false));
  }, []);

  const filtered = useMemo(
    () =>
      pipelines.filter((p) => {
        if (filter === "assembled" && !p.is_assembled) return false;
        if (filter === "registered" && !p.is_registered) return false;
        if (nameFilter && !p.pipeline_name.toLowerCase().includes(nameFilter.toLowerCase()))
          return false;
        return true;
      }),
    [pipelines, filter, nameFilter],
  );

  return (
    <div>
      <h1 className="mb-1 text-2xl font-semibold text-white">Pipelines</h1>
      <p className="mb-6 text-sm text-slate-400">
        Every pipeline this store knows about, whether it's only been assembled locally or
        registered with a backend orchestrator.
      </p>

      <div className="mb-4 flex items-center gap-3">
        <input
          value={nameFilter}
          onChange={(e) => setNameFilter(e.target.value)}
          placeholder="Filter by name..."
          className="w-64 rounded-lg border border-slate-800 bg-slate-900 px-3 py-1.5 text-sm text-slate-100 placeholder:text-slate-500 focus:border-sky-500 focus:outline-none"
        />
        <div className="flex gap-1 rounded-lg border border-slate-800 bg-slate-900 p-1">
          {FILTER_TABS.map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`rounded-md px-3 py-1 text-xs font-medium capitalize transition-colors ${
                filter === f ? "bg-slate-700 text-white" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {loading && <p className="text-sm text-slate-500">Loading...</p>}
      {error && <p className="text-sm text-rose-400">{error}</p>}
      {!loading && !error && filtered.length === 0 && (
        <div className="rounded-lg border border-dashed border-slate-800 p-8 text-center text-sm text-slate-500">
          No pipelines match. Assemble, register, or run one to see it here.
        </div>
      )}

      {filtered.length > 0 && (
        <div className="overflow-hidden rounded-lg border border-slate-800">
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-900 text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-4 py-2.5 font-medium">Pipeline</th>
                <th className="px-4 py-2.5 font-medium">Assembled</th>
                <th className="px-4 py-2.5 font-medium">Registered</th>
                <th className="px-4 py-2.5 font-medium">Runs</th>
                <th className="px-4 py-2.5 font-medium">Last run</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {filtered.map((p) => (
                <tr key={p.pipeline_name} className="hover:bg-slate-900/60">
                  <td className="px-4 py-3">
                    <Link
                      to={`/pipelines/${encodeURIComponent(p.pipeline_name)}`}
                      className="font-medium text-slate-100 hover:text-sky-400"
                    >
                      {p.pipeline_name}
                    </Link>
                  </td>
                  <td className="px-4 py-3">
                    {p.is_assembled ? (
                      <span className="flex items-center gap-1.5 text-slate-400">
                        <CheckCircle2 size={14} className="text-emerald-400" />
                        {p.last_assembled_at && new Date(p.last_assembled_at).toLocaleString()}
                      </span>
                    ) : (
                      <span className="flex items-center gap-1.5 text-slate-600">
                        <CircleDashed size={14} />
                        not assembled
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    {p.is_registered ? (
                      <span className="flex items-center gap-1.5 text-slate-400">
                        <Cloud size={14} className="text-sky-400" />
                        {p.backend}
                      </span>
                    ) : (
                      <span className="flex items-center gap-1.5 text-slate-600">
                        <CloudOff size={14} />
                        not registered
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-slate-400">{p.run_count}</td>
                  <td className="px-4 py-3">
                    {p.last_run_status ? (
                      <StatusBadge status={p.last_run_status} />
                    ) : (
                      <span className="text-slate-600">—</span>
                    )}
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
