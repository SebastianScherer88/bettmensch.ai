import { Fragment, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ChevronDown, ChevronRight, Package } from "lucide-react";
import { api, type ArtifactSummary } from "../api";
import { StatusBadge } from "../components/StatusBadge";
import { ArtifactPreviewCard } from "../components/ArtifactPreviewCard";

export function ArtifactsSearch() {
  const [pipelineNames, setPipelineNames] = useState<string[]>([]);
  const [pipelineName, setPipelineName] = useState("");
  const [since, setSince] = useState("");
  const [until, setUntil] = useState("");
  const [artifacts, setArtifacts] = useState<ArtifactSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    api.listPipelines().then((pipelines) => setPipelineNames(pipelines.map((p) => p.pipeline_name)));
  }, []);

  const search = () => {
    setLoading(true);
    api
      .listArtifacts({
        pipelineName: pipelineName || undefined,
        since: since ? new Date(since).toISOString() : undefined,
        until: until ? new Date(until).toISOString() : undefined,
      })
      .then(setArtifacts)
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    search();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const rowKey = (a: ArtifactSummary) => `${a.pipeline_run_id}:${a.task_name}:${a.output_name}`;

  return (
    <div>
      <h1 className="mb-1 text-2xl font-semibold text-white">Artifacts</h1>
      <p className="mb-6 text-sm text-slate-400">
        Every task output recorded across runs, searchable by pipeline and run date.
      </p>

      <div className="mb-4 flex flex-wrap items-end gap-3">
        <div>
          <label className="mb-1 block text-xs text-slate-500">Pipeline</label>
          <select
            value={pipelineName}
            onChange={(e) => setPipelineName(e.target.value)}
            className="w-52 rounded-lg border border-slate-800 bg-slate-900 px-3 py-1.5 text-sm text-slate-100 focus:border-sky-500 focus:outline-none"
          >
            <option value="">All pipelines</option>
            {pipelineNames.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="mb-1 block text-xs text-slate-500">Since</label>
          <input
            type="datetime-local"
            value={since}
            onChange={(e) => setSince(e.target.value)}
            className="rounded-lg border border-slate-800 bg-slate-900 px-3 py-1.5 text-sm text-slate-100 focus:border-sky-500 focus:outline-none"
          />
        </div>
        <div>
          <label className="mb-1 block text-xs text-slate-500">Until</label>
          <input
            type="datetime-local"
            value={until}
            onChange={(e) => setUntil(e.target.value)}
            className="rounded-lg border border-slate-800 bg-slate-900 px-3 py-1.5 text-sm text-slate-100 focus:border-sky-500 focus:outline-none"
          />
        </div>
        <button
          onClick={search}
          className="rounded-lg bg-sky-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-sky-500"
        >
          Search
        </button>
      </div>

      {loading && <p className="text-sm text-slate-500">Loading...</p>}
      {error && <p className="text-sm text-rose-400">{error}</p>}
      {!loading && !error && artifacts.length === 0 && (
        <div className="rounded-lg border border-dashed border-slate-800 p-8 text-center text-sm text-slate-500">
          No artifacts match this search.
        </div>
      )}

      {artifacts.length > 0 && (
        <div className="overflow-hidden rounded-lg border border-slate-800">
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-900 text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th className="w-8" />
                <th className="px-4 py-2.5 font-medium">Pipeline</th>
                <th className="px-4 py-2.5 font-medium">Run</th>
                <th className="px-4 py-2.5 font-medium">Task</th>
                <th className="px-4 py-2.5 font-medium">Output</th>
                <th className="px-4 py-2.5 font-medium">Type</th>
                <th className="px-4 py-2.5 font-medium">Run status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {artifacts.map((a) => {
                const key = rowKey(a);
                const isOpen = expanded === key;
                return (
                  <Fragment key={key}>
                    <tr
                      className="cursor-pointer hover:bg-slate-900/60"
                      onClick={() => setExpanded(isOpen ? null : key)}
                    >
                      <td className="px-2 text-slate-500">
                        {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                      </td>
                      <td className="px-4 py-3 font-medium text-slate-100">{a.pipeline_name}</td>
                      <td className="px-4 py-3">
                        <Link
                          to={`/runs/${a.pipeline_run_id}`}
                          onClick={(e) => e.stopPropagation()}
                          className="font-mono text-xs text-slate-400 hover:text-sky-400"
                        >
                          {a.pipeline_run_id.slice(0, 8)}
                        </Link>
                      </td>
                      <td className="px-4 py-3 text-slate-300">{a.task_name}</td>
                      <td className="px-4 py-3 text-slate-300">{a.output_name}</td>
                      <td className="px-4 py-3">
                        {a.materializer ? (
                          <span className="flex items-center gap-1.5 text-xs text-slate-400">
                            <Package size={13} /> {a.materializer}
                          </span>
                        ) : (
                          <span className="text-slate-600">—</span>
                        )}
                      </td>
                      <td className="px-4 py-3">
                        <StatusBadge status={a.run_status} />
                      </td>
                    </tr>
                    {isOpen && (
                      <tr>
                        <td colSpan={7} className="bg-slate-950/40 px-4 py-3">
                          <ArtifactPreviewCard outputName={a.output_name} artifactKey={a.artifact_key} />
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
