import { useEffect, useState } from "react";
import { api, type ArtifactPreview } from "../api";
import { JsonTree } from "./JsonTree";

export function ArtifactPreviewCard({
  outputName,
  artifactKey,
}: {
  outputName: string;
  artifactKey: string;
}) {
  const [preview, setPreview] = useState<ArtifactPreview | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    setPreview(null);
    api
      .previewArtifact(artifactKey)
      .then(setPreview)
      .finally(() => setLoading(false));
  }, [artifactKey]);

  return (
    <div className="rounded-lg border border-slate-800/60 bg-slate-950/40 p-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-sm font-medium text-slate-200">{outputName}</span>
        {preview && (
          <span className="rounded bg-slate-800 px-2 py-0.5 text-xs text-slate-400">
            {preview.materializer ?? "unknown"}
          </span>
        )}
      </div>
      <div className="mb-2 truncate font-mono text-xs text-slate-500">{artifactKey}</div>
      {loading && <p className="text-xs text-slate-500">Loading preview...</p>}
      {preview?.previewable && <JsonTree value={preview.value} />}
      {preview && !preview.previewable && (
        <p className="text-xs text-slate-500">
          {preview.value_type ?? "This value"} isn&apos;t JSON-previewable in this viewer.
        </p>
      )}
    </div>
  );
}
