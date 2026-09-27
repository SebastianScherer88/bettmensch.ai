import { Box, LogIn, Braces, Hash } from "lucide-react";
import type { IOSource } from "../api";
import { JsonTree } from "./JsonTree";

function StaticValue({ value }: { value: unknown }) {
  if (value !== null && typeof value === "object") {
    return (
      <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 p-2">
        <JsonTree value={value} />
      </div>
    );
  }

  return (
    <span className="inline-flex items-center gap-1.5 rounded-lg border border-amber-500/25 bg-amber-500/10 px-3 py-1.5 font-mono text-sm text-amber-300">
      <Braces size={13} className="shrink-0 opacity-70" />
      {JSON.stringify(value)}
    </span>
  );
}

export function SourceVisual({ source }: { source: IOSource }) {
  if (source.kind === "static") {
    return <StaticValue value={source.value} />;
  }

  if (source.kind === "pipeline_input") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-lg border border-violet-500/25 bg-violet-500/10 px-3 py-1.5 text-sm text-violet-300">
        <LogIn size={13} className="shrink-0" />
        pipeline input
        <span className="font-mono font-medium text-violet-200">{source.name}</span>
      </span>
    );
  }

  return (
    <span className="inline-flex items-center overflow-hidden rounded-lg border border-sky-500/25 bg-sky-500/10 text-sm text-sky-300">
      <span className="flex items-center gap-1.5 px-3 py-1.5">
        <Box size={13} className="shrink-0" />
        <span className="font-mono font-medium text-sky-200">{source.task}</span>
      </span>
      <span className="h-full w-px bg-sky-500/25" />
      <span className="flex items-center gap-1.5 px-3 py-1.5">
        <Hash size={12} className="shrink-0 opacity-70" />
        <span className="font-mono">{source.output}</span>
      </span>
    </span>
  );
}
