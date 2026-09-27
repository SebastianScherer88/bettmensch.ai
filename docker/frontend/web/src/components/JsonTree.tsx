import { useState, type ReactNode } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

function Row({ name, children }: { name?: string; children: ReactNode }) {
  return (
    <div className="py-0.5 font-mono text-sm leading-relaxed">
      {name !== undefined && <span className="text-slate-400">{name}: </span>}
      {children}
    </div>
  );
}

function ValueNode({ value, name, depth }: { value: unknown; name?: string; depth: number }) {
  const [open, setOpen] = useState(depth < 2);

  if (value === null || value === undefined) {
    return (
      <Row name={name}>
        <span className="text-slate-500">null</span>
      </Row>
    );
  }

  if (typeof value === "boolean") {
    return (
      <Row name={name}>
        <span className="text-amber-400">{String(value)}</span>
      </Row>
    );
  }

  if (typeof value === "number") {
    return (
      <Row name={name}>
        <span className="text-sky-400">{value}</span>
      </Row>
    );
  }

  if (typeof value === "string") {
    return (
      <Row name={name}>
        <span className="text-emerald-400">&quot;{value}&quot;</span>
      </Row>
    );
  }

  if (Array.isArray(value)) {
    return (
      <div>
        <button
          onClick={() => setOpen(!open)}
          className="flex items-center gap-1 font-mono text-sm text-slate-300 hover:text-white"
        >
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          {name !== undefined && <span className="text-slate-400">{name}:</span>}
          <span className="text-slate-500">Array[{value.length}]</span>
        </button>
        {open && (
          <div className="ml-3 border-l border-slate-800 pl-3">
            {value.map((v, i) => (
              <ValueNode key={i} value={v} name={String(i)} depth={depth + 1} />
            ))}
          </div>
        )}
      </div>
    );
  }

  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    return (
      <div>
        <button
          onClick={() => setOpen(!open)}
          className="flex items-center gap-1 font-mono text-sm text-slate-300 hover:text-white"
        >
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          {name !== undefined && <span className="text-slate-400">{name}:</span>}
          <span className="text-slate-500">
            {"{"}
            {entries.length}
            {"}"}
          </span>
        </button>
        {open && (
          <div className="ml-3 border-l border-slate-800 pl-3">
            {entries.map(([k, v]) => (
              <ValueNode key={k} value={v} name={k} depth={depth + 1} />
            ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <Row name={name}>
      <span>{String(value)}</span>
    </Row>
  );
}

export function JsonTree({ value }: { value: unknown }) {
  return (
    <div className="overflow-x-auto rounded-lg bg-slate-900/50 p-3">
      <ValueNode value={value} depth={0} />
    </div>
  );
}
