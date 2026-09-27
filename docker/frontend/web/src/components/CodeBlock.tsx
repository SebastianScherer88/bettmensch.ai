import { Highlight, themes } from "prism-react-renderer";
import { FileCode2 } from "lucide-react";

export function CodeBlock({ code, label = "python" }: { code: string; label?: string }) {
  const trimmed = code.replace(/\n+$/, "");

  return (
    <div className="overflow-hidden rounded-xl border border-slate-800 bg-[#0d1117]">
      <div className="flex items-center gap-2 border-b border-slate-800 bg-slate-900/60 px-4 py-2">
        <FileCode2 size={14} className="text-slate-500" />
        <span className="font-mono text-xs text-slate-500">{label}</span>
      </div>
      <Highlight theme={themes.nightOwl} code={trimmed} language="python">
        {({ className, style, tokens, getLineProps, getTokenProps }) => (
          <pre
            className={`${className} overflow-x-auto px-4 py-3 text-[13px] leading-relaxed`}
            style={{ ...style, background: "transparent" }}
          >
            {tokens.map((line, i) => (
              <div key={i} {...getLineProps({ line })} className="table-row">
                <span className="table-cell select-none pr-4 text-right text-slate-600">
                  {i + 1}
                </span>
                <span className="table-cell">
                  {line.map((token, key) => (
                    <span key={key} {...getTokenProps({ token })} />
                  ))}
                </span>
              </div>
            ))}
          </pre>
        )}
      </Highlight>
    </div>
  );
}
