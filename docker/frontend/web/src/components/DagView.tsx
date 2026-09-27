import { useEffect, useMemo, useRef, useState } from "react";
import { Box, RotateCcw } from "lucide-react";
import type { DagTask, RunStatus } from "../api";

const NODE_W = 190;
const NODE_H = 58;
const COL_GAP = 32;
const ROW_GAP = 90;
const PADDING = 32;
const VIEWPORT_HEIGHT = 480;

const STATUS_STROKE: Record<string, string> = {
  running: "#38bdf8",
  succeeded: "#34d399",
  failed: "#fb7185",
};

interface DagViewProps {
  tasks: DagTask[];
  edges: [string, string][];
  statusByTask?: Record<string, RunStatus>;
  selectedTask?: string | null;
  onTaskClick?: (taskName: string) => void;
}

export function DagView({ tasks, edges, statusByTask, selectedTask, onTaskClick }: DagViewProps) {
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const dragState = useRef<{ startX: number; startY: number; panX: number; panY: number } | null>(
    null,
  );
  const [dragging, setDragging] = useState(false);

  const layout = useMemo(() => {
    const ranks: Record<number, DagTask[]> = {};
    for (const task of tasks) {
      (ranks[task.rank] ??= []).push(task);
    }
    const rankIndices = Object.keys(ranks)
      .map(Number)
      .sort((a, b) => a - b);
    const maxCount = Math.max(1, ...rankIndices.map((r) => ranks[r].length));
    const totalWidth = maxCount * NODE_W + (maxCount - 1) * COL_GAP;

    const positions: Record<string, { x: number; y: number }> = {};
    rankIndices.forEach((rank, rowIndex) => {
      const nodes = ranks[rank];
      const rowWidth = nodes.length * NODE_W + (nodes.length - 1) * COL_GAP;
      const xOffset = (totalWidth - rowWidth) / 2;
      nodes.forEach((task, i) => {
        positions[task.name] = {
          x: xOffset + i * (NODE_W + COL_GAP),
          y: rowIndex * (NODE_H + ROW_GAP),
        };
      });
    });

    const height = rankIndices.length * NODE_H + Math.max(0, rankIndices.length - 1) * ROW_GAP;
    return { positions, width: totalWidth, height };
  }, [tasks]);

  useEffect(() => {
    setPan({ x: 0, y: 0 });
  }, [tasks]);

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    dragState.current = { startX: e.clientX, startY: e.clientY, panX: pan.x, panY: pan.y };
    setDragging(true);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragState.current) return;
    const dx = e.clientX - dragState.current.startX;
    const dy = e.clientY - dragState.current.startY;
    setPan({ x: dragState.current.panX + dx, y: dragState.current.panY + dy });
  };

  const endDrag = () => {
    dragState.current = null;
    setDragging(false);
  };

  if (tasks.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-slate-800 p-8 text-center text-sm text-slate-500">
        No tasks to visualize.
      </div>
    );
  }

  const contentWidth = layout.width + PADDING * 2;
  const contentHeight = layout.height + PADDING * 2;

  return (
    <div className="relative overflow-hidden rounded-xl border border-slate-800 bg-slate-950/40">
      <button
        onClick={() => setPan({ x: 0, y: 0 })}
        className="absolute right-3 top-3 z-10 flex items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-900/90 px-2.5 py-1.5 text-xs text-slate-400 hover:text-slate-200"
        title="Reset view"
      >
        <RotateCcw size={12} /> Reset
      </button>
      <div
        className={`flex w-full items-start justify-center overflow-hidden ${dragging ? "cursor-grabbing" : "cursor-grab"}`}
        style={{ height: Math.min(VIEWPORT_HEIGHT, contentHeight + 40) }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerLeave={endDrag}
      >
        <svg
          width={contentWidth}
          height={contentHeight}
          className="shrink-0"
          style={{ transform: `translate(${pan.x}px, ${pan.y}px)` }}
        >
          <g transform={`translate(${PADDING}, ${PADDING})`}>
            {edges.map(([from, to], i) => {
              const a = layout.positions[from];
              const b = layout.positions[to];
              if (!a || !b) return null;
              const x1 = a.x + NODE_W / 2;
              const y1 = a.y + NODE_H;
              const x2 = b.x + NODE_W / 2;
              const y2 = b.y;
              const midY = (y1 + y2) / 2;
              return (
                <path
                  key={i}
                  d={`M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}`}
                  fill="none"
                  stroke="#334155"
                  strokeWidth={2}
                />
              );
            })}

            {tasks.map((task) => {
              const pos = layout.positions[task.name];
              const status = statusByTask?.[task.name];
              const stroke = status ? STATUS_STROKE[status] : "#334155";
              const isSelected = selectedTask === task.name;
              return (
                <g
                  key={task.name}
                  transform={`translate(${pos.x}, ${pos.y})`}
                  onPointerDown={(e) => e.stopPropagation()}
                  onClick={() => onTaskClick?.(task.name)}
                  className={onTaskClick ? "cursor-pointer" : undefined}
                >
                  <rect
                    width={NODE_W}
                    height={NODE_H}
                    rx={10}
                    fill={isSelected ? "#1e293b" : "#0f172a"}
                    stroke={stroke}
                    strokeWidth={isSelected ? 2.5 : 1.5}
                  />
                  <foreignObject width={NODE_W} height={NODE_H}>
                    <div className="flex h-full items-center gap-2 px-3">
                      <Box size={15} className="shrink-0 text-slate-400" />
                      <div className="min-w-0">
                        <div className="truncate text-sm font-medium text-slate-100">
                          {task.name}
                        </div>
                        {status && (
                          <div
                            className="truncate text-[11px] capitalize"
                            style={{ color: stroke }}
                          >
                            {status}
                          </div>
                        )}
                      </div>
                    </div>
                  </foreignObject>
                </g>
              );
            })}
          </g>
        </svg>
      </div>
      <div className="border-t border-slate-800/60 px-3 py-1.5 text-center text-[11px] text-slate-600">
        drag to pan
      </div>
    </div>
  );
}
