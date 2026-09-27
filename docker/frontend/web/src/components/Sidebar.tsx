import { NavLink } from "react-router-dom";
import { Workflow, Play, Package, type LucideIcon } from "lucide-react";
import { Logo } from "./Logo";

const LINKS: { to: string; label: string; description: string; icon: LucideIcon; accent: string }[] = [
  {
    to: "/pipelines",
    label: "Pipelines",
    description: "Assembled & registered",
    icon: Workflow,
    accent: "indigo",
  },
  {
    to: "/runs",
    label: "Runs",
    description: "Executions & status",
    icon: Play,
    accent: "violet",
  },
  {
    to: "/artifacts",
    label: "Artifacts",
    description: "Search & inspect",
    icon: Package,
    accent: "amber",
  },
];

const ACCENT_CLASSES: Record<string, { badge: string; icon: string; activeBorder: string }> = {
  indigo: {
    badge: "bg-indigo-500/10 group-hover:bg-indigo-500/15",
    icon: "text-indigo-400",
    activeBorder: "border-indigo-500/40",
  },
  violet: {
    badge: "bg-violet-500/10 group-hover:bg-violet-500/15",
    icon: "text-violet-400",
    activeBorder: "border-violet-500/40",
  },
  amber: {
    badge: "bg-amber-500/10 group-hover:bg-amber-500/15",
    icon: "text-amber-400",
    activeBorder: "border-amber-500/40",
  },
};

export function Sidebar() {
  return (
    <aside className="w-80 shrink-0 border-r border-slate-800 bg-slate-950 p-6">
      <div className="mb-10 flex items-center gap-3 px-1">
        <Logo size={32} />
        <span className="text-xl font-semibold tracking-tight text-white">
          bettmensch<span className="text-sky-400">.ai</span>
        </span>
      </div>
      <nav className="space-y-3">
        {LINKS.map(({ to, label, description, icon: Icon, accent }) => {
          const colors = ACCENT_CLASSES[accent];
          return (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `group flex items-center gap-4 rounded-2xl border px-4 py-4 transition-colors ${
                  isActive
                    ? `bg-slate-900 ${colors.activeBorder}`
                    : "border-transparent hover:bg-slate-900/60"
                }`
              }
            >
              <div
                className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-xl ${colors.badge}`}
              >
                <Icon size={22} className={colors.icon} />
              </div>
              <div className="min-w-0">
                <div className="text-base font-semibold text-slate-100">{label}</div>
                <div className="truncate text-sm text-slate-500">{description}</div>
              </div>
            </NavLink>
          );
        })}
      </nav>
    </aside>
  );
}
