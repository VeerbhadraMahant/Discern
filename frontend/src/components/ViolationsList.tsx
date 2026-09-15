import type { Violation } from "../lib/types";
import { AlertTriangleIcon, HardHatIcon, ZoneIcon, CheckCircleIcon } from "./icons";

const TYPE_ICON: Record<string, typeof HardHatIcon> = {
  missing_ppe: HardHatIcon,
  restricted_zone: ZoneIcon,
  unsafe_proximity: AlertTriangleIcon,
};

const SEVERITY_STYLE: Record<string, string> = {
  critical: "bg-critical/10 text-critical border-critical/20",
  warning: "bg-warning/10 text-warning border-warning/20",
  info: "bg-slate/10 text-charcoal border-slate/20",
};

const TYPE_LABEL: Record<string, string> = {
  missing_ppe: "Missing PPE",
  restricted_zone: "Restricted zone entry",
  unsafe_proximity: "Unsafe proximity",
};

export default function ViolationsList({ violations }: { violations: Violation[] }) {
  return (
    <div className="rounded-card bg-paper p-5 shadow-subtle">
      <h2 className="font-heading text-sm font-semibold uppercase tracking-wide text-charcoal">
        Violations {violations.length > 0 && <span className="text-critical">({violations.length})</span>}
      </h2>

      {violations.length === 0 ? (
        <div className="mt-4 flex items-center gap-2 rounded-input bg-safe/10 px-3 py-3 text-sm font-medium text-safe">
          <CheckCircleIcon width={18} height={18} />
          No violations detected
        </div>
      ) : (
        <ul className="mt-4 space-y-2">
          {violations.map((v) => {
            const Icon = TYPE_ICON[v.type] ?? AlertTriangleIcon;
            return (
              <li key={v.id} className={`rounded-input border px-3 py-3 ${SEVERITY_STYLE[v.severity]}`}>
                <div className="flex items-center gap-2">
                  <Icon width={16} height={16} />
                  <span className="text-xs font-semibold uppercase tracking-wide">{TYPE_LABEL[v.type] ?? v.type}</span>
                  <span className="ml-auto rounded-pill bg-obsidian/90 px-2 py-0.5 text-[11px] font-semibold uppercase text-paper">
                    {v.severity}
                  </span>
                </div>
                <p className="mt-1.5 text-sm leading-snug text-ink">{v.description}</p>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
