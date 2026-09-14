import type { ToolPlan } from "../lib/types";

const LABELS: Record<string, string> = {
  low_light_enhancement: "Low-light enhancement",
  dehaze: "Dehaze",
  denoise: "Denoise",
  person_detection: "Person detection",
  ppe_reasoning: "PPE compliance check",
  zone_reasoning: "Restricted-zone check",
};

export default function ToolsUsed({ plan }: { plan: ToolPlan }) {
  const tools = [...plan.restoration_tools, ...plan.detection_tools];
  if (tools.length === 0) return null;

  return (
    <div className="rounded-card bg-paper p-5 shadow-subtle">
      <h2 className="font-heading text-sm font-semibold uppercase tracking-wide text-charcoal">Tools engaged</h2>
      <div className="mt-3 flex flex-wrap gap-2">
        {tools.map((t) => (
          <span key={t} className="rounded-pill bg-obsidian px-3 py-1.5 text-xs font-medium text-paper">
            {LABELS[t] ?? t}
          </span>
        ))}
      </div>
    </div>
  );
}
