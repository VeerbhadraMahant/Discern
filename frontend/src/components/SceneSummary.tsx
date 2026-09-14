import type { SceneContext } from "../lib/types";
import { EyeIcon, CloudFogIcon } from "./icons";

const LIGHTING_LABEL: Record<string, string> = {
  bright: "Bright",
  normal: "Normal light",
  low_light: "Low light",
  night: "Night",
};

const VISIBILITY_TONE: Record<string, string> = {
  good: "bg-safe/10 text-safe",
  reduced: "bg-warning/10 text-warning",
  poor: "bg-critical/10 text-critical",
};

export default function SceneSummary({ scene }: { scene: SceneContext }) {
  return (
    <div className="rounded-card bg-paper p-5 shadow-subtle">
      <h2 className="font-heading text-sm font-semibold uppercase tracking-wide text-charcoal">Scene conditions</h2>
      <p className="mt-2 text-sm leading-snug text-ink">{scene.summary}</p>

      <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
        <div className="rounded-input bg-cloud px-3 py-2">
          <dt className="text-xs text-slate">Setting</dt>
          <dd className="mt-0.5 font-medium capitalize text-ink">{scene.setting}</dd>
        </div>
        <div className="rounded-input bg-cloud px-3 py-2">
          <dt className="text-xs text-slate">Lighting</dt>
          <dd className="mt-0.5 font-medium text-ink">{LIGHTING_LABEL[scene.lighting] ?? scene.lighting}</dd>
        </div>
        <div className="rounded-input bg-cloud px-3 py-2">
          <dt className="flex items-center gap-1 text-xs text-slate">
            <CloudFogIcon width={12} height={12} /> Weather
          </dt>
          <dd className="mt-0.5 font-medium capitalize text-ink">{scene.weather}</dd>
        </div>
        <div className={`rounded-input px-3 py-2 ${VISIBILITY_TONE[scene.visibility] ?? "bg-cloud"}`}>
          <dt className="flex items-center gap-1 text-xs opacity-80">
            <EyeIcon width={12} height={12} /> Visibility
          </dt>
          <dd className="mt-0.5 font-medium capitalize">{scene.visibility}</dd>
        </div>
      </dl>

      <div className="mt-4 flex flex-wrap gap-2 text-xs">
        <span className={`rounded-pill px-3 py-1 font-medium ${scene.human_presence ? "bg-accent/10 text-accent" : "bg-cloud text-slate"}`}>
          {scene.human_presence ? "People present" : "No people expected"}
        </span>
        <span className={`rounded-pill px-3 py-1 font-medium ${scene.machinery_presence ? "bg-accent/10 text-accent" : "bg-cloud text-slate"}`}>
          {scene.machinery_presence ? "Machinery present" : "No machinery"}
        </span>
      </div>
    </div>
  );
}
