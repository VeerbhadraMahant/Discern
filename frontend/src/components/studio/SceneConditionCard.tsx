import type { SceneContext } from "../../lib/types";
import { CloudFog, Eye, Sun, Moon, Users, HardHat } from "lucide-react";

interface Props {
  scene: SceneContext;
}

export default function SceneConditionCard({ scene }: Props) {
  return (
    <div className="border border-hermes-ink/15 bg-white p-5 shadow-lift space-y-4 font-mono">
      <div className="flex items-center justify-between pb-2 border-b border-hermes-ink/10">
        <div className="flex items-center gap-2">
          <span className="flex h-2 w-2 bg-hermes-blue" />
          <h2 className="text-xs font-bold uppercase tracking-wider text-hermes-ink">
            # SCENE_UNDERSTANDING
          </h2>
        </div>
        <span className="bg-hermes-blue text-white px-2 py-0.5 text-[11px] uppercase font-bold">
          Gemini Vision
        </span>
      </div>

      <p className="text-xs font-medium text-hermes-ink font-body leading-relaxed">
        {scene.summary}
      </p>

      {/* Grid of parameters */}
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="bg-hermes-paper p-2.5 border border-hermes-ink/10">
          <span className="text-[11px] text-hermes-muted uppercase block">
            // SETTING
          </span>
          <span className="font-bold text-hermes-ink capitalize mt-0.5 block truncate">
            {scene.setting}
          </span>
        </div>

        <div className="bg-hermes-paper p-2.5 border border-hermes-ink/10">
          <span className="text-[11px] text-hermes-muted uppercase block">
            // LIGHTING
          </span>
          <span className="font-bold text-hermes-ink capitalize mt-0.5 block truncate">
            {scene.lighting}
          </span>
        </div>

        <div className="bg-hermes-paper p-2.5 border border-hermes-ink/10">
          <span className="text-[11px] text-hermes-muted uppercase block">
            // WEATHER
          </span>
          <span className="font-bold text-hermes-ink capitalize mt-0.5 block truncate">
            {scene.weather}
          </span>
        </div>

        <div className="bg-hermes-paper p-2.5 border border-hermes-ink/10">
          <span className="text-[11px] text-hermes-muted uppercase block">
            // VISIBILITY
          </span>
          <span className="font-bold text-hermes-blue capitalize mt-0.5 block truncate">
            {scene.visibility}
          </span>
        </div>
      </div>

      {/* Presence Flags */}
      <div className="flex flex-wrap gap-2 pt-1 text-xs">
        <span
          className={`px-2.5 py-1 border font-bold uppercase ${
            scene.human_presence
              ? "bg-hermes-blue/10 border-hermes-blue text-hermes-blue"
              : "bg-hermes-paper border-hermes-ink/10 text-hermes-muted"
          }`}
        >
          {scene.human_presence ? ":WORKERS_DETECTED:" : ":NO_WORKERS:"}
        </span>

        <span
          className={`px-2.5 py-1 border font-bold uppercase ${
            scene.machinery_presence
              ? "bg-hermes-warning/10 border-hermes-warning text-hermes-warning"
              : "bg-hermes-paper border-hermes-ink/10 text-hermes-muted"
          }`}
        >
          {scene.machinery_presence ? ":MACHINERY_ACTIVE:" : ":NO_MACHINERY:"}
        </span>
      </div>
    </div>
  );
}
