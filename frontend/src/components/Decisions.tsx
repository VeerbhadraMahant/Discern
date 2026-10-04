import type { Plan, SceneProfile } from "../api/types";

export function ProfileList({ profile }: { profile: SceneProfile }) {
  const rows: Array<[string, string]> = [
    ["Scene", profile.scene_label],
    ["Illumination", profile.illumination],
    ["Visibility", profile.visibility],
    ["Object scale", profile.object_scale],
    ["Object density", profile.object_density],
    ["Confidence", profile.confidence.toFixed(2)],
  ];
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1">
      {rows.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="font-semibold">{k}</dt>
          <dd className="num font-normal">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function Timeline({ plan, label = "Decision timeline" }: { plan: Plan; label?: string }) {
  return (
    <div className="flex flex-col gap-2">
      <p className="font-semibold">{label}</p>
      <p className="font-normal">
        Restorer: {plan.restorer}. {plan.use_restored ? "Restored image kept" : "Original kept"}.{" "}
        {plan.sr_factor ? `Super-resolution ${plan.sr_factor}x.` : "No super-resolution."}
      </p>
      <ol className="flex flex-col border-l-2 border-ink pl-4">
        {plan.decisions.map((d, i) => (
          <li key={i} className="py-1.5 font-normal">
            {d}
          </li>
        ))}
      </ol>
    </div>
  );
}
