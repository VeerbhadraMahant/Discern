import { useState } from "react";
import type { Violation } from "../../lib/types";
import { 
  AlertTriangle, 
  ShieldAlert, 
  HardHat, 
  CheckCircle2, 
  Radio, 
  ChevronRight,
  ShieldCheck
} from "lucide-react";

interface Props {
  violations: Violation[];
  highlightedDetectionId: string | null;
  onSelectViolation: (detectionId: string | null) => void;
}

export default function ViolationsInspector({
  violations,
  highlightedDetectionId,
  onSelectViolation,
}: Props) {
  const [acknowledgedIds, setAcknowledgedIds] = useState<Set<string>>(new Set());

  const toggleAcknowledge = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setAcknowledgedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <div className="border border-hermes-ink/15 bg-white p-5 shadow-lift space-y-4 font-mono">
      <div className="flex items-center justify-between pb-2 border-b border-hermes-ink/10">
        <div className="flex items-center gap-2">
          <ShieldAlert className="h-4 w-4 text-hermes-critical" />
          <h2 className="text-xs font-bold uppercase tracking-wider text-hermes-ink">
            # VIOLATION_LOGS
          </h2>
        </div>
        <span
          className={`px-2 py-0.5 text-[11px] uppercase font-bold ${
            violations.length > 0 ? "bg-hermes-critical text-white" : "bg-hermes-safe text-white"
          }`}
        >
          {violations.length} Active Infraction{violations.length === 1 ? "" : "s"}
        </span>
      </div>

      {violations.length === 0 ? (
        <div className="border border-hermes-safe/30 bg-hermes-safe/5 p-6 text-center">
          <CheckCircle2 className="h-7 w-7 text-hermes-safe mx-auto mb-2" />
          <p className="font-bold text-hermes-ink text-xs uppercase">Site Compliance Verified</p>
          <p className="text-xs text-hermes-charcoal mt-1 font-body">
            All ground personnel are clear of exclusion perimeters and compliant with PPE regulations.
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {violations.map((v) => {
            const isAcked = acknowledgedIds.has(v.id);
            const isTargetHighlighted = v.related_detection_ids.some(
              (did) => did === highlightedDetectionId
            );

            return (
              <div
                key={v.id}
                onClick={() => {
                  const targetId = v.related_detection_ids[0] || null;
                  onSelectViolation(isTargetHighlighted ? null : targetId);
                }}
                className={`border p-4 transition-all cursor-pointer ${
                  isTargetHighlighted
                    ? "border-hermes-blue bg-hermes-blue/10 shadow-hermes"
                    : "border-hermes-ink/20 bg-hermes-paper hover:border-hermes-blue"
                }`}
              >
                {/* Header */}
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs font-bold uppercase text-hermes-critical">
                    :{v.type.toUpperCase()}:
                  </span>
                  <div className="flex items-center gap-1.5">
                    {v.oshaCode && (
                      <span className="bg-white border border-hermes-ink/20 px-1.5 py-0.5 text-[11px] text-hermes-muted">
                        {v.oshaCode}
                      </span>
                    )}
                    <span className="bg-hermes-critical text-white px-2 py-0.5 text-[11px] font-bold uppercase">
                      {v.severity}
                    </span>
                  </div>
                </div>

                {/* Description */}
                <p className="mt-2 text-xs text-hermes-ink font-body font-medium leading-relaxed">
                  {v.description}
                </p>

                {/* Recommendation */}
                {v.recommendation && (
                  <div className="mt-2 bg-white p-2.5 border border-hermes-ink/10 text-xs">
                    <span className="text-[11px] font-bold uppercase text-hermes-blue block mb-0.5">
                      ACTION_PROTOCOL:
                    </span>
                    <p className="text-hermes-charcoal font-body leading-snug">{v.recommendation}</p>
                  </div>
                )}

                {/* Footer */}
                <div className="mt-3 pt-2.5 border-t border-hermes-ink/10 flex items-center justify-between text-xs">
                  <span className="text-[11px] text-hermes-muted">
                    TARGET: {v.related_detection_ids.join(", ") || "ZONE"}
                  </span>

                  <button
                    onClick={(e) => toggleAcknowledge(v.id, e)}
                    className={`px-3 py-1 text-[11px] font-bold uppercase transition-all ${
                      isAcked
                        ? "bg-hermes-safe text-white"
                        : "bg-hermes-blue text-white hover:bg-hermes-dark"
                    }`}
                  >
                    {isAcked ? "ACKNOWLEDGED" : "DISPATCH_ALERT"}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
