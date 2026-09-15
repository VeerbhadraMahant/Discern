import { useState } from "react";
import type { AnalyzeResponse } from "../../lib/types";
import { Download, X, FileText, Check, Copy, Printer } from "lucide-react";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  analysis: AnalyzeResponse;
  scenarioTitle?: string;
}

export default function ExportReportModal({ isOpen, onClose, analysis, scenarioTitle }: Props) {
  const [copied, setCopied] = useState(false);

  if (!isOpen) return null;

  const jsonReport = JSON.stringify(
    {
      report_generated: new Date().toISOString(),
      system: "Discern Adaptive Vision Agent v2.4",
      scenario: scenarioTitle || "Custom Stream Capture",
      scene_understanding: analysis.scene,
      agent_tool_plan: analysis.plan,
      execution_trace: analysis.steps,
      detected_targets: analysis.detections,
      safety_violations: analysis.violations,
    },
    null,
    2
  );

  const handleCopy = () => {
    navigator.clipboard.writeText(jsonReport);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([jsonReport], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `discern-safety-audit-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm font-mono text-hermes-ink">
      <div className="relative w-full max-w-2xl border-2 border-hermes-blue bg-white p-6 shadow-terminal max-h-[90vh] flex flex-col">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1 text-hermes-muted hover:text-hermes-ink"
        >
          <X className="h-4 w-4" />
        </button>

        <div className="space-y-1 pb-3 border-b border-hermes-ink/15">
          <div className="flex items-center gap-2 text-xs">
            <FileText className="h-4 w-4 text-hermes-blue" />
            <span className="uppercase text-hermes-muted font-bold">
              // TELEMETRY_EXPORT
            </span>
          </div>
          <h3 className="hermes-title text-xl font-bold uppercase text-hermes-ink">
            Safety Audit Record
          </h3>
          <p className="text-xs text-hermes-charcoal font-body">
            Standardized OSHA/ISO incident record with raw scene context, OpenCV restoration metadata, and violation logs.
          </p>
        </div>

        {/* JSON Code Viewer */}
        <div className="mt-4 flex-1 overflow-auto bg-black p-4 text-[11px] text-white border border-hermes-ink/20">
          <pre className="whitespace-pre-wrap">{jsonReport}</pre>
        </div>

        {/* Action Buttons */}
        <div className="mt-5 flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-hermes-ink/15 text-xs">
          <button
            onClick={handleCopy}
            className="hermes-btn-ghost text-hermes-ink border-hermes-ink/30 hover:bg-hermes-ink/10 flex items-center gap-2"
          >
            {copied ? <Check className="h-3.5 w-3.5 text-hermes-safe" /> : <Copy className="h-3.5 w-3.5" />}
            <span>{copied ? "COPIED_TO_CLIPBOARD" : "COPY_JSON"}</span>
          </button>

          <div className="flex items-center gap-2">
            <button
              onClick={() => window.print()}
              className="px-3 py-2 border border-hermes-ink/30 text-hermes-ink hover:bg-hermes-ink/10 font-bold uppercase"
            >
              <Printer className="h-3.5 w-3.5" />
            </button>

            <button
              onClick={handleDownload}
              className="hermes-btn-primary bg-hermes-blue text-white hover:bg-hermes-dark flex items-center gap-2"
            >
              <Download className="h-3.5 w-3.5" />
              <span>Download (.JSON)</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
