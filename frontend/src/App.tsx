import { useState } from "react";
import UploadPanel from "./components/UploadPanel";
import PipelineTrace from "./components/PipelineTrace";
import SceneSummary from "./components/SceneSummary";
import ViolationsList from "./components/ViolationsList";
import AnnotatedFrame from "./components/AnnotatedFrame";
import ToolsUsed from "./components/ToolsUsed";
import { SpinnerIcon, AlertTriangleIcon } from "./components/icons";
import { analyzeFrame } from "./lib/api";
import type { AnalyzeResponse } from "./lib/types";

type Status = "idle" | "loading" | "error";

export default function App() {
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [previewName, setPreviewName] = useState<string | null>(null);

  const handleSelect = async (file: File) => {
    setStatus("loading");
    setError(null);
    setPreviewName(file.name);
    try {
      const res = await analyzeFrame(file);
      setResult(res);
      setStatus("idle");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed");
      setStatus("error");
    }
  };

  return (
    <div className="min-h-dvh bg-cloud">
      <header className="border-b border-slate/15 bg-paper">
        <div className="mx-auto flex max-w-7xl items-center gap-3 px-4 py-4 sm:px-6">
          <div className="flex h-9 w-9 items-center justify-center rounded-icon bg-obsidian font-heading text-sm font-bold text-paper">
            D
          </div>
          <div>
            <h1 className="font-heading text-lg font-semibold leading-none text-ink">Discern</h1>
            <p className="text-xs text-charcoal">Adaptive vision agent for high-risk site monitoring</p>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
        {!result && status !== "loading" && (
          <div className="mx-auto max-w-2xl">
            <UploadPanel onSelect={handleSelect} />
            {status === "error" && error && (
              <div className="mt-4 flex items-center gap-2 rounded-input border border-critical/20 bg-critical/10 px-4 py-3 text-sm font-medium text-critical">
                <AlertTriangleIcon width={18} height={18} />
                {error}
              </div>
            )}
          </div>
        )}

        {status === "loading" && (
          <div className="mx-auto flex max-w-2xl flex-col items-center gap-3 rounded-card-lg bg-paper p-16 text-center shadow-subtle">
            <SpinnerIcon width={28} height={28} className="text-accent" />
            <p className="font-heading text-lg font-semibold text-ink">Analyzing {previewName}</p>
            <p className="text-sm text-charcoal">Agent is classifying the scene, selecting tools, and reasoning about violations…</p>
          </div>
        )}

        {result && status === "idle" && (
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_360px]">
            <div className="space-y-6">
              <AnnotatedFrame src={result.annotated_image} />
              <button
                onClick={() => {
                  setResult(null);
                  setError(null);
                }}
                className="rounded-pill bg-obsidian px-5 py-2.5 text-sm font-medium text-paper shadow-pill transition-opacity hover:opacity-90"
              >
                Analyze another frame
              </button>
            </div>
            <div className="space-y-6">
              <SceneSummary scene={result.scene} />
              <ViolationsList violations={result.violations} />
              <ToolsUsed plan={result.plan} />
              <PipelineTrace steps={result.steps} />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
