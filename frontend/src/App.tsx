import { useState, useEffect } from "react";
import type { AppView, AnalyzeResponse } from "./lib/types";
import { SAMPLE_SCENARIOS } from "./lib/sampleScenarios";
import { analyzeFrame, urlToFile } from "./lib/api";

// Layout
import Navbar from "./components/layout/Navbar";
import Footer from "./components/layout/Footer";

// Landing Page Components
import LandingHero from "./components/landing/LandingHero";
import InteractiveDemo from "./components/landing/InteractiveDemo";
import PipelineShowcase from "./components/landing/PipelineShowcase";
import IndustryVerticals from "./components/landing/IndustryVerticals";
import TechnicalSpecs from "./components/landing/TechnicalSpecs";
import RoiCalculator from "./components/landing/RoiCalculator";
import CtaSection from "./components/landing/CtaSection";

// Studio Components
import StudioHeader from "./components/studio/StudioHeader";
import FrameWorkbench from "./components/studio/FrameWorkbench";
import SceneConditionCard from "./components/studio/SceneConditionCard";
import ToolsEngagedCard from "./components/studio/ToolsEngagedCard";
import ViolationsInspector from "./components/studio/ViolationsInspector";
import AgentPipelineDAG from "./components/studio/AgentPipelineDAG";
import UploadModal from "./components/studio/UploadModal";
import ExportReportModal from "./components/studio/ExportReportModal";

// Monitoring Components
import MultiCamGrid from "./components/monitoring/MultiCamGrid";
import IncidentAuditLog from "./components/monitoring/IncidentAuditLog";

import { AlertTriangle, Sparkles, Loader2 } from "lucide-react";

export default function App() {
  const [currentView, setCurrentView] = useState<AppView>("landing");
  const [selectedPresetId, setSelectedPresetId] = useState<string>("dawn-construction");
  const [customResult, setCustomResult] = useState<AnalyzeResponse | null>(null);
  const [customFileName, setCustomFileName] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  // Modals
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [isExportOpen, setIsExportOpen] = useState(false);

  // Target hover highlighting across frame & violation cards
  const [highlightedDetectionId, setHighlightedDetectionId] = useState<string | null>(null);

  // Active scenario preset
  const activePreset = SAMPLE_SCENARIOS.find((s) => s.id === selectedPresetId) || SAMPLE_SCENARIOS[0];

  // Active analysis is only ever a REAL backend result (from an upload or from
  // running the agent on a preset's sample photo) - null means this frame has
  // not been analyzed yet, so the UI shows the raw image with no fabricated
  // detections/violations rather than pre-written mock numbers.
  const activeAnalysis: AnalyzeResponse | null = customResult;
  const activeAnnotatedImage = customResult?.annotated_image || activePreset.photoUrl;
  const activeRawImage = customResult?.raw_image || activePreset.photoUrl;
  const activeRestoredImage = customResult?.restored_image || activePreset.photoUrl;
  // CSS filters simulate the raw/restored look on a single real photo for an
  // un-analyzed preset; a real backend result already has distinct images.
  const activeRawFilter = customResult ? "none" : activePreset.rawFilter;
  const activeRestoredFilter = customResult ? "none" : activePreset.restoredFilter;

  // Run the real agent pipeline (restoration -> YOLO -> Gemini reasoning) against
  // whichever bytes are given - a user upload or a sample scenario's photo.
  const runAnalysis = async (file: File, label: string) => {
    setIsAnalyzing(true);
    setAnalysisError(null);
    setCurrentView("studio");

    try {
      const res = await analyzeFrame(file);
      setCustomResult(res);
    } catch (err) {
      console.warn("Agent analysis failed:", err);
      setCustomResult(null);
      setAnalysisError(
        err instanceof Error
          ? `Analysis failed: ${err.message}`
          : "Analysis failed: the agent backend is unreachable."
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleUploadFile = async (file: File) => {
    setCustomFileName(file.name);
    await runAnalysis(file, file.name);
  };

  const handleAnalyzePreset = async () => {
    setCustomFileName(null);
    const file = await urlToFile(activePreset.photoUrl, `${activePreset.id}.jpg`);
    await runAnalysis(file, activePreset.title);
  };

  const handleSelectPreset = (presetId: string) => {
    setSelectedPresetId(presetId);
    setCustomResult(null);
    setCustomFileName(null);
    setAnalysisError(null);
  };

  const handleReset = () => {
    setCustomResult(null);
    setCustomFileName(null);
    setSelectedPresetId("dawn-construction");
    setAnalysisError(null);
  };

  return (
    <div className="min-h-screen bg-hermes-paper text-hermes-ink flex flex-col font-body antialiased selection:bg-hermes-blue selection:text-white">
      {/* Global Navigation Header (Hermes Cobalt) */}
      <Navbar
        currentView={currentView}
        onViewChange={(view) => {
          setCurrentView(view);
          window.scrollTo({ top: 0, behavior: "smooth" });
        }}
        onOpenUpload={() => setIsUploadOpen(true)}
      />

      {/* Main Content Areas based on View */}
      <main className="flex-1">
        {/* =========================================================================
            VIEW 1: PUBLIC LANDING PAGE (HERMES THEME)
           ========================================================================= */}
        {currentView === "landing" && (
          <div>
            <LandingHero
              onLaunch={() => {
                setCurrentView("studio");
                window.scrollTo({ top: 0, behavior: "smooth" });
              }}
              onSelectPreset={(presetId) => {
                handleSelectPreset(presetId);
                setCurrentView("studio");
                window.scrollTo({ top: 0, behavior: "smooth" });
              }}
            />
            <InteractiveDemo
              onSelectScenario={(presetId) => {
                handleSelectPreset(presetId);
                setCurrentView("studio");
                window.scrollTo({ top: 0, behavior: "smooth" });
              }}
            />
            <PipelineShowcase />
            <IndustryVerticals />
            <TechnicalSpecs />
            <RoiCalculator
              onLaunch={() => {
                setCurrentView("studio");
                window.scrollTo({ top: 0, behavior: "smooth" });
              }}
            />
            <CtaSection
              onLaunch={() => {
                setCurrentView("studio");
                window.scrollTo({ top: 0, behavior: "smooth" });
              }}
            />
          </div>
        )}

        {/* =========================================================================
            VIEW 2: LIVE MONITORING STUDIO / WORKBENCH
           ========================================================================= */}
        {currentView === "studio" && (
          <div className="mx-auto max-w-7xl px-4 py-8 sm:px-8 space-y-6">
            {/* Header & Preset Switcher */}
            <StudioHeader
              selectedPresetId={selectedPresetId}
              onSelectPreset={handleSelectPreset}
              onOpenUpload={() => setIsUploadOpen(true)}
              onOpenExport={() => setIsExportOpen(true)}
              onReset={handleReset}
              customFileName={customFileName}
              canExport={!!activeAnalysis}
            />

            {/* Error / Offline Toast Banner if any */}
            {analysisError && (
              <div className="flex items-center justify-between gap-3 bg-hermes-blue/10 p-4 border border-hermes-blue text-xs text-hermes-ink font-mono">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-hermes-blue shrink-0" />
                  <span>{analysisError}</span>
                </div>
                <button
                  onClick={() => setAnalysisError(null)}
                  className="text-xs text-hermes-muted hover:text-hermes-ink font-bold uppercase"
                >
                  [DISMISS]
                </button>
              </div>
            )}

            {/* Processing Loading Overlay */}
            {isAnalyzing ? (
              <div className="flex flex-col items-center justify-center p-20 bg-white border border-hermes-ink/20 text-center space-y-4 font-mono shadow-terminal">
                <Loader2 className="h-10 w-10 text-hermes-blue animate-spin" />
                <div>
                  <h3 className="hermes-title text-lg font-bold text-hermes-ink uppercase">
                    Agent Inspecting Frame...
                  </h3>
                  <p className="text-xs text-hermes-charcoal mt-1 max-w-md font-body">
                    Classifying atmospheric visibility &bull; Applying OpenCV restoration &bull; Running YOLOv8n localization &bull; Synthesizing OSHA safety violations.
                  </p>
                </div>
              </div>
            ) : (
              /* Main Split Grid */
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
                {/* Left Column: CCTV Frame Viewer & Trace (7 Cols) */}
                <div className="lg:col-span-7 space-y-6">
                  <FrameWorkbench
                    annotatedImage={activeAnnotatedImage}
                    rawImage={activeRawImage}
                    restoredImage={activeRestoredImage}
                    rawFilter={activeRawFilter}
                    restoredFilter={activeRestoredFilter}
                    detections={activeAnalysis?.detections ?? []}
                    violations={activeAnalysis?.violations ?? []}
                    cameraName={customFileName || activePreset.camera}
                    conditionLabel={activePreset.condition}
                    highlightedDetectionId={highlightedDetectionId}
                    onHoverDetection={setHighlightedDetectionId}
                  />

                  {/* Latency & Step DAG below the video */}
                  {activeAnalysis && <AgentPipelineDAG steps={activeAnalysis.steps} />}
                </div>

                {/* Right Column: Telemetry, Violations, and Tools (5 Cols) */}
                <div className="lg:col-span-5 space-y-6">
                  {activeAnalysis ? (
                    <>
                      {/* Scene Understanding Card */}
                      <SceneConditionCard scene={activeAnalysis.scene} />

                      {/* Violations Inspector with direct highlight trigger */}
                      <ViolationsInspector
                        violations={activeAnalysis.violations}
                        highlightedDetectionId={highlightedDetectionId}
                        onSelectViolation={setHighlightedDetectionId}
                      />

                      {/* Tools Engaged Card */}
                      <ToolsEngagedCard plan={activeAnalysis.plan} />
                    </>
                  ) : (
                    <div className="border border-hermes-ink/15 bg-white p-6 shadow-lift space-y-4 font-mono text-center">
                      <Sparkles className="h-8 w-8 text-hermes-blue mx-auto" />
                      <div>
                        <h3 className="hermes-title text-lg font-bold text-hermes-ink uppercase">
                          Frame Not Yet Analyzed
                        </h3>
                        <p className="text-xs text-hermes-charcoal mt-1 font-body leading-relaxed">
                          This is the raw sample photo for &ldquo;{activePreset.title}&rdquo;. Run the live agent to
                          restore the frame, localize people with YOLOv8n, and reason about safety violations.
                        </p>
                      </div>
                      <button
                        onClick={handleAnalyzePreset}
                        className="hermes-btn-primary bg-hermes-blue text-white hover:bg-hermes-dark text-xs w-full flex items-center justify-center gap-2"
                      >
                        <Sparkles className="h-3.5 w-3.5" />
                        <span>Run Agent Analysis</span>
                      </button>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* =========================================================================
            VIEW 3: MULTI-CAMERA CONTROL MATRIX
           ========================================================================= */}
        {currentView === "multicam" && (
          <div className="mx-auto max-w-7xl px-4 py-8 sm:px-8">
            <MultiCamGrid
              onSelectCameraPreset={(presetId) => {
                handleSelectPreset(presetId);
                setCurrentView("studio");
                window.scrollTo({ top: 0, behavior: "smooth" });
              }}
            />
          </div>
        )}

        {/* =========================================================================
            VIEW 4: INCIDENT AUDIT LEDGER
           ========================================================================= */}
        {currentView === "audit" && (
          <div className="mx-auto max-w-7xl px-4 py-8 sm:px-8">
            <IncidentAuditLog
              onSelectAuditPreset={(presetId) => {
                handleSelectPreset(presetId);
                setCurrentView("studio");
                window.scrollTo({ top: 0, behavior: "smooth" });
              }}
            />
          </div>
        )}
      </main>

      {/* Global Modals */}
      <UploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onUploadFile={handleUploadFile}
        isLoading={isAnalyzing}
      />

      {activeAnalysis && (
        <ExportReportModal
          isOpen={isExportOpen}
          onClose={() => setIsExportOpen(false)}
          analysis={activeAnalysis}
          scenarioTitle={customFileName || activePreset.title}
        />
      )}

      {/* Global Footer (Hermes Cobalt) */}
      <Footer
        onLaunch={() => {
          setCurrentView("studio");
          window.scrollTo({ top: 0, behavior: "smooth" });
        }}
      />
    </div>
  );
}
