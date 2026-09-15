export interface SceneContext {
  setting: string;
  lighting: "bright" | "normal" | "low_light" | "night";
  weather: "clear" | "fog" | "rain" | "dust" | "glare" | "unknown";
  visibility: "good" | "reduced" | "poor";
  human_presence: boolean;
  machinery_presence: boolean;
  summary: string;
}

export interface ToolPlan {
  restoration_tools: string[];
  detection_tools: string[];
  reasoning: string;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Detection {
  id: string;
  label: string;
  confidence: number;
  box: BoundingBox;
  source: string;
}

export interface Violation {
  id: string;
  type: "missing_ppe" | "restricted_zone" | "unsafe_proximity";
  severity: "critical" | "warning" | "info";
  description: string;
  related_detection_ids: string[];
  oshaCode?: string;
  recommendation?: string;
}

export interface PipelineStep {
  name: string;
  status: "skipped" | "completed";
  detail: string;
  duration_ms: number;
}

export interface AnalyzeResponse {
  scene: SceneContext;
  plan: ToolPlan;
  steps: PipelineStep[];
  detections: Detection[];
  violations: Violation[];
  annotated_image: string;
  raw_image?: string;
  restored_image?: string;
}

export interface ScenarioPreset {
  id: string;
  title: string;
  site: string;
  camera: string;
  condition: string;
  tag: string;
  description: string;
  /** Real photograph used as the CCTV frame for this scenario. */
  photoUrl: string;
  photoCredit: string;
  /** CSS filter simulating the degraded raw sensor read on top of photoUrl. */
  rawFilter: string;
  /** CSS filter simulating the agent's classical-CV restoration on top of photoUrl. */
  restoredFilter: string;
  analysis: AnalyzeResponse;
}

export interface CameraFeed {
  id: string;
  name: string;
  location: string;
  zone: string;
  status: "active" | "alert" | "degraded";
  fps: number;
  resolution: string;
  lux: number;
  weatherCondition: string;
  violationsCount: number;
  lastChecked: string;
  presetId: string;
}

export interface AuditRecord {
  id: string;
  timestamp: string;
  cameraId: string;
  cameraName: string;
  zone: string;
  weather: string;
  lighting: string;
  violationType: string;
  severity: "critical" | "warning" | "info";
  description: string;
  restorationApplied: string;
  status: "open" | "acknowledged" | "resolved";
}

export type AppView = "landing" | "studio" | "multicam" | "audit";
