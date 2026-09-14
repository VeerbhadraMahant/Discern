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
}
