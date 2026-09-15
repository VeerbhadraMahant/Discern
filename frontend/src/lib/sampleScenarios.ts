import type { ScenarioPreset, CameraFeed, AuditRecord } from "./types";

// Helper function to create realistic procedural CCTV frame SVG Data URLs
function makeCctvSvg(
  scenario: "dawn" | "fog" | "dust" | "glare" | "clear",
  mode: "raw" | "restored" | "annotated"
): string {
  const width = 1280;
  const height = 720;

  // Colors and atmosphere based on condition and restoration
  let bgGradient = "";
  let overlayEffects = "";
  let person1Visible = true;
  let person2Visible = true;
  let machineVisible = true;

  if (scenario === "dawn") {
    if (mode === "raw") {
      bgGradient = `
        <defs>
          <linearGradient id="skyDawn" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#0a1128" />
            <stop offset="40%" stop-color="#1c1936" />
            <stop offset="80%" stop-color="#2a1b24" />
            <stop offset="100%" stop-color="#181318" />
          </linearGradient>
        </defs>
        <rect width="100%" height="100%" fill="url(#skyDawn)" />
        <rect width="100%" height="100%" fill="#000000" opacity="0.65" />
      `;
    } else {
      // CLAHE + Gamma enhanced
      bgGradient = `
        <defs>
          <linearGradient id="skyDawnEnh" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#1e2a4a" />
            <stop offset="40%" stop-color="#3b3760" />
            <stop offset="80%" stop-color="#60404a" />
            <stop offset="100%" stop-color="#352e35" />
          </linearGradient>
        </defs>
        <rect width="100%" height="100%" fill="url(#skyDawnEnh)" />
      `;
    }
  } else if (scenario === "fog") {
    if (mode === "raw") {
      bgGradient = `
        <defs>
          <linearGradient id="fogRaw" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#bdc3c7" />
            <stop offset="100%" stop-color="#95a5a6" />
          </linearGradient>
        </defs>
        <rect width="100%" height="100%" fill="url(#fogRaw)" />
        <rect width="100%" height="100%" fill="#ffffff" opacity="0.75" />
      `;
    } else {
      // Dehazed (Dark channel prior)
      bgGradient = `
        <defs>
          <linearGradient id="fogDehazed" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#4a6572" />
            <stop offset="60%" stop-color="#34495e" />
            <stop offset="100%" stop-color="#2c3e50" />
          </linearGradient>
        </defs>
        <rect width="100%" height="100%" fill="url(#fogDehazed)" />
      `;
    }
  } else if (scenario === "dust") {
    if (mode === "raw") {
      bgGradient = `
        <defs>
          <linearGradient id="dustRaw" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#b08d57" />
            <stop offset="100%" stop-color="#7a5229" />
          </linearGradient>
        </defs>
        <rect width="100%" height="100%" fill="url(#dustRaw)" />
        <rect width="100%" height="100%" fill="#d4a373" opacity="0.65" />
      `;
    } else {
      // Dehazed + Denoised
      bgGradient = `
        <defs>
          <linearGradient id="dustDehazed" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#6b5b45" />
            <stop offset="100%" stop-color="#3d2b1f" />
          </linearGradient>
        </defs>
        <rect width="100%" height="100%" fill="url(#dustDehazed)" />
      `;
    }
  } else if (scenario === "glare") {
    if (mode === "raw") {
      bgGradient = `
        <defs>
          <radialGradient id="glareRaw" cx="70%" cy="30%" r="60%">
            <stop offset="0%" stop-color="#ffffff" />
            <stop offset="30%" stop-color="#fff8e7" />
            <stop offset="70%" stop-color="#94a3b8" />
            <stop offset="100%" stop-color="#475569" />
          </radialGradient>
        </defs>
        <rect width="100%" height="100%" fill="url(#glareRaw)" />
      `;
    } else {
      // Dynamic range compensated
      bgGradient = `
        <defs>
          <radialGradient id="glareComp" cx="70%" cy="30%" r="60%">
            <stop offset="0%" stop-color="#e2e8f0" />
            <stop offset="50%" stop-color="#64748b" />
            <stop offset="100%" stop-color="#1e293b" />
          </radialGradient>
        </defs>
        <rect width="100%" height="100%" fill="url(#glareComp)" />
      `;
    }
  } else {
    // Clear daytime
    bgGradient = `
      <defs>
        <linearGradient id="clearSky" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#60a5fa" />
          <stop offset="60%" stop-color="#93c5fd" />
          <stop offset="100%" stop-color="#e2e8f0" />
        </linearGradient>
      </defs>
      <rect width="100%" height="100%" fill="url(#clearSky)" />
    `;
  }

  // Site Scenery Elements (Ground, Scaffolding, Crane / Excavator, Barrier)
  const scenery = `
    <!-- Ground plane -->
    <polygon points="0,520 1280,500 1280,720 0,720" fill="#2d3748" opacity="0.9" />
    <polygon points="0,550 1280,530 1280,720 0,720" fill="#1a202c" />

    <!-- Industrial Structure / Scaffolding -->
    <line x1="120" y1="180" x2="120" y2="540" stroke="#718096" stroke-width="6" />
    <line x1="280" y1="180" x2="280" y2="530" stroke="#718096" stroke-width="6" />
    <line x1="120" y1="260" x2="280" y2="260" stroke="#718096" stroke-width="4" />
    <line x1="120" y1="380" x2="280" y2="380" stroke="#718096" stroke-width="4" />
    <line x1="120" y1="260" x2="280" y2="380" stroke="#4a5568" stroke-width="3" />
    <line x1="120" y1="380" x2="280" y2="260" stroke="#4a5568" stroke-width="3" />

    <!-- Heavy Machinery (Excavator / Gantry) -->
    <g id="heavyEquipment" transform="translate(680, 290)">
      <!-- Track base -->
      <rect x="0" y="160" width="220" height="40" rx="10" fill="#2d3748" />
      <circle cx="30" cy="180" r="14" fill="#1a202c" />
      <circle cx="75" cy="180" r="14" fill="#1a202c" />
      <circle cx="120" cy="180" r="14" fill="#1a202c" />
      <circle cx="165" cy="180" r="14" fill="#1a202c" />
      <circle cx="200" cy="180" r="10" fill="#1a202c" />
      
      <!-- Cabin -->
      <rect x="40" y="90" width="90" height="75" rx="6" fill="#d97706" />
      <rect x="50" y="100" width="45" height="35" rx="3" fill="#93c5fd" opacity="0.6" />
      <!-- Counterweight -->
      <rect x="130" y="100" width="45" height="65" rx="4" fill="#b45309" />
      <!-- Boom Arm -->
      <polygon points="40,110 -140,20 -130,5 50,95" fill="#d97706" />
      <!-- Hydraulic Arm -->
      <polygon points="-140,20 -190,130 -175,135 -130,25" fill="#b45309" />
      <!-- Bucket -->
      <path d="M-195,130 L-220,165 L-170,175 L-165,140 Z" fill="#4b5563" />
    </g>

    <!-- Danger Zone Markings on Ground -->
    <polygon points="480,480 920,470 980,660 420,670" fill="#ef4444" opacity="0.15" stroke="#ef4444" stroke-width="2" stroke-dasharray="10,6" />
    <text x="560" y="630" fill="#f87171" font-family="sans-serif" font-size="14" font-weight="700" letter-spacing="2">DANGER: 15m ROTATION RADIUS</text>
  `;

  // Workers representation
  const workers = `
    <!-- Worker 1 (Inside Danger Zone, No Helmet) -->
    <g id="worker1" transform="translate(540, 390)">
      <!-- Head / Hair without helmet -->
      <circle cx="25" cy="20" r="12" fill="#4a3728" />
      <!-- Face -->
      <circle cx="25" cy="23" r="9" fill="#fbcfe8" />
      <!-- Torso / Vest -->
      <rect x="12" y="32" width="26" height="42" rx="4" fill="#f97316" />
      <line x1="16" y1="36" x2="16" y2="70" stroke="#ffffff" stroke-width="3" />
      <line x1="34" y1="36" x2="34" y2="70" stroke="#ffffff" stroke-width="3" />
      <!-- Legs -->
      <rect x="14" y="74" width="9" height="48" fill="#1e3a8a" />
      <rect x="27" y="74" width="9" height="48" fill="#1e3a8a" />
      <!-- Boots -->
      <rect x="12" y="118" width="13" height="8" rx="2" fill="#78350f" />
      <rect x="27" y="118" width="13" height="8" rx="2" fill="#78350f" />
    </g>

    <!-- Worker 2 (Safe Zone, Fully Compliant) -->
    <g id="worker2" transform="translate(230, 420)">
      <!-- Hard Hat (Yellow Helmet) -->
      <ellipse cx="25" cy="18" rx="14" ry="9" fill="#facc15" stroke="#ca8a04" stroke-width="1" />
      <!-- Face -->
      <circle cx="25" cy="23" r="8" fill="#fed7aa" />
      <!-- Torso / Hi-Vis Yellow Vest -->
      <rect x="12" y="32" width="26" height="40" rx="4" fill="#84cc16" />
      <line x1="16" y1="34" x2="16" y2="68" stroke="#ffffff" stroke-width="3" />
      <line x1="34" y1="34" x2="34" y2="68" stroke="#ffffff" stroke-width="3" />
      <!-- Legs -->
      <rect x="14" y="72" width="9" height="44" fill="#334155" />
      <rect x="27" y="72" width="9" height="44" fill="#334155" />
      <!-- Boots -->
      <rect x="12" y="112" width="13" height="8" rx="2" fill="#1e293b" />
      <rect x="27" y="112" width="13" height="8" rx="2" fill="#1e293b" />
    </g>
  `;

  // Overlays & Annotations
  let annotations = "";
  if (mode === "annotated") {
    annotations = `
      <!-- BBox 1: Worker 1 (Critical Violation) -->
      <g>
        <rect x="530" y="380" width="70" height="155" fill="none" stroke="#ef4444" stroke-width="3" rx="4" />
        <rect x="530" y="352" width="165" height="26" fill="#ef4444" rx="4" />
        <text x="538" y="370" fill="#ffffff" font-family="Inter, sans-serif" font-size="12" font-weight="700">PERSON 1 [94%] · CRITICAL</text>
        
        <!-- Callout pointer -->
        <circle cx="565" cy="405" r="16" fill="none" stroke="#ef4444" stroke-width="2" stroke-dasharray="3,3" />
        <line x1="600" y1="400" x2="650" y2="370" stroke="#ef4444" stroke-width="2" />
        <rect x="650" y="355" width="170" height="24" fill="#1e293b" rx="4" stroke="#ef4444" stroke-width="1" />
        <text x="658" y="371" fill="#fca5a5" font-family="Inter, sans-serif" font-size="11" font-weight="600">! Missing Hard Hat</text>
      </g>

      <!-- BBox 2: Worker 2 (Compliant Safe) -->
      <g>
        <rect x="220" y="410" width="68" height="148" fill="none" stroke="#10b981" stroke-width="2" rx="4" />
        <rect x="220" y="386" width="150" height="22" fill="#10b981" rx="4" />
        <text x="228" y="402" fill="#ffffff" font-family="Inter, sans-serif" font-size="11" font-weight="700">PERSON 2 [98%] · COMPLIANT</text>
      </g>

      <!-- BBox 3: Machinery BBox -->
      <g>
        <rect x="470" y="270" width="460" height="270" fill="none" stroke="#f59e0b" stroke-width="2" stroke-dasharray="6,4" rx="6" />
        <rect x="470" y="246" width="180" height="22" fill="#f59e0b" rx="4" />
        <text x="478" y="262" fill="#151515" font-family="Inter, sans-serif" font-size="11" font-weight="700">EXCAVATOR DANGER ZONE</text>
      </g>
    `;
  }

  // CCTV HUD Overlay
  const hud = `
    <!-- Top HUD Bar -->
    <rect x="20" y="20" width="1240" height="42" fill="#070709" opacity="0.75" rx="6" />
    <circle cx="42" cy="41" r="5" fill="#ef4444" />
    <text x="56" y="45" fill="#ffffff" font-family="monospace" font-size="13" font-weight="700">REC ● CAM-04_NORTH_ZONE</text>
    <text x="320" y="45" fill="#94a3b8" font-family="monospace" font-size="12">1080p @ 30fps · H.265 · 1.4 Mbps</text>
    <text x="760" y="45" fill="#94a3b8" font-family="monospace" font-size="12">LUX: 12.4 lx · SENSOR: ATMOS-4</text>
    <text x="1060" y="45" fill="#2597d0" font-family="monospace" font-size="13" font-weight="700">DISCERN AGENT: ACTIVE</text>

    <!-- Bottom Timestamp Bar -->
    <rect x="20" y="660" width="480" height="36" fill="#070709" opacity="0.75" rx="6" />
    <text x="36" y="683" fill="#ffffff" font-family="monospace" font-size="13">2026-09-15 05:42:19 UTC · SITE #104</text>

    <!-- Crosshair Target Reticle in Center -->
    <circle cx="640" cy="360" r="28" fill="none" stroke="#2597d0" stroke-width="1.5" opacity="0.4" stroke-dasharray="8,4" />
    <line x1="610" y1="360" x2="670" y2="360" stroke="#2597d0" stroke-width="1" opacity="0.4" />
    <line x1="640" y1="330" x2="640" y2="390" stroke="#2597d0" stroke-width="1" opacity="0.4" />
  `;

  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="100%" height="100%">
      ${bgGradient}
      ${scenery}
      ${workers}
      ${annotations}
      ${hud}
    </svg>
  `;

  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

export const SAMPLE_SCENARIOS: ScenarioPreset[] = [
  {
    id: "dawn-construction",
    title: "Dawn Dusk Low-Light",
    site: "North Industrial Excavation — Zone 4",
    camera: "CAM-04 (Perimeter Tower)",
    condition: "Low Light / Dawn (12.4 Lux)",
    tag: "Low Light",
    description: "Dawn lighting obscuring workers operating in the swing radius of a 30-ton excavator. Discern detects low ambient illumination, engages OpenCV CLAHE + Gamma restoration, locates workers with YOLOv8n, and identifies missing helmet and unsafe proximity.",
    rawImageUrl: makeCctvSvg("dawn", "raw"),
    restoredImageUrl: makeCctvSvg("dawn", "restored"),
    annotatedImageUrl: makeCctvSvg("dawn", "annotated"),
    analysis: {
      scene: {
        setting: "Construction site, excavation trench with heavy excavator",
        lighting: "low_light",
        weather: "clear",
        visibility: "reduced",
        human_presence: true,
        machinery_presence: true,
        summary: "Dawn low-light environment at active construction excavation zone with heavy tracked machinery and multiple ground personnel.",
      },
      plan: {
        restoration_tools: ["low_light_enhancement"],
        detection_tools: ["person_detection", "ppe_reasoning", "zone_reasoning"],
        reasoning: "Lighting is severely degraded at dawn (12 Lux), reducing contrast around worker heads and hazard boundaries. Low-light enhancement (CLAHE + Gamma 1.8) selected to expose shadow detail before YOLOv8n inference and Gemini PPE reasoning.",
      },
      steps: [
        { name: "Scene understanding", status: "completed", detail: "Dawn excavation site, low visibility, human & machinery detected", duration_ms: 382 },
        { name: "Tool selection", status: "completed", detail: "Selected CLAHE enhancement, YOLOv8n, PPE & Zone compliance", duration_ms: 12 },
        { name: "Restoration (OpenCV CLAHE)", status: "completed", detail: "Applied CLAHE contrast limit 3.0 + Gamma boost 1.8", duration_ms: 18 },
        { name: "Person detection (YOLOv8n)", status: "completed", detail: "2 persons localized with 94% and 98% confidence", duration_ms: 31 },
        { name: "Violation reasoning (Gemini)", status: "completed", detail: "2 active violations identified (Missing Helmet, Zone Breach)", duration_ms: 410 },
      ],
      detections: [
        {
          id: "p1",
          label: "Person 1",
          confidence: 0.94,
          box: { x: 530, y: 380, width: 70, height: 155 },
          source: "yolov8n_restored",
        },
        {
          id: "p2",
          label: "Person 2",
          confidence: 0.98,
          box: { x: 220, y: 410, width: 68, height: 148 },
          source: "yolov8n_restored",
        },
      ],
      violations: [
        {
          id: "v1",
          type: "missing_ppe",
          severity: "critical",
          description: "Worker (Person 1) is operating without a required Type I/II protective hard hat in an active heavy machinery excavation quadrant.",
          related_detection_ids: ["p1"],
          oshaCode: "OSHA 1926.100(a)",
          recommendation: "Immediate radio alert to Site Foreman to halt worker and mandate certified head protection.",
        },
        {
          id: "v2",
          type: "restricted_zone",
          severity: "critical",
          description: "Worker (Person 1) is positioned inside the 15-meter swing radius danger zone of operating tracked excavator without spotter coordination.",
          related_detection_ids: ["p1"],
          oshaCode: "OSHA 1926.600(a)(6)",
          recommendation: "Trigger audible perimeter exclusion alarm and verify clear line-of-sight.",
        },
      ],
      annotated_image: makeCctvSvg("dawn", "annotated"),
      raw_image: makeCctvSvg("dawn", "raw"),
      restored_image: makeCctvSvg("dawn", "restored"),
    },
  },
  {
    id: "fog-maritime-port",
    title: "Dense Maritime Sea Fog",
    site: "Port Logistics Terminal — Berth 7 Gantry",
    camera: "CAM-12 (Quay Gantry South)",
    condition: "Dense Sea Fog / Visibility < 30m",
    tag: "Dense Fog",
    description: "Coastal fog rolls in over container loading berths, scattering light and obscuring dock workers near automated straddle carriers. Discern deploys Dark-Channel Prior Dehazing before running zone violation checks.",
    rawImageUrl: makeCctvSvg("fog", "raw"),
    restoredImageUrl: makeCctvSvg("fog", "restored"),
    annotatedImageUrl: makeCctvSvg("fog", "annotated"),
    analysis: {
      scene: {
        setting: "Maritime container terminal berth, wet pavement, overhead gantry crane",
        lighting: "normal",
        weather: "fog",
        visibility: "poor",
        human_presence: true,
        machinery_presence: true,
        summary: "Dense maritime advection fog causing severe atmospheric scattering and masking quay boundaries.",
      },
      plan: {
        restoration_tools: ["dehaze"],
        detection_tools: ["person_detection", "ppe_reasoning", "zone_reasoning"],
        reasoning: "Atmospheric transmission is reduced below 25% due to aerosol fog particles. Fast Dark-Channel Prior Dehazing applied to recover edge sharpness before person detection.",
      },
      steps: [
        { name: "Scene understanding", status: "completed", detail: "Port berth, dense fog, poor visibility, gantry machinery active", duration_ms: 395 },
        { name: "Tool selection", status: "completed", detail: "Selected Dark-Channel Prior Dehaze, YOLOv8n, Zone reasoning", duration_ms: 10 },
        { name: "Restoration (OpenCV Dehaze)", status: "completed", detail: "Applied Dark-Channel Prior atmospheric light recovery", duration_ms: 24 },
        { name: "Person detection (YOLOv8n)", status: "completed", detail: "1 person detected in hazardous automated corridor", duration_ms: 29 },
        { name: "Violation reasoning (Gemini)", status: "completed", detail: "1 critical unauthorized entry violation logged", duration_ms: 375 },
      ],
      detections: [
        {
          id: "p1",
          label: "Dock Worker 1",
          confidence: 0.91,
          box: { x: 530, y: 380, width: 70, height: 155 },
          source: "yolov8n_dehazed",
        },
      ],
      violations: [
        {
          id: "v1",
          type: "restricted_zone",
          severity: "critical",
          description: "Unescorted worker detected within the automated straddle carrier path during restricted crane hoist operations under zero-visibility protocol.",
          related_detection_ids: ["p1"],
          oshaCode: "OSHA 1917.151 / Maritime Standard",
          recommendation: "E-Stop automatic gantry traverse and alert Port Operations Center.",
        },
      ],
      annotated_image: makeCctvSvg("fog", "annotated"),
      raw_image: makeCctvSvg("fog", "raw"),
      restored_image: makeCctvSvg("fog", "restored"),
    },
  },
  {
    id: "dust-open-pit-mine",
    title: "Open-Pit Mine Dust Storm",
    site: "Apex Copper Mine — Haul Road Ramp #2",
    camera: "CAM-08 (Pit Crest East)",
    condition: "High Dust Particulate (PM10 > 450)",
    tag: "Dust Storm",
    description: "Heavy 200-ton haul truck pass creates a dense dust plume blinding fixed CCTV models. Discern engages combined Dehazing and Bilateral Denoising to isolate pedestrian surveyors.",
    rawImageUrl: makeCctvSvg("dust", "raw"),
    restoredImageUrl: makeCctvSvg("dust", "restored"),
    annotatedImageUrl: makeCctvSvg("dust", "annotated"),
    analysis: {
      scene: {
        setting: "Open-pit mine haul road, mineral extraction ramp, heavy haul truck",
        lighting: "bright",
        weather: "dust",
        visibility: "poor",
        human_presence: true,
        machinery_presence: true,
        summary: "High particulate mineral dust dispersion on haul ramp reducing contrast and edge definition.",
      },
      plan: {
        restoration_tools: ["dehaze", "denoise"],
        detection_tools: ["person_detection", "ppe_reasoning", "zone_reasoning"],
        reasoning: "High-frequency dust noise and yellow-brown Rayleigh scattering obscuring haul truck blindspots. Bilateral filter and dehaze selected.",
      },
      steps: [
        { name: "Scene understanding", status: "completed", detail: "Open-pit mine, heavy dust storm, haul truck traffic present", duration_ms: 410 },
        { name: "Tool selection", status: "completed", detail: "Selected Dehaze + Bilateral Denoise, YOLOv8n, Unsafe Proximity", duration_ms: 11 },
        { name: "Restoration (Dehaze + Denoise)", status: "completed", detail: "Bilateral filter (d=9, sigma=75) + Dehaze", duration_ms: 32 },
        { name: "Person detection (YOLOv8n)", status: "completed", detail: "1 surveyor detected near haul truck blind spot", duration_ms: 34 },
        { name: "Violation reasoning (Gemini)", status: "completed", detail: "1 critical proximity violation flagged", duration_ms: 390 },
      ],
      detections: [
        {
          id: "p1",
          label: "Surveyor",
          confidence: 0.89,
          box: { x: 530, y: 380, width: 70, height: 155 },
          source: "yolov8n_denoised",
        },
      ],
      violations: [
        {
          id: "v1",
          type: "unsafe_proximity",
          severity: "critical",
          description: "Pedestrian surveyor walking within the 25-meter blind spot of a reversing 240-ton CAT 793F haul truck during dust-obscured conditions.",
          related_detection_ids: ["p1"],
          oshaCode: "MSHA 30 CFR § 56.9100",
          recommendation: "Activate proximity warning audio beacon on in-cab truck display.",
        },
      ],
      annotated_image: makeCctvSvg("dust", "annotated"),
      raw_image: makeCctvSvg("dust", "raw"),
      restored_image: makeCctvSvg("dust", "restored"),
    },
  },
  {
    id: "glare-steel-refinery",
    title: "Solar Glare & High Contrast",
    site: "Vulcan Steel Mill — Outdoor Coil Storage",
    camera: "CAM-02 (Storage Yard Mast)",
    condition: "Direct Solar Glare & Harsh Shadows",
    tag: "High Glare",
    description: "Low winter sun reflecting off galvanized steel sheet coils causes extreme overexposure and deep shadows. Discern adapts dynamic range compensation to inspect worker PPE.",
    rawImageUrl: makeCctvSvg("glare", "raw"),
    restoredImageUrl: makeCctvSvg("glare", "restored"),
    annotatedImageUrl: makeCctvSvg("glare", "annotated"),
    analysis: {
      scene: {
        setting: "Outdoor industrial metal yard, direct sunlight reflection, overhead crane",
        lighting: "bright",
        weather: "glare",
        visibility: "reduced",
        human_presence: true,
        machinery_presence: true,
        summary: "Intense solar specular reflection off metallic surfaces creating sensor saturation and blown highlights.",
      },
      plan: {
        restoration_tools: ["low_light_enhancement"],
        detection_tools: ["person_detection", "ppe_reasoning"],
        reasoning: "High dynamic range imbalance. Local adaptive tone mapping applied to recover shadowed worker features without clipping highlights.",
      },
      steps: [
        { name: "Scene understanding", status: "completed", detail: "Steel storage yard, severe specular glare and shadow contrast", duration_ms: 370 },
        { name: "Tool selection", status: "completed", detail: "Selected Adaptive Tone Mapping, YOLOv8n, PPE reasoning", duration_ms: 9 },
        { name: "Restoration (OpenCV Tone Map)", status: "completed", detail: "Applied local tone curve compression", duration_ms: 19 },
        { name: "Person detection (YOLOv8n)", status: "completed", detail: "2 riggers detected in shaded bay", duration_ms: 28 },
        { name: "Violation reasoning (Gemini)", status: "completed", detail: "1 warning violation logged (Missing Hi-Vis Vest)", duration_ms: 365 },
      ],
      detections: [
        {
          id: "p1",
          label: "Rigger 1",
          confidence: 0.93,
          box: { x: 530, y: 380, width: 70, height: 155 },
          source: "yolov8n_tonemapped",
        },
      ],
      violations: [
        {
          id: "v1",
          type: "missing_ppe",
          severity: "warning",
          description: "Worker (Person 1) wearing dark outerwear without ANSI/ISEA 107-2020 Class 2 High-Visibility reflective apparel in yard with mobile forklifts.",
          related_detection_ids: ["p1"],
          oshaCode: "OSHA 1926.201 / ANSI 107",
          recommendation: "Issue hi-vis vest reminder at yard check-in turnstile.",
        },
      ],
      annotated_image: makeCctvSvg("glare", "annotated"),
      raw_image: makeCctvSvg("glare", "raw"),
      restored_image: makeCctvSvg("glare", "restored"),
    },
  },
  {
    id: "clear-control-test",
    title: "Clear Daytime (Control Benchmark)",
    site: "Titan Logistics Hub — Main Yard",
    camera: "CAM-01 (Gate Control Tower)",
    condition: "Clear Sunlight / 85,000 Lux",
    tag: "Clear Weather",
    description: "Benchmark test in optimal weather conditions. Discern's agent intelligently skips restoration steps, saving compute cycles and latency while directly running person detection and compliance checks.",
    rawImageUrl: makeCctvSvg("clear", "raw"),
    restoredImageUrl: makeCctvSvg("clear", "restored"),
    annotatedImageUrl: makeCctvSvg("clear", "annotated"),
    analysis: {
      scene: {
        setting: "Logistics staging yard, concrete apron, gate barrier",
        lighting: "bright",
        weather: "clear",
        visibility: "good",
        human_presence: true,
        machinery_presence: false,
        summary: "Optimal daylight conditions with high ambient contrast and no atmospheric degradation.",
      },
      plan: {
        restoration_tools: [],
        detection_tools: ["person_detection", "ppe_reasoning"],
        reasoning: "Frame conditions are pristine (clear weather, good visibility, 85,000 Lux). Restoration bypassed completely to minimize latency and conserve GPU compute.",
      },
      steps: [
        { name: "Scene understanding", status: "completed", detail: "Clear daylight yard, excellent visibility, pedestrians present", duration_ms: 320 },
        { name: "Tool selection", status: "completed", detail: "Restoration skipped; direct dispatch to YOLOv8n & PPE check", duration_ms: 8 },
        { name: "Restoration", status: "skipped", detail: "Frame conditions did not warrant restoration (0ms compute saved)", duration_ms: 0 },
        { name: "Person detection (YOLOv8n)", status: "completed", detail: "1 person detected with 98% confidence", duration_ms: 22 },
        { name: "Violation reasoning (Gemini)", status: "completed", detail: "Full PPE compliance verified, zero violations", duration_ms: 310 },
      ],
      detections: [
        {
          id: "p2",
          label: "Site Inspector",
          confidence: 0.98,
          box: { x: 220, y: 410, width: 68, height: 148 },
          source: "yolov8n_direct",
        },
      ],
      violations: [],
      annotated_image: makeCctvSvg("clear", "annotated"),
      raw_image: makeCctvSvg("clear", "raw"),
      restored_image: makeCctvSvg("clear", "raw"),
    },
  },
];

export const SAMPLE_CAMERAS: CameraFeed[] = [
  {
    id: "cam-04",
    name: "CAM-04 · North Excavation",
    location: "North Industrial Excavation",
    zone: "Zone 4 Heavy Equipment",
    status: "alert",
    fps: 30,
    resolution: "1920x1080",
    lux: 12.4,
    weatherCondition: "Low-Light / Dawn",
    violationsCount: 2,
    lastChecked: "Just now",
    presetId: "dawn-construction",
  },
  {
    id: "cam-12",
    name: "CAM-12 · Quay Gantry South",
    location: "Port Logistics Terminal",
    zone: "Berth 7 Automated Corridor",
    status: "alert",
    fps: 25,
    resolution: "2560x1440",
    lux: 180.0,
    weatherCondition: "Dense Fog",
    violationsCount: 1,
    lastChecked: "2m ago",
    presetId: "fog-maritime-port",
  },
  {
    id: "cam-08",
    name: "CAM-08 · Pit Crest East",
    location: "Apex Copper Mine",
    zone: "Haul Ramp #2 Intersection",
    status: "alert",
    fps: 30,
    resolution: "1920x1080",
    lux: 940.0,
    weatherCondition: "Dust Plume",
    violationsCount: 1,
    lastChecked: "5m ago",
    presetId: "dust-open-pit-mine",
  },
  {
    id: "cam-01",
    name: "CAM-01 · Gate Control Tower",
    location: "Titan Logistics Hub",
    zone: "Main Gate Checkpoint",
    status: "active",
    fps: 60,
    resolution: "3840x2160",
    lux: 85000.0,
    weatherCondition: "Clear Day",
    violationsCount: 0,
    lastChecked: "1m ago",
    presetId: "clear-control-test",
  },
];

export const SAMPLE_AUDIT_LOGS: AuditRecord[] = [
  {
    id: "AUD-8921",
    timestamp: "2026-09-15 05:42:19",
    cameraId: "CAM-04",
    cameraName: "North Excavation Perimeter",
    zone: "Zone 4 Trench",
    weather: "Dawn / 12 Lux",
    lighting: "Low Light",
    violationType: "Missing Hard Hat + Machinery Proximity",
    severity: "critical",
    description: "Worker inside 15m excavator swing radius without approved helmet. CLAHE + Gamma restoration applied before detection.",
    restorationApplied: "CLAHE + Gamma (18ms)",
    status: "open",
  },
  {
    id: "AUD-8920",
    timestamp: "2026-09-15 05:38:04",
    cameraId: "CAM-12",
    cameraName: "Quay Gantry South",
    zone: "Berth 7 Quay",
    weather: "Dense Fog",
    lighting: "Normal",
    violationType: "Restricted Automated Zone Breach",
    severity: "critical",
    description: "Pedestrian in automated straddle corridor during hoist maneuver. Dehaze restored visibility from 25m to 120m.",
    restorationApplied: "Dark-Channel Prior Dehaze (24ms)",
    status: "acknowledged",
  },
  {
    id: "AUD-8919",
    timestamp: "2026-09-15 05:14:52",
    cameraId: "CAM-08",
    cameraName: "Pit Crest East",
    zone: "Haul Ramp #2",
    weather: "Mine Dust Plume",
    lighting: "Bright",
    violationType: "Haul Truck Blindspot Intrusion",
    severity: "critical",
    description: "Surveyor walking within 25m blindspot of reversing CAT 793F truck. Bilateral denoise + dehaze engaged.",
    restorationApplied: "Bilateral Filter + Dehaze (32ms)",
    status: "resolved",
  },
  {
    id: "AUD-8918",
    timestamp: "2026-09-15 04:59:11",
    cameraId: "CAM-02",
    cameraName: "Storage Yard Mast",
    zone: "Coil Storage Bay",
    weather: "Specular Glare",
    lighting: "High Contrast",
    violationType: "Missing Hi-Vis Apparel",
    severity: "warning",
    description: "Worker in dark clothing in high-traffic forklift transit bay. Adaptive tone mapping recovered shadow detail.",
    restorationApplied: "Tone Map Compression (19ms)",
    status: "resolved",
  },
  {
    id: "AUD-8917",
    timestamp: "2026-09-15 04:30:00",
    cameraId: "CAM-01",
    cameraName: "Gate Control Tower",
    zone: "Turnstile Gate",
    weather: "Clear",
    lighting: "Daylight",
    violationType: "None (Full Compliance)",
    severity: "info",
    description: "Periodic shift inspection. Scene classified as pristine; restoration bypassed dynamically (0ms latency penalty).",
    restorationApplied: "Skipped (Bypassed)",
    status: "resolved",
  },
];
