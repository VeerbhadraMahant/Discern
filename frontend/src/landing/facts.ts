/**
 * Single source of truth for every claim on the landing page.
 *
 * Everything here was measured or recorded in this repository's own runs and is described in the
 * repository README (sections "Measured results", "How it works", "Licenses and limits"), the
 * model registry `configs/models.yaml` and the decision log in `discern-plan.md`. Nothing is
 * invented: no users, testimonials, logos, benchmarks from elsewhere or hosted-Space numbers.
 * Each item carries a `source` string so the page can show where it came from.
 */

export const REPO_URL = "https://github.com/VeerbhadraMahant/Discern";
export const PAPER_URL = "https://arxiv.org/abs/2605.31174";
export const PAPER_TITLE = "Detect in Any Scene (DetAS / DetAS-X), arXiv 2605.31174";

export const ANNOUNCEMENT = "Open models only. Work in progress.";
export const NOT_DEPLOYED = "The hosted Hugging Face Space is not deployed yet.";

/** One plain sentence under the hero, built from the facts below (no meta string with separators). */
export const DATELINE = "Volume 1. Built on open models and an 8 GB GPU. Work in progress, and not deployed yet.";

export type DatasetId = "coco" | "bddClear" | "bddRainy" | "bddNight" | "hazydet" | "darkface";

export const DATASETS: ReadonlyArray<{ id: DatasetId; label: string; note: string }> = [
  { id: "coco", label: "COCO", note: "COCO val2017, 100-image gate subset (scored before the crowd-box fix)" },
  { id: "bddClear", label: "BDD clear", note: "BDD100K validation, clear weather, 100 images" },
  { id: "bddRainy", label: "BDD rainy", note: "BDD100K validation, rainy, 100 images" },
  { id: "bddNight", label: "BDD night", note: "BDD100K validation, night, 100 images" },
  { id: "hazydet", label: "HazyDet", note: "HazyDet real-world hazy set, 100 images" },
  { id: "darkface", label: "DarkFace", note: "DarkFace low-light set, 100 images" },
];

export interface Method {
  id: string;
  name: string;
  scores: Record<DatasetId, number>;
  source: string;
}

const BASELINE_SOURCE = "README, Measured results table (F1 at IoU 0.5, fixed 100-image gate subsets)";

/** F1 at IoU 0.5 on the fixed 100-image gate subsets. */
export const METHODS: readonly Method[] = [
  {
    id: "yolo-world",
    name: "YOLO-World v2",
    scores: { coco: 0.66, bddClear: 0.52, bddRainy: 0.533, bddNight: 0.436, hazydet: 0.387, darkface: 0 },
    source: BASELINE_SOURCE,
  },
  {
    id: "owlv2",
    name: "OWLv2 base",
    scores: { coco: 0.492, bddClear: 0.613, bddRainy: 0.643, bddNight: 0.516, hazydet: 0.664, darkface: 0.137 },
    source: BASELINE_SOURCE,
  },
  {
    id: "gdino",
    name: "Grounding DINO base",
    scores: { coco: 0.621, bddClear: 0.259, bddRainy: 0.256, bddNight: 0.202, hazydet: 0.187, darkface: 0.148 },
    source: BASELINE_SOURCE,
  },
  {
    id: "qwen",
    name: "Qwen3-VL-4B zero-shot",
    scores: { coco: 0.547, bddClear: 0.534, bddRainy: 0.586, bddNight: 0.501, hazydet: 0.411, darkface: 0.155 },
    source: BASELINE_SOURCE,
  },
  {
    id: "legacy",
    name: "Legacy v0 pipeline",
    scores: { coco: 0.614, bddClear: 0.494, bddRainy: 0.543, bddNight: 0.36, hazydet: 0.208, darkface: 0 },
    source: BASELINE_SOURCE,
  },
];

/** Format a score exactly as the table and the chart show it. */
export function fmt(n: number): string {
  return n.toFixed(3);
}

export interface Stat {
  id: string;
  /** Sentence parts: the number sits inside the sentence at body size, weight 600. */
  before: string;
  value: string;
  after: string;
  source: string;
}

const VIDEO_SOURCE = "README, Measured results: 24 synthetic 8-second clips built from labelled BDD frames";

export const STATS: readonly Stat[] = [
  { id: "verifier", before: "The answer verifier passed", value: fmt(0.962), after: "of answers on 24 synthetic video clips.", source: VIDEO_SOURCE },
  { id: "boxf1", before: "Box F1 on the same clips was", value: fmt(0.567), after: "at IoU 0.5.", source: VIDEO_SOURCE },
  { id: "mae", before: "Counts were off by", value: "2.8", after: "objects on average, and mostly too low.", source: VIDEO_SOURCE },
  { id: "exact", before: "Only", value: fmt(0.136), after: "of counts matched the true count exactly.", source: VIDEO_SOURCE },
  {
    id: "gpu",
    before: "Video took about",
    value: "22",
    after: "GPU-seconds per video-second on an RTX 4060 8 GB.",
    source: "README: measured locally; hosted speed and quota were not measured",
  },
  {
    id: "tests",
    before: "The repository has",
    value: "600+",
    after: "automated tests, which run on CPU with test doubles.",
    source: "Repository test suite (tests/ and frontend/src)",
  },
];

export interface Feature {
  id: string;
  title: string;
  text: string;
}

export const FEATURES: readonly Feature[] = [
  {
    id: "clean",
    title: "Adaptive cleaning per shot",
    text: "A vision-language model labels each shot and code picks a restorer, which the model can veto, so a clear shot is left alone.",
  },
  {
    id: "fusion",
    title: "Multi-detector fusion",
    text: "Three open-vocabulary detectors propose boxes that are grouped by overlap and crop similarity; fusing the best two roughly matches the best single detector.",
  },
  {
    id: "evidence",
    title: "Grounded evidence",
    text: "Answers about finding, counting or timing point to boxes, tracks and timestamps you can inspect.",
  },
  {
    id: "verified",
    title: "Verified answers",
    text: "Every number and timestamp in an answer is checked against computed facts before it is shown.",
  },
  {
    id: "followups",
    title: "Follow-up questions",
    text: "Follow-ups reuse earlier result sets; follow-up chains were tested with test doubles only.",
  },
  {
    id: "index",
    title: "Video index and retrieval",
    text: "Shot detection, per-frame fusion and annotated video export turn a clip into an index you can query.",
  },
  {
    id: "tracking",
    title: "Tracking and re-identification",
    text: "ByteTrack links boxes over time and re-identification joins tracks; counts still come out mostly too low.",
  },
  {
    id: "memory",
    title: "Experience memory (SEEH)",
    text: "6,800 harvested records feed the decisions, but end to end this did not beat plain DetAS.",
  },
  {
    id: "feedback",
    title: "Feedback loop",
    text: "Opt-in feedback is stored and the promotion logic is unit tested; turning it into new memory is not wired yet.",
  },
  {
    id: "privacy",
    title: "Private by default",
    text: "Uploads are session-scoped, deleted on a time limit, and kept only with explicit opt-in.",
  },
  {
    id: "local",
    title: "Runs locally on 8 GB",
    text: "Measured on an RTX 4060 8 GB laptop GPU with a 4-bit Qwen3-VL-4B; also packaged for a ZeroGPU Space.",
  },
  {
    id: "traces",
    title: "Observable traces",
    text: "Every node records its decision, rationale, duration and GPU time, and the Trace tab shows them.",
  },
];

export interface Step {
  id: string;
  title: string;
  text: string;
  paper?: ReadonlyArray<"SAIR" | "MED" | "SEEH">;
}

export const STEPS: readonly Step[] = [
  { id: "upload", title: "Upload", text: "Add a video or an image. It stays in your session." },
  {
    id: "clean",
    title: "Perceive and clean",
    text: "A vision-language model labels the scene; code picks a restorer and the model can veto it.",
    paper: ["SAIR", "SEEH"],
  },
  {
    id: "detect",
    title: "Detect and fuse",
    text: "YOLO-World, OWLv2 and Grounding DINO propose boxes; groups are merged by overlap and crop similarity.",
    paper: ["MED", "SEEH"],
  },
  {
    id: "track",
    title: "Track and verify",
    text: "ByteTrack and re-identification build tracks; a parsed query plan runs in code, not in the model.",
  },
  {
    id: "answer",
    title: "Answer with evidence",
    text: "The model only phrases the answer; a verifier checks every number and timestamp before it is shown.",
  },
];

export interface QueryType {
  id: string;
  name: string;
  question: string;
  evidence: string;
  verified: boolean;
}

export const QUERY_TYPES: readonly QueryType[] = [
  { id: "locate", name: "Locate", question: "Where does the white van appear?", evidence: "Boxes on frames and a time range per track.", verified: true },
  { id: "count", name: "Count", question: "How many people cross the road?", evidence: "The tracks that were counted, each with its time range.", verified: true },
  { id: "temporal", name: "Temporal", question: "What happens right after the bus stops?", evidence: "Time ranges of the events, ordered on the clip timeline.", verified: false },
  { id: "relation", name: "Relation", question: "Is the cyclist to the left of the truck?", evidence: "Both tracks and the frames where the relation holds.", verified: false },
  { id: "describe", name: "Describe", question: "What is in the second shot?", evidence: "Detected objects of that shot with crops.", verified: false },
  { id: "refine", name: "Refine", question: "Only the ones after 0:05.", evidence: "The earlier result set, narrowed, with the rule applied.", verified: false },
];

export interface Example {
  id: string;
  tab: string;
  question: string;
  answer: string;
  evidence: string[];
  note: string;
}

/** Illustrative only. The page shows the stamp "Example, not a recorded result" next to these. */
export const EXAMPLES: readonly Example[] = [
  {
    id: "locate",
    tab: "Locate",
    question: "Where does the white van appear?",
    answer: "The white van is visible from 0:03 to 0:07, entering from the right.",
    evidence: ["Track 2, label van, 0:03 to 0:07", "Status: accepted, with a crop and a rationale"],
    note: "Locate is verified with real models.",
  },
  {
    id: "count",
    tab: "Count",
    question: "How many people are in the clip?",
    answer: "Three people were tracked. Counts can run low, so check the tracks listed below.",
    evidence: ["Track 1, label person, 0:00 to 0:04", "Track 4, label person, 0:02 to 0:08", "Track 5, label person, 0:05 to 0:08"],
    note: "Count is verified with real models, and in our runs counts were mostly too low.",
  },
  {
    id: "refine",
    tab: "Follow-up",
    question: "And only the ones after 0:05?",
    answer: "Two of them, tracks 4 and 5, are present after 0:05.",
    evidence: ["Reuses the earlier result set, no new detection", "Verifier compares 2 against the narrowed track list"],
    note: "Follow-up chains were tested with test doubles only.",
  },
];

export interface ModelRow {
  name: string;
  role: string;
  license: string;
  hosted: string; // hosted-use note, not legal advice
}

const REGISTRY = "configs/models.yaml";
export const MODELS_SOURCE = `${REGISTRY}, licenses verified 2026-10-03 against upstream files`;

export const MODELS: readonly ModelRow[] = [
  { name: "Qwen3-VL-4B (4-bit) / 8B", role: "Agent VLM: scene labels, adjudication, answer wording", license: "Apache-2.0", hosted: "Permissive license" },
  { name: "YOLO-World v2", role: "Fast open-vocabulary detector", license: "AGPL-3.0 (via Ultralytics)", hosted: "AGPL-3.0: check the source-sharing terms for hosted use" },
  { name: "OWLv2 base", role: "Accurate open-vocabulary detector", license: "Apache-2.0", hosted: "Permissive license" },
  { name: "Grounding DINO base", role: "Accurate open-vocabulary detector", license: "Apache-2.0", hosted: "Permissive license" },
  { name: "SigLIP 2 base", role: "Embedder for crop similarity", license: "Apache-2.0", hosted: "Permissive license" },
  { name: "SwinIR", role: "Denoising restorer", license: "Apache-2.0", hosted: "Permissive license" },
  { name: "Real-ESRGAN x4plus", role: "Super-resolution", license: "BSD-3-Clause", hosted: "Permissive license" },
  { name: "RIDCP", role: "Dehazing restorer", license: "CC-BY-NC-4.0", hosted: "Non-commercial only" },
  { name: "MPRNet", role: "Deraining restorer", license: "Academic Public License", hosted: "Non-commercial only" },
  { name: "Zero-DCE++", role: "Low-light restorer", license: "CC-BY-NC-4.0", hosted: "Non-commercial only" },
  { name: "LLFlow", role: "Low-light restorer", license: "CC-BY-NC-SA-4.0", hosted: "Academic only; not in the public profile" },
  { name: "Rex-Omni", role: "Optional dense detector", license: "IDEA License 1.0", hosted: "Research only; not in the public profile" },
  { name: "Classical fallbacks", role: "Dehaze, low-light, denoise without a model", license: "Project license", hosted: "Same as Discern" },
];

export const STACK = [
  "Python 3.12",
  "PyTorch",
  "Gradio",
  "Gradio JS client",
  "Vite",
  "React 19",
  "TypeScript",
  "Tailwind v4",
  "MLflow",
  "GitHub Actions CI",
] as const;

export type MilestoneStatus = "done" | "limits" | "open";

export interface Milestone {
  id: string;
  title: string;
  status: MilestoneStatus;
  text: string;
}

export const MILESTONES: readonly Milestone[] = [
  { id: "M0", title: "Foundations", status: "done", text: "Package, config, trace, fakes, model registry, CI." },
  { id: "M1", title: "Baselines", status: "done", text: "Five baselines logged on fixed 100-image gate subsets." },
  { id: "M2", title: "SAIR on images", status: "limits", text: "Avoids harmful restoration but did not beat no-restoration on rainy or DarkFace." },
  { id: "M3", title: "MED on images", status: "limits", text: "Fusing two detectors is about equal to the best single detector; margins are thin." },
  { id: "M4", title: "Hosted ZeroGPU Space", status: "open", text: "Not deployed yet. Hosted speed and quota are unmeasured." },
  { id: "M5", title: "Video ingest and tracking", status: "limits", text: "Works on 24 synthetic clips; the counting tolerance was not met." },
  { id: "M6", title: "Video index and queries", status: "limits", text: "Locate and count verified with real models; other types on test doubles." },
  { id: "M7", title: "Follow-ups", status: "limits", text: "Built and tested with test doubles only." },
  { id: "M8", title: "Video eval set and gate", status: "limits", text: "Gate thresholds set; the self-hosted runner is not registered, so CI runs it by hand." },
  { id: "M9", title: "SEEH experience", status: "limits", text: "Built and measured; it did not beat plain DetAS." },
  { id: "M10", title: "Feedback loop", status: "limits", text: "Store, monitoring and promotion logic are tested; the loop was never run on real feedback." },
  { id: "M11", title: "React frontend", status: "done", text: "This client, with a demo mode that needs no backend." },
];

export interface Faq {
  id: string;
  q: string;
  a: string;
}

export const RUN_LOCALLY = `git clone ${REPO_URL}.git
cd Discern
uv sync --group gpu --group eval
uv run python space/app.py

# second terminal
cd frontend
npm ci
npm run dev`;

export const FAQ: readonly Faq[] = [
  {
    id: "what",
    q: "What is Discern?",
    a: "A research prototype: upload a video or image, Discern cleans it per shot, and you ask questions in plain language. Answers about finding, counting, timing or relating objects point to boxes, tracks and timestamps you can inspect.",
  },
  {
    id: "cost",
    q: "What does it cost?",
    a: "The code is open and runs on your own hardware, so there is no fee beyond that hardware. A hosted Space is not deployed yet, so there is no hosted price or quota to quote.",
  },
  {
    id: "hardware",
    q: "What hardware does it need?",
    a: "The local profile was measured on an RTX 4060 8 GB laptop GPU with a 4-bit Qwen3-VL-4B. Video runs at about 22 GPU-seconds per video-second there, so it is slow.",
  },
  {
    id: "deployed",
    q: "Is it deployed?",
    a: "Not yet. It is packaged for a Hugging Face ZeroGPU Space, but nothing has been published, and every number on this page comes from local runs.",
  },
  {
    id: "data",
    q: "What data is stored?",
    a: "Uploads are session-scoped and deleted on a time limit. Media is kept for feedback only if you tick an explicit opt-in. Start over deletes your session immediately.",
  },
  {
    id: "accurate",
    q: "How accurate is it?",
    a: "Modestly. On 24 synthetic video clips the count exact-match rate was 0.136, box F1 was 0.567 and counts were mostly too low. The answer verifier passed 0.962 of answers, which checks consistency with computed facts, not truth.",
  },
  {
    id: "models",
    q: "Which models does it use?",
    a: "Open models only: YOLO-World v2, OWLv2, Grounding DINO, Qwen3-VL, SigLIP 2, Real-ESRGAN, SwinIR, MPRNet, Zero-DCE++, LLFlow and classical fallbacks. See the models table above.",
  },
  {
    id: "commercial",
    q: "Can I use it commercially?",
    a: "Check each model first. RIDCP, MPRNet and Zero-DCE++ are non-commercial, LLFlow is academic only, Rex-Omni is research only, and YOLO-World via Ultralytics is AGPL-3.0. Qwen3-VL, OWLv2, Grounding DINO, SigLIP 2, SwinIR and Real-ESRGAN are Apache-2.0 or BSD.",
  },
  {
    id: "run",
    q: "How do I run it locally?",
    a: "Clone the repository, run uv sync --group gpu --group eval, start the backend with uv run python space/app.py, then in frontend/ run npm ci and npm run dev. The full commands are in the Run it locally block below.",
  },
  {
    id: "detas",
    q: "What is DetAS?",
    a: "Detect in Any Scene (DetAS and DetAS-X, arXiv 2605.31174): self-adaptive image restoration (SAIR), multi-expertise detection (MED) and self-evolving experience harvesting (SEEH). Discern extends those ideas to video and question answering.",
  },
];

export const DID_NOT_WORK: ReadonlyArray<{ id: string; title: string; text: string; source: string }> = [
  {
    id: "restore",
    title: "Restoration often lowers F1",
    text: "Always restoring dropped F1 on HazyDet (0.593 vs 0.664 unrestored) and BDD night (0.438 vs 0.516). The image selector avoids that harm, but mostly by never accepting a restored image.",
    source: "README, Findings on the degraded sets",
  },
  {
    id: "seeh",
    title: "Experience memory did not help",
    text: "SEEH harvested 6,800 records over 4 degraded datasets and 50 images each. End to end, using it did not beat plain DetAS. The 4B model ignored the evidence placed in its prompt.",
    source: "README, Experience (SEEH); discern-plan.md decision 22",
  },
  {
    id: "fusion",
    title: "More detectors is not better",
    text: "Fusing the best two detectors is about equal to the best single one (differences from -0.003 to +0.012). Fusing all three is worse.",
    source: "README, Findings on the degraded sets",
  },
  {
    id: "counts",
    title: "Counts come out too low",
    text: "On 24 synthetic clips the count exact-match rate was 0.136 and 37 of 51 label counts were under the truth.",
    source: "README, Video evaluation",
  },
  {
    id: "gpu",
    title: "GPU cost is high",
    text: "About 22 GPU-seconds per video-second on an RTX 4060. Hosted speed and quota were not measured.",
    source: "README, Video evaluation",
  },
  {
    id: "hosted",
    title: "The hosted Space is not deployed yet",
    text: "Nothing here is a claim about a public service. Query types other than locate and count were tested with test doubles only.",
    source: "README, Status",
  },
];

export const PRIVACY: ReadonlyArray<{ id: string; title: string; text: string }> = [
  { id: "session", title: "Session-scoped", text: "Your upload belongs to one session and is not shared with other sessions." },
  { id: "ttl", title: "Deleted on a time limit", text: "Sessions expire on a time limit, and Start over deletes yours immediately." },
  { id: "optin", title: "Retention is opt-in", text: "Media is kept for feedback only after you tick an explicit checkbox; it is off by default." },
  { id: "verifier", title: "Answers are checked", text: "Every number and timestamp in an answer is verified against computed facts before display." },
  { id: "tracking", title: "No analytics scripts", text: "The frontend ships no analytics or tracking code. Fonts load from Google Fonts." },
];
