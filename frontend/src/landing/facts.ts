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

export const NOT_DEPLOYED = "The hosted Hugging Face Space is not deployed yet.";

/** The one honest status line under the hero buttons. */
export const HERO_STATUS = "Open models, an 8 GB GPU, and not deployed yet.";

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
  /** Illustrative answer. The page labels it "Example, not a recorded result". */
  answer: string;
  /** The kind of evidence this question type points at. */
  evidence: string;
  /** Example evidence lines, also illustrative. */
  lines: string[];
  verified: boolean;
}

export const QUERY_TYPES: readonly QueryType[] = [
  {
    id: "locate",
    name: "Locate",
    question: "Where does the white van appear?",
    answer: "The white van is visible from 0:03 to 0:07, entering from the right.",
    evidence: "Boxes on frames and a time range per track.",
    lines: ["Track 2, label van, 0:03 to 0:07", "Status: accepted, with a crop and a rationale"],
    verified: true,
  },
  {
    id: "count",
    name: "Count",
    question: "How many people are in the clip?",
    answer: "Three people were tracked. Counts can run low, so check the tracks listed here.",
    evidence: "The tracks that were counted, each with its time range.",
    lines: ["Track 1, label person, 0:00 to 0:04", "Track 4, label person, 0:02 to 0:08", "Track 5, label person, 0:05 to 0:08"],
    verified: true,
  },
  {
    id: "temporal",
    name: "Temporal",
    question: "What happens right after the bus stops?",
    answer: "The bus stops at 0:06. Right after, one person crosses in front of it.",
    evidence: "Time ranges of the events, ordered on the clip timeline.",
    lines: ["Event 1, bus stopped, from 0:06", "Event 2, person crossing, 0:07 to 0:09"],
    verified: false,
  },
  {
    id: "relation",
    name: "Relation",
    question: "Is the cyclist to the left of the truck?",
    answer: "Yes, from 0:02 to 0:05 the cyclist is to the left of the truck.",
    evidence: "Both tracks and the frames where the relation holds.",
    lines: ["Track 3, label cyclist, 0:01 to 0:06", "Track 6, label truck, 0:00 to 0:08", "Relation holds on frames from 0:02 to 0:05"],
    verified: false,
  },
  {
    id: "describe",
    name: "Describe",
    question: "What is in the second shot?",
    answer: "The second shot shows a street with two cars and a person. This answer is not grounded in a single track.",
    evidence: "Detected objects of that shot with crops.",
    lines: ["Shot 2, detected: car, car, person", "Marked: not grounded in detections"],
    verified: false,
  },
  {
    id: "refine",
    name: "Follow-up",
    question: "And only the ones after 0:05?",
    answer: "Two of them, tracks 4 and 5, are present after 0:05.",
    evidence: "The earlier result set, narrowed, with the rule applied.",
    lines: ["Reuses the earlier result set, no new detection", "Verifier compares 2 against the narrowed track list"],
    verified: false,
  },
];

/** Locate and count ran with real models; follow-up chains were tested with test doubles only. */
export const QUERY_NOTE = "Locate and count were run with real models. The other four are implemented and tested with test doubles only.";

export interface SheetFrame {
  id: "degraded" | "cleaned" | "fused" | "tracked" | "answered";
  /** Timecode of the still, in the one format: m:ss. */
  time: string;
  title: string;
  caption: string;
}

/** The contact sheet: one clip walked through the pipeline. The stills are drawn, not recorded. */
export const SHEET: readonly SheetFrame[] = [
  { id: "degraded", time: "0:02", title: "Degraded", caption: "A hazy frame comes in. Fog and noise hide the car and the person." },
  { id: "cleaned", time: "0:03", title: "Cleaned", caption: "A restorer is applied only because this shot needed it. A clear shot would be left alone." },
  { id: "fused", time: "0:04", title: "Fused", caption: "Three detectors propose boxes; boxes that overlap and look alike are merged into one." },
  { id: "tracked", time: "0:05", title: "Tracked", caption: "ByteTrack links the box across frames, so the car keeps one identity over time." },
  { id: "answered", time: "0:06", title: "Answered", caption: "The answer cites the track and its time range, and a verifier checks every number in it." },
];

export interface Pillar {
  id: "clean" | "evidence" | "checked";
  title: string;
  text: string;
  /** One measured proof sentence, with the number inside it. */
  proof: string;
  source: string;
}

export const PILLARS: readonly Pillar[] = [
  {
    id: "clean",
    title: "Clean sight",
    text: "It restores a shot only when restoring helps, and says when it chose not to.",
    proof: "Always restoring dropped F1 on HazyDet from 0.664 to 0.593, which is why restoration is optional.",
    source: "README, Findings on the degraded sets",
  },
  {
    id: "evidence",
    title: "Shown evidence",
    text: "Every answer points at tracks, crops and timestamps you can open and check.",
    proof: "Locate and count answers were verified with real models, and each lists the tracks it rests on.",
    source: "README, Video evaluation",
  },
  {
    id: "checked",
    title: "Checked numbers",
    text: "A verifier compares each number in an answer with computed facts before showing it.",
    proof: "The verifier passed 0.962 of answers on 24 synthetic clips. That checks consistency, not truth.",
    source: "README, Video evaluation",
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

export interface Principle {
  id: string;
  title: string;
  text: string;
  /** One short evidence line shown under the principle. */
  evidence: string;
  source: string;
}

/** The engineering and research method this system is built on. Every statement is backed by the repository. */
export const METHOD: readonly Principle[] = [
  {
    id: "paper",
    title: "Start from a paper, then measure it",
    text: "The design follows DetAS and DetAS-X: self-adaptive restoration, multi-expertise detection and experience memory. Each mechanism can be switched off, so it can be ablated, and every deviation from the paper is written down.",
    evidence: "Ablations: no restoration, always restore, and full restoration; fusion with and without adjudication",
    source: "discern-plan.md, ablation checklists and decision log",
  },
  {
    id: "measure",
    title: "Measure before building",
    text: "The evaluation harness and the baselines came before any clever component: F1 at IoU 0.5 with class-aware greedy matching, each detector alone, the vision-language model zero-shot and the old v0 pipeline. Thresholds are tuned on a separate split, so reported numbers are never tuned on the test images.",
    evidence: "100-image gate subsets, 50-image tuning splits",
    source: "README, Measured results; discern-plan.md decision 12",
  },
  {
    id: "code",
    title: "Code computes, the model judges",
    text: "Counts, times, boxes and metrics come from deterministic code. The vision-language model only selects, adjudicates and phrases, and a verifier checks every number and timestamp in an answer against the computed facts.",
    evidence: "Answer verifier pass rate 0.962 on 24 synthetic clips",
    source: "README, Engineering rules and Video evaluation",
  },
  {
    id: "structured",
    title: "Every model call is structured",
    text: "Output is validated against a schema, a malformed reply gets one repair retry and then a deterministic fallback, so it never crashes a request. Prompts are versioned files with golden snapshot tests.",
    evidence: "Schema check, one repair retry, then a fallback",
    source: "README, Engineering rules; src/discern/agent/llm_io.py and prompts",
  },
  {
    id: "doubles",
    title: "Test with doubles first",
    text: "The logic is covered by 600+ CPU tests that use fake models. Real-model runs then confirm it. Continuous integration runs the linter, the type checker and the tests on every push.",
    evidence: "600+ CPU tests, lint, type check and tests in CI",
    source: "README, Run it locally; .github/workflows/ci.yml",
  },
  {
    id: "trace",
    title: "Trace everything",
    text: "Every node records its decision, its rationale, its duration and whether it fell back, and the Trace tab in the interface shows them.",
    evidence: "Decision, rationale, duration and fallback per node",
    source: "README, Engineering rules; src/discern/trace",
  },
  {
    id: "fail",
    title: "Report failures",
    text: "Ablations are published even when they are negative: restoration often lowers F1, experience memory did not help and counts run low. The README and this page say so.",
    evidence: "See the section on what did not work",
    source: "README, Findings on the degraded sets and Video evaluation",
  },
  {
    id: "constraints",
    title: "Constraints as design",
    text: "Open models only, a budget of zero dollars and an 8 GB GPU, with one codebase and two configuration profiles. Privacy is the default: uploads are deleted on a time limit and kept only with an explicit opt-in.",
    evidence: "Profiles local_lite and hosted_full",
    source: "discern-plan.md, goals; README, Run it locally",
  },
];

export interface Faq {
  id: string;
  q: string;
  a: string;
}

export const RUN_LOCALLY = `git clone ${REPO_URL}.git && cd Discern
uv sync --group gpu --group eval && uv run python space/app.py
cd frontend && npm ci && npm run dev`;

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
    a: "Clone the repository, run uv sync --group gpu --group eval, start the backend with uv run python space/app.py, then in frontend/ run npm ci and npm run dev. The three commands are in the Run it locally block below; run the third in a second terminal.",
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
  { id: "tracking", title: "No analytics scripts", text: "The frontend ships no analytics or tracking code, and its fonts are served from the same site." },
];
