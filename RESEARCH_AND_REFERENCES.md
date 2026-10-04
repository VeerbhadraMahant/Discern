# Research and references

Everything this project drew on: the paper it implements, the methods and models behind each
component, the datasets, platform documentation, tooling, design sources, and the original
experiments we ran to answer questions the literature did not.

**How to read the verification notes.** Model IDs, pinned revisions and licenses were checked against
the Hugging Face API, the GitHub API and upstream license files on **2026-10-03** and are recorded in
`configs/models.yaml`. Paper titles, authors and arXiv numbers in sections 2 to 5 were written from
memory of the literature, not re-fetched in the session that wrote this file, except the DetAS paper
itself (a local copy is kept at `docs/paper/DetAS.pdf`, which is gitignored). Check an arXiv number
before citing it formally.

## 1. The foundational paper

**Detect in Any Scene: An Agentic Framework for Object Detection with Experience-Aware Reasoning**
(DetAS and DetAS-X), arXiv 2605.31174.

What we took from it:

| Idea | Where it lives in Discern |
|-|-|
| SAIR: scene label, one restorer per label, original-vs-restored image selection, super-resolution to a 2K target | `src/discern/agent/sair.py`, `agent/nodes/{perception,restorer_select,image_select,sr_select}.py` |
| MED: top-K=2 detector experts, instance grouping by IoU > 0.5 and crop cosine > 0.5 (32x32 crops, box expansion alpha=0.25), crop-level adjudication with group rejection | `vision/grouping.py`, `agent/med.py`, `agent/nodes/{detector_select,adjudicate}.py` |
| SEEH: 50 labelled samples per dataset, per-node metrics recorded by scene profile, top-3 similar profiles retrieved at inference | `src/discern/experience/` |
| DetAS-X attributes: object scale, density, visibility, illumination | `SceneProfile` in `agent/schemas.py` |
| F1 at IoU 0.5 as the headline metric | `eval/metrics.py` |

All paper constants are in `configs/thresholds.yaml`. Every place we interpreted an unspecified detail
or deviated is tabled in README section 3.1 (and in the local system design document).

## 2. Methods behind each component

### 2.1 Restoration and enhancement

| Component | Method | Reference | Used for |
|-|-|-|-|
| Display dehazing (`vision/dehaze.py`) and the classical dehaze restorer | Dark channel prior | K. He, J. Sun, X. Tang. *Single image haze removal using dark channel prior.* CVPR 2009; TPAMI 2011 | Haze removal for the "clear view" and the classical `dehaze` restorer |
| Transmission refinement | Guided filter | K. He, J. Sun, X. Tang. *Guided image filtering.* ECCV 2010; TPAMI 2013 | Removes block halos from the transmission map |
| Learned dehazing (registry entry, not in default profiles) | RIDCP | Wu et al. *RIDCP: Revitalizing real image dehazing via high-quality codebook priors.* CVPR 2023 (arXiv 2304.03322). Code: github.com/RQ-Wu/RIDCP_dehazing. License CC BY-NC 4.0 | Optional dehaze restorer |
| Deraining | MPRNet | Zamir et al. *Multi-stage progressive image restoration.* CVPR 2021 (arXiv 2102.02808). Code: github.com/swz30/MPRNet. Academic Public License | `restorer_derain` |
| Denoising | SwinIR | Liang et al. *SwinIR: Image restoration using Swin Transformer.* ICCVW 2021 (arXiv 2108.10257). Code: github.com/JingyunLiang/SwinIR. Apache-2.0 | `restorer_denoise` |
| Low light, fast | Zero-DCE++ | Li, Guo, Loy. *Learning to enhance low-light image via zero-reference deep curve estimation.* TPAMI 2021 (arXiv 2103.00860). Code: github.com/Li-Chongyi/Zero-DCE_extension. CC BY-NC 4.0 | `restorer_lowlight` |
| Low light, quality (registry only) | LLFlow | Wang et al. *Low-light image enhancement with normalizing flow.* AAAI 2022 (arXiv 2109.05923). Code: github.com/wyf0912/LLFlow. CC BY-NC-SA 4.0 | Not in the public profile |
| Super-resolution | Real-ESRGAN | Wang et al. *Real-ESRGAN: Training real-world blind super-resolution with pure synthetic data.* ICCVW 2021 (arXiv 2107.10833). Code: github.com/xinntao/Real-ESRGAN. BSD-3-Clause | 2K-target super-resolution, tiled |
| Classical fallbacks (frozen in `legacy/v0`) | CLAHE plus gamma; non-local means; simple dark channel prior | Standard OpenCV techniques | Restorers that need no weights |

### 2.2 Detection and grounding

| Component | Reference | Used for |
|-|-|-|
| YOLO-World v2 (`yolov8s-worldv2.pt`) | Cheng et al. *YOLO-World: Real-time open-vocabulary object detection.* CVPR 2024 (arXiv 2401.17270). Via Ultralytics (AGPL-3.0) | Fast general open-vocabulary detector |
| OWLv2 (`google/owlv2-base-patch16-ensemble`) | Minderer et al. *Scaling open-vocabulary object detection.* NeurIPS 2023 (arXiv 2306.09683) | Accurate local detector; best single detector on most of our degraded sets |
| Grounding DINO (`IDEA-Research/grounding-dino-base`) | Liu et al. *Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection.* ECCV 2024 (arXiv 2303.05499) | Accurate detector on the hosted profile |
| Rex-Omni (`IDEA-Research/Rex-Omni`) | IDEA Research model card. IDEA License 1.0, non-commercial research | Dense-object detector, optional, research use only |
| VLM grounding | Qwen3-VL's native box output, normalised to a 0 to 1000 coordinate convention, converted in one adapter | Relational phrases; zero-shot baseline |

### 2.3 Vision-language model and embeddings

| Component | Reference | Used for |
|-|-|-|
| Qwen3-VL (4B 4-bit local, 8B bf16 hosted) | Qwen Team. *Qwen3-VL technical report* (arXiv 2511.21631). Apache-2.0. The paper's own agent is Qwen3-VL-8B | Perception, restorer veto, image selection, adjudication, query parsing, answer phrasing, captions |
| SigLIP 2 (`google/siglip2-base-patch16-224`) | Tschannen et al. *SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features* (arXiv 2502.14786). Apache-2.0 | Frame embeddings for retrieval, track re-identification, crop similarity |
| 4-bit loading | bitsandbytes NF4 quantization (Dettmers et al., *QLoRA*, NeurIPS 2023, arXiv 2305.14314, introduced NF4) | Fit the 4B VLM in 8 GB |

### 2.4 Video

| Component | Reference | Used for |
|-|-|-|
| ByteTrack | Zhang et al. *ByteTrack: Multi-object tracking by associating every detection box.* ECCV 2022 (arXiv 2110.06864). Implementation: `supervision` (MIT) | Linking fused boxes into tracks |
| Shot detection | PySceneDetect ContentDetector (BSD-3-Clause), threshold 27 | One SAIR plan per shot |
| Decoding and encoding | PyAV (FFmpeg bindings) and FFmpeg H.264 | Frame decoding; annotated video export |
| Keyframe selection | Variance of the Laplacian as a sharpness measure (Pech-Pacheco et al., 2000) | Sharpest frame near each shot's midpoint |
| Track re-identification | Appearance matching on SigLIP 2 crop embeddings, cosine at least 0.8, time gap at most 2 s | Merging fragmented tracks to limit over-counting. Thresholds are ours, tuned on the 24-clip set |

### 2.5 Agent engineering ideas

| Idea | Source | Where used |
|-|-|-|
| Structured outputs with schema validation, one repair retry, deterministic fallback | General practice for LLM reliability (JSON schema validation; Pydantic v2) | `agent/llm_io.py` |
| Set-of-marks style prompting: numbered boxes drawn on a crop so the VLM answers with an index | Yang et al. *Set-of-Mark prompting unleashes extraordinary visual grounding in GPT-4V* (arXiv 2310.11441) | Group and track adjudication |
| Retrieval of past decisions into the prompt, with per-option statistics | The paper's SEEH, in the spirit of retrieval-augmented prompting | `experience/injection.py` |
| Verifying generated numbers against computed facts | Our design; related to "faithfulness checking" of generated text | `query/answer.py` |

## 3. Datasets and benchmarks

All datasets are used for local evaluation only. None are redistributed or shipped in the Space.
Image sets came from community Hugging Face mirrors because official download paths needed
registration; MARIS (underwater) was not used.

| Dataset | Reference | Source we used | Use |
|-|-|-|-|
| BDD100K | Yu et al. *BDD100K: A diverse driving dataset for heterogeneous multitask learning.* CVPR 2020 (arXiv 1805.04687) | HF mirror `dgural/bdd100k` (validation) | Clear, rainy and night road subsets; source of the video clips |
| HazyDet | Feng et al. *HazyDet: Open-source benchmark for drone-view object detection with depth-cues in hazy scenes* (arXiv 2409.19833) | HF dataset `ironmanfcf/HazyDet`, `real_world.zip` | Real hazy aerial frames |
| DarkFace | Yuan et al., UG2+ Challenge Dark Face dataset (low-light faces) | HF dataset `hieupth/dark_face` | Extreme low light |
| COCO val2017 | Lin et al. *Microsoft COCO: Common objects in context.* ECCV 2014 (arXiv 1405.0312) | Official annotations | Clean reference |
| Synthetic degradation | Our implementation in `eval/degrade.py`: fog from the atmospheric scattering model I = J t + A (1 - t) with t = exp(-beta d) (the model underlying the dark channel prior literature), gamma plus shot noise for low light, streak overlays for rain, Gaussian and Poisson sensor noise | Generated, seeded | 24 video clips and unit tests |

Dataset licenses are noted in README section 19. BDD100K and HazyDet are research or
non-commercial.

## 4. Metrics

| Metric | Definition as implemented |
|-|-|
| F1 at IoU 0.5 | Class-aware greedy matching, as in the paper (`eval/metrics.py`) |
| Count exact-match and MAE | Distinct accepted tracks versus label counts (`eval/video_metrics.py`) |
| Verifier pass rate | Share of answers whose numbers and times match the computed facts within 0.5 s |
| GPU seconds per video second | Measured on the local RTX 4060 (`serve/engine.py` accounting) |

## 5. Platform, tooling and documentation

| Topic | Source | What we relied on it for |
|-|-|-|
| Hugging Face ZeroGPU | Hugging Face Spaces documentation (ZeroGPU and the `spaces.GPU` decorator) | Per-call duration limits, supported Python versions (3.12.12 and 3.10.13 only), shared Blackwell slice (about 48 GB, sm_120), per-visitor daily quota |
| Gradio 6 and the Gradio JS client (`@gradio/client`) | gradio.app documentation | Hosted UI, named API endpoints, browser client |
| Hugging Face Hub API | huggingface.co docs | Verifying model IDs, revisions, licenses; dataset downloads |
| transformers | Hugging Face transformers docs (Qwen3-VL, OWLv2, Grounding DINO, SigLIP 2 implementations) | Pure-PyTorch model code, avoiding custom CUDA ops on sm_120 |
| PyTorch CUDA 12.8 wheels | pytorch.org (`cu128` index) | Blackwell-compatible builds; the RTX 4060 (sm_89) also runs them |
| uv | docs.astral.sh/uv | Dependency groups, lockfile, isolated environments |
| ruff, mypy, pytest | Their documentation | Lint, strict typing, tests |
| MLflow | mlflow.org docs | Experiment tracking with a local sqlite store |
| GitHub Actions and self-hosted runners | GitHub docs | CI, evaluation gate, and the rule that untrusted pull requests must not run on a self-hosted runner |
| Vercel | vercel.com docs | Static hosting of the React frontend |
| React 19, Vite, Tailwind v4, Vitest, Phosphor Icons | Their documentation | Frontend |
| Supervision, PySceneDetect, PyAV | Project documentation | Tracking, shot detection, decoding |

## 6. Design and brand sources

| Source | Used for |
|-|-|
| Refero style "Miranda" (https://styles.refero.design/style/3f6e3076-e77f-487e-b212-3b5946a34e87) | The editorial broadsheet look: parchment, bone, ink, one ember accent, sharp geometry, flat components |
| ui-ux-pro-max design guidance (accessibility, touch, motion, responsive rules) | Contrast minimums, 44 px targets, reduced-motion handling, loading states, breakpoints 375/768/1024/1440 |
| WCAG 2.x (contrast ratio 4.5:1 for text, 3:1 for large text and UI marks) | Colour decisions; measured pairs are in `frontend/src/lib/contrast.test.ts` |
| IBM Plex Sans and Mono; Montserrat | Typography, self-hosted (SIL OFL 1.1; provenance in `frontend/public/fonts/LICENSES.md`) |
| Fontsource packages | Font files for the Latin subsets |
| Project documents `docs/typography.md`, `docs/brand-guidelines.md`, `frontend/design-system/MASTER.md` | The resulting rules for type, voice, motifs and components |

## 7. Original experiments (our own research)

These answer questions the paper and the literature did not. Numbers are in README section 12; the
scripts that produced them are in `scripts/` and log to MLflow.

| Question | Experiment | Finding |
|-|-|-|
| How do the off-the-shelf detectors compare under degradation? | `run_baselines.py` on six datasets | OWLv2 leads on degraded sets; YOLO-World leads on COCO; Grounding DINO base is weak at our thresholds |
| Does restoration help detection? | `run_ablations_m2.py` (always-restore vs never vs SAIR) | Usually not. The image selector avoids the harm by rarely accepting the restored image |
| Does fusing detectors help? | `run_ablations_m3.py` (singles, pairs, all three, with and without VLM adjudication) | Best pair about equals best single; all three is worse; adjudication adds little because the VLM rarely rejects |
| Can the VLM adjudicate groups? | `diagnose_adjudication.py` | On BDD night it kept 42 of 50 false groups and all 9 real ones |
| Does experience memory improve decisions? | `harvest.py` (6800 records) and `run_ablation_m9.py` (prompt injection, per-node policy, joint policy) | No. The 4B VLM ignores injected evidence; harvest F1 with cheap fusion does not predict end-to-end behaviour |
| How does the video pipeline behave? | `make_synthetic_clips.py`, `eval_video_smoke.py` on 24 clips | Count exact-match 0.136, MAE 2.8, box F1 0.567, verifier pass rate 0.962, 22 GPU-seconds per video second; counts mostly low |
| Why does a hazy image still look hazy after cleaning? | Traced `plan_image` and the live engine on hazy aerial frames | The restorer is vetoed for detection reasons, then only super-resolution runs. Fix: a separate display-only clear view (dark channel prior, guided filter, neutral airlight, reduced chroma gain). Per-channel airlight and per-channel auto-levels caused strong colour casts and were rejected after side-by-side comparison |
| Which grouping thresholds and detector operating points? | Per-detector threshold tuning on separate 50-image splits | One global cutoff wrongly favoured YOLO-World; per-detector thresholds fixed it |

## 8. Where the numbers and decisions are recorded

- README sections 3.1 (deviations), 12 (results) and 13 (not done).
- `configs/models.yaml` for model IDs, pinned revisions and licenses; `configs/thresholds.yaml` for
  paper values and gate thresholds, each with a comment on its origin.
- Local, gitignored planning documents (`discern-plan.md` decision log, `system-design.md`) hold the
  dated narrative. MLflow runs under `mlruns/` hold the raw numbers.
- `legacy/v0/` is the frozen first version used as a baseline.
