# Discern

Degradation-aware, grounded video query agent. Upload a video or image, Discern adaptively
cleans it, and you ask questions in natural language, including follow-ups. Answers about finding,
counting, timing or relating objects point to boxes, tracks and timestamps you can inspect.

It builds on the ideas of *Detect in Any Scene* (DetAS / DetAS-X, arXiv 2605.31174): self-adaptive
image restoration (SAIR), multi-expertise detection (MED) and self-evolving experience harvesting
(SEEH), extended here to video and question answering. Open models only; runs locally on an 8 GB
GPU and is packaged for a Hugging Face ZeroGPU Space.

Status: working research prototype. Everything below was measured on this repository's own runs;
nothing here is a claim about the hosted Space, which has not been deployed.

## How it works

1. **Perception and restoration (SAIR).** A VLM labels the scene (fog, rain, low light, ...). Code
   maps the label to a restorer; the VLM may veto it and compares original and restored frames on
   detection-oriented criteria. Super-resolution is decided in code (target size) and by the VLM.
2. **Detection (MED).** Open-vocabulary detectors (YOLO-World, OWLv2, Grounding DINO) propose boxes;
   proposals are grouped per the paper's IoU plus crop-similarity rule; groups are adjudicated by
   the VLM on a numbered crop (the evaluation and harvest runs skip the VLM where two detectors agree).
3. **Video.** Shot detection, one SAIR plan per shot, per-frame fusion, ByteTrack, re-identification,
   per-track adjudication, annotated video export.
4. **Questions.** A parsed query plan (locate, count, temporal, relation, describe, refine) is
   executed in code; the VLM only phrases the answer, and a verifier checks every number and
   timestamp against the computed facts before it is shown. Follow-ups reuse earlier result sets.
   Only locate and count were run with real models; the other query types and follow-up chains
   are tested with fakes only.
5. **Experience (SEEH).** Harvested per-scene evidence (restorer, super-resolution and detector-set
   F1) is injected into the decision prompts, or, where it is decisive, applied by code. Opt-in
   feedback is stored and the promotion logic is unit tested, but turning feedback into new memory
   versions is not wired (the harvest step of `scripts/feedback_loop.py` is a stub).

Engineering rules: the VLM never invents a quantity, every VLM call is schema-validated with one
repair retry and a deterministic fallback, every node is traced, no model ID is hardcoded.

## Measured results

F1 at IoU 0.5 on fixed 100-image gate subsets (detector thresholds tuned on separate 50-image
splits). Datasets come from unofficial Hugging Face mirrors (BDD100K validation, HazyDet
real-world, DarkFace, COCO val2017); MARIS was not used.

| method | COCO | BDD clear | BDD rainy | BDD night | HazyDet | DarkFace |
|-|-|-|-|-|-|-|
| YOLO-World v2 | 0.660 | 0.520 | 0.533 | 0.436 | 0.387 | 0.000 |
| OWLv2 base | 0.492 | 0.613 | 0.643 | 0.516 | 0.664 | 0.137 |
| Grounding DINO base | 0.621 | 0.259 | 0.256 | 0.202 | 0.187 | 0.148 |
| Qwen3-VL-4B (zero-shot) | 0.547 | 0.534 | 0.586 | 0.501 | 0.411 | 0.155 |
| legacy v0 (classical restoration + YOLOv8n) | 0.614 | 0.494 | 0.543 | 0.360 | 0.208 | 0.000 |

The COCO subset keeps COCO's crowd boxes as ground truth (the loader now skips them; the numbers above
predate that fix). The legacy v0 row picks its restorer from the dataset's scene label.

Findings on the degraded sets (OWLv2 as the detector):

- Restoration with the available restorers usually lowers F1 (always-restore: HazyDet 0.593 vs 0.664
  unrestored, night 0.438 vs 0.516). The VLM image selector avoids that harm by never accepting a
  restored image, so SAIR behaves like "do not restore"; it matches always-restore on rainy
  (0.643 vs 0.648) but is below it on DarkFace (0.137 vs 0.189).
- Fusing the best two detectors is about equal to the best single detector on the full gate
  (differences from -0.003 to +0.012). Fusing all three is clearly worse. Full MED with VLM
  adjudication, on the first 12 gate images per dataset, was ahead of the best single detector on
  all four datasets, but by noise-level margins on two of them.
- Experience (SEEH). Harvesting 50 images per dataset over 34 configurations (6800 records) gave a
  memory. Using it did not improve results. End-to-end F1 on 40 gate images per dataset, DetAS without experience versus
  experience applied by code per node versus by best joint configuration:
  HazyDet 0.550 / 0.544 / 0.530, BDD night 0.565 / 0.524 / 0.524, BDD rainy 0.652 / 0.663 / 0.663,
  DarkFace 0.214 / 0.203 / 0.203. Putting the evidence in the prompt changed nothing, because the 4B VLM
  ignored it (it chose super-resolution on 16 of 16 test images whichever way the evidence pointed). The
  harvest statistics (per-image F1 on 50 images, cheap fusion) do not predict end-to-end behaviour well
  enough to beat the defaults.
- The adjudicating VLM kept 42 of 50 false groups (and all 9 real ones) in a diagnostic on BDD night,
  so it discriminates poorly; most of its value is cost-free fusion, not judgement.

Video, on 24 synthetic 8-second clips built from labelled BDD frames (slow zoom; some fogged
synthetically), with true object counts taken from the image labels: count exact-match 0.136, mean
absolute error 2.8, box F1 0.567, answer verifier pass rate 0.962, about 22 GPU-seconds per
video-second on an RTX 4060. Counts are mostly too low (37 of 51 label counts under the truth).
The GPU time was measured on the local RTX 4060; the hosted speed and quota were not measured.

## Repository layout

- `src/discern/` the library: `agent/` (nodes, SAIR, MED), `vision/`, `models/` (registry, VRAM
  manager, adapters), `video/`, `index/`, `query/`, `experience/`, `feedback/`, `monitor/`,
  `eval/`, `serve/` (engine and Gradio app), `trace/`, `config/`
- `configs/` model registry, profiles (`hosted_full`, `local_lite`), thresholds
- `scripts/` dataset preparation, baselines, ablations, harvest, gate, Space bundle
- `frontend/` React client for the backend's JSON API (Vercel config in `vercel.json`)
- `space/` Hugging Face Space entrypoint; `legacy/v0/` the frozen first version (baseline only)
- `tests/` CPU tests with fakes; GPU tests carry the `gpu` marker

## Development

```
uv sync                      # core + dev (what CI uses)
uv sync --group gpu --group eval   # local models and evaluation
uv run pytest
uv run ruff check .
uv run mypy
```

Select the profile with `DISCERN_PROFILE=local_lite|hosted_full`. Datasets, weights, MLflow runs and
sessions live under gitignored directories and are never committed.

## Licenses and limits

Code is ours; models keep their own licenses, recorded per entry in `configs/models.yaml`. RIDCP,
MPRNet, Zero-DCE++ (non-commercial), LLFlow and Rex-Omni (research only) restrict use; LLFlow and
Rex-Omni are not part of the public profile. YOLO-World via Ultralytics is AGPL-3.0. Uploaded media
is session-scoped, deleted on a time limit, and kept for feedback only with explicit opt-in.
