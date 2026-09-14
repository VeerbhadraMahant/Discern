# Discern

An agentic vision system for safety monitoring on high-risk sites (construction, mining, ports, industrial plants). Instead of running one fixed detection model regardless of conditions, Discern's agent inspects each frame, classifies the scene (lighting, weather, visibility, setting), and dynamically selects which restoration steps and detection tools to run before reasoning about PPE and restricted-zone violations.

## Pipeline

For each uploaded frame:

1. **Scene understanding** — Claude (vision) classifies lighting, weather, visibility, setting, and presence of people/machinery.
2. **Tool selection** — the same call selects which restoration and detection tools are worth running, and explains why.
3. **Restoration** (conditional) — OpenCV-based low-light enhancement (CLAHE + gamma), dehazing (dark-channel prior), or denoising, only when selected.
4. **Person detection** (conditional) — YOLOv8n locates people in the (restored) frame.
5. **Violation reasoning** (conditional) — Claude inspects the frame plus detected people and reports missing-PPE / restricted-zone / unsafe-proximity violations.

The frontend renders the agent's trace, the annotated frame, and the violations found.

## Setup

### Backend

```bash
cd backend
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash; use venv\Scripts\activate.bat on cmd
pip install -r requirements.txt
cp .env.example .env           # then set ANTHROPIC_API_KEY
uvicorn app.main:app --reload --port 8000
```

The first run downloads `yolov8n.pt` automatically (~6MB).

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 — API calls are proxied to `http://localhost:8000`.

## Notes

- Requires an `ANTHROPIC_API_KEY` (Claude vision + tool use) for the scene-understanding and violation-reasoning steps.
- Person detection uses the standard pretrained YOLOv8n COCO weights; PPE compliance and restricted-zone judgment are done by Claude's vision reasoning over the detected people, since COCO has no PPE/hazard-zone classes.
- Restoration tools are lightweight classical CV (no extra model downloads) so the agent can apply them conditionally with no added latency budget for model loading.
