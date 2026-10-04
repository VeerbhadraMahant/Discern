# Discern frontend API contract

The browser talks to the Gradio app (a local `python space/app.py`, or the Hugging Face Space) with the
Gradio JS client (`@gradio/client`): `client.predict("/discern_x", {...})` for one-shot calls and
`client.submit("/discern_ingest", {...})` (an async iterator) for progress. The backend exposes these
endpoints with explicit `api_name`s from `src/discern/serve/api.py`, registered by `gradio_app.py`.
All responses are JSON objects; media are returned as URLs the browser can load. Calls are made from the
visitor's browser, so the visitor's own ZeroGPU quota applies.

Errors: every endpoint returns `{ "ok": false, "error": "<human readable message>" }` instead of raising;
success responses have `"ok": true`.

## Endpoints

1. `/discern_info` () -> `{ok, profile, limits:{max_upload_mb, max_video_seconds, max_queries, ttl_seconds, max_pixels}, models:[{role, name, license}], measured_gpu_seconds: number|null, memory_version: string|null}`
2. `/discern_upload` (file) -> `{ok, session_id, kind:"image"|"video", name, size_bytes}`
3. `/discern_clean` (session_id) -> `{ok, original_url, cleaned_url, profile:{scene_label, illumination, visibility, object_scale, object_density, confidence}, plan:{restorer, use_restored, sr_factor|null, decisions:string[]}, events:TraceEvent[]}`
4. `/discern_detect` (session_id, targets: string such as "car, person") -> `{ok, annotated_url, detections:[{label, score, box:[x1,y1,x2,y2], detector}], events}`
5. `/discern_ingest` (session_id), a generator: yields `{ok:true, stage:"progress", progress:0..1, message}` and then a final `{ok:true, stage:"done", duration, fps, width, height, shots:[{id, t_start, t_end, profile, plan, before_url, after_url}], video_url, events}`
6. `/discern_ask` (session_id, question) -> `{ok, answer, grounded: boolean, clarification: boolean, result_set_id|null, evidence:{tracks:[{id, label, t_start, t_end, n_frames, mean_score, status:"accepted"|"rejected", rationale, crop_url|null}], summary:string}, events}`
7. `/discern_annotated_video` (session_id, result_set_id) -> `{ok, video_url}`
8. `/discern_trace` (session_id) -> `{ok, events:TraceEvent[]}`
9. `/discern_feedback` (session_id, result_set_id, track_id: number|null, verdict:"correct"|"wrong"|"missing", note: string, retain_media: boolean) -> `{ok}`
10. `/discern_cleanup` (session_id) -> `{ok}` (delete the visitor's session now)

`TraceEvent` = `{node, started_at, duration_ms, gpu_ms, prompt_version|null, input_summary, decision, rationale, fallback_used}`.
`retain_media` defaults to false in the UI and is sent as true only after an explicit checkbox.
