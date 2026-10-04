"""Gradio UI. The only module that imports gradio; it is not imported by the tests.

Handlers hold no state of their own: the browser keeps session ids in `gr.State`, and everything
else lives in the `Engine`. Errors are shown as messages, never as stack traces.

The same app also serves the React frontend's JSON API (`discern.serve.api`, contract in
frontend/API.md) under the api names in `API_NAMES`. Launch it with `launch_kwargs(engine)`:
`allowed_paths` is the session directory, the only place the returned `*_url` files live, and
`strict_cors=False` (opt in with `DISCERN_ALLOW_ANY_ORIGIN=1`) lets a frontend on another origin
call the app from the browser; by default Gradio accepts only same-origin and localhost origins.
On a Hugging Face Space the app listens on 7860 and the Space URL is the Gradio root the client
connects to; ZeroGPU quota is the calling visitor's when the browser sends their HF token.
"""

import logging
import os
from collections.abc import Callable
from typing import Any

import gradio as gr

from discern.agent.schemas import ShotPlan
from discern.feedback.schema import Feedback
from discern.serve.api import DiscernApi
from discern.serve.engine import (
    TRACE_COLUMNS,
    CleanResult,
    Engine,
    EngineError,
    Evidence,
    build_engine,
    trace_rows,
)
from discern.serve.gpu import on_space
from discern.serve.session import ALLOWED_EXTENSIONS

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset({"jpg", "jpeg", "png", "webp"})
FILE_TYPES = [f".{e}" for e in sorted(ALLOWED_EXTENSIONS)]
UNGROUNDED_LABEL = "Ungrounded description: not backed by detections or tracks."
RATIONALE_NODES = ("perception", "restorer_select", "image_select", "sr_select")
JUMP_JS = (
    "(t) => { const v = document.querySelector('#player video');"
    " if (v && t !== null) { v.currentTime = Number(t); v.play(); } }"
)
USER_ERRORS = (EngineError, ValueError, KeyError, OSError)
API_NAMES = (
    "discern_info",
    "discern_upload",
    "discern_clean",
    "discern_detect",
    "discern_ingest",
    "discern_ask",
    "discern_annotated_video",
    "discern_trace",
    "discern_feedback",
    "discern_cleanup",
)
ANY_ORIGIN_ENV_VAR = "DISCERN_ALLOW_ANY_ORIGIN"


def _guard(n_outputs: int) -> Callable[[Callable[..., tuple[Any, ...]]], Callable[..., Any]]:
    """Turn any failure into a message in the last output; other outputs stay as they are."""

    def decorate(fn: Callable[..., tuple[Any, ...]]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except USER_ERRORS as err:
                logger.info("request refused: %s", err)
                message = f"Could not complete the request: {err}"
            except Exception:
                logger.exception("request failed")
                message = "Something went wrong while processing this request. Please try again."
            return (*([gr.skip()] * (n_outputs - 1)), message)

        wrapper.__name__ = fn.__name__
        return wrapper

    return decorate


def _extension(path: str) -> str:
    return path.rsplit(".", 1)[-1].lower() if "." in path else ""


def _decision_markdown(plan: ShotPlan, events: Any) -> str:
    rationale = {e.node: e for e in events if e.node in RATIONALE_NODES}
    lines = ["**Cleaning decisions**", ""]
    lines += [f"- {d}" for d in plan.decisions]
    lines += ["", "**Why**", ""]
    for node in RATIONALE_NODES:
        if node in rationale and rationale[node].rationale:
            lines.append(f"- {node}: {rationale[node].rationale}")
    return "\n".join(lines)


def _evidence_views(evidence: Evidence) -> tuple[list[Any], list[list[Any]], Any]:
    gallery = [
        (t.crop, f"#{t.track_id} {t.label} {t.t_start:.1f}s to {t.t_end:.1f}s")
        for t in evidence.tracks
        if t.crop is not None
    ]
    table = [[t.track_id, t.label, round(t.t_start, 2), round(t.t_end, 2)] for t in evidence.tracks]
    choices = [(f"#{t.track_id} {t.label} at {t.t_start:.1f}s", t.t_start) for t in evidence.tracks]
    return gallery, table, gr.update(choices=choices, value=choices[0][1] if choices else None)


def build_app(engine: Engine) -> gr.Blocks:
    s = engine.settings.thresholds.serve
    # Gradio copies uploads and returned images into its own cache; expire them with the session.
    expiry = (int(s.cleanup_interval_seconds), int(s.ttl_seconds))
    with gr.Blocks(title="Discern", delete_cache=expiry) as demo:
        gr.Markdown("# Discern\nClean a degraded image or video, then ask questions about it.")
        gr.Markdown(engine.limits_markdown())
        if on_space():
            with gr.Row():
                gr.LoginButton()
                who = gr.Markdown()
            demo.load(_whoami, None, who)
        image_sid = gr.State(None)
        video_sid = gr.State(None)
        last_result = gr.State(None)

        with gr.Tabs():
            with gr.Tab("Upload and Clean"):
                upload = gr.File(label="Image or video", file_types=FILE_TYPES, type="filepath")
                clean = gr.Button("Clean", variant="primary")
                clean_status = gr.Markdown()
                with gr.Row():
                    before = gr.Image(label="Before", interactive=False)
                    after = gr.Image(label="After", interactive=False)
                plan_md = gr.Markdown()
                shots = gr.Dataframe(
                    headers=["shot", "start s", "end s", "decisions"], label="Shots", visible=False
                )
                previews = gr.Gallery(label="Shot keyframes: before, then after", visible=False)
                with gr.Row():
                    targets = gr.Textbox(
                        label="Targets to find (images)", placeholder="car, person"
                    )
                    detect = gr.Button("Detect")
                annotated = gr.Image(label="Detections", interactive=False)
                detections = gr.Dataframe(headers=["label", "score", "x1", "y1", "x2", "y2"])

            with gr.Tab("Ask"):
                chat = gr.Chatbot(label="Conversation")
                with gr.Row():
                    question = gr.Textbox(
                        label="Question", placeholder="How many cars are there?", scale=4
                    )
                    send = gr.Button("Ask", variant="primary", scale=1)
                ask_status = gr.Markdown(f"Up to {s.max_queries_per_session} questions per video.")
                grounding = gr.Markdown()
                with gr.Row():
                    crops = gr.Gallery(label="Evidence crops")
                    evidence_table = gr.Dataframe(
                        headers=["track", "label", "start s", "end s"], label="Evidence tracks"
                    )
                with gr.Row():
                    jump_to = gr.Dropdown(label="Jump to", choices=[])
                    jump = gr.Button("Jump")
                    render = gr.Button("Show annotated video")
                player = gr.Video(label="Video", elem_id="player", interactive=False)

            with gr.Tab("Trace"):
                refresh_trace = gr.Button("Refresh")
                trace_table = gr.Dataframe(headers=TRACE_COLUMNS, label="Trace of this session")

            with gr.Tab("Feedback"):
                gr.Markdown(
                    "Tell us whether a result was right. By default only your verdict, label "
                    "and box are stored; your media is stored only if you tick the box below."
                )
                fb_result = gr.Textbox(label="Result id", placeholder="R1 (video) or image")
                fb_verdict = gr.Radio(["correct", "wrong", "missing"], label="Verdict")
                fb_track = gr.Number(label="Track id (video, optional)", precision=0)
                fb_label = gr.Textbox(label="Object label")
                fb_box = gr.Textbox(
                    label="Box x1,y1,x2,y2 in original pixels (required for missing)"
                )
                fb_note = gr.Textbox(label="Note")
                fb_retain = gr.Checkbox(
                    label="Keep my media so it can be used for training (opt-in)", value=False
                )
                fb_send = gr.Button("Send feedback")
                fb_status = gr.Markdown()

            with gr.Tab("About"):
                refresh_about = gr.Button("Refresh")
                about = gr.Markdown(engine.about_markdown())
                monitor = gr.Markdown(engine.monitoring_markdown())

        # ---- handlers ----------------------------------------------------------------------

        @_guard(10)
        def on_clean(path: str | None, progress: Any = gr.Progress()) -> tuple[Any, ...]:  # noqa: B008
            if not path:
                raise EngineError("choose an image or a video first")
            if _extension(path) in IMAGE_EXTENSIONS:
                sid, stored = engine.register_upload(path)
                result = engine.clean_image(stored, sid)
                return (
                    sid,
                    None,
                    result.original,
                    result.cleaned,
                    _decision_markdown(result.plan, result.events),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    None,
                    None,
                    "Image cleaned. Enter targets below to detect them.",
                )
            session = engine.ingest(path, lambda f, m: progress(f, desc=m))
            rows = [
                [sh.id, round(sh.t_start, 2), round(sh.t_end, 2), "; ".join(plan.decisions)]
                for sh in session.ingest.shots
                for plan in [session.ingest.plans[sh.id]]
            ]
            pairs = engine.shot_previews(session)
            gallery = [
                (img, f"shot {sh.id} {kind}")
                for sh, (b, a) in zip(session.ingest.shots, pairs, strict=False)
                for kind, img in (("before", b), ("after", a))
            ]
            return (
                None,
                session.session_id,
                pairs[0][0] if pairs else None,
                pairs[0][1] if pairs else None,
                f"{len(rows)} shot(s). The decisions per shot are in the table.",
                gr.update(value=rows, visible=True),
                gr.update(value=gallery, visible=True),
                None,
                None,
                "Video ready. Open the Ask tab.",
            )

        clean.click(
            on_clean,
            upload,
            [
                image_sid,
                video_sid,
                before,
                after,
                plan_md,
                shots,
                previews,
                chat,
                last_result,
                clean_status,
            ],
        )

        @_guard(4)
        def on_detect(sid: str | None, text: str) -> tuple[Any, ...]:
            result: CleanResult = engine.clean_result(sid)
            out = engine.detect_targets(result, text.split(","))
            rows = [
                [d.label, round(d.score, 3), *(round(v, 1) for v in d.box)] for d in out.detections
            ]
            return out.annotated, rows, "image", f"{len(rows)} detection(s)."

        detect.click(
            on_detect, [image_sid, targets], [annotated, detections, last_result, clean_status]
        )

        @_guard(8)
        def on_ask(
            sid: str | None, text: str, history: list[dict[str, Any]] | None
        ) -> tuple[Any, ...]:
            session = engine.session(sid)
            answer, evidence = engine.ask(session, text)
            gallery, table, choices = _evidence_views(evidence)
            shown = answer if evidence.grounded else f"{UNGROUNDED_LABEL}\n\n{answer}"
            history = [
                *(history or []),
                {"role": "user", "content": text},
                {"role": "assistant", "content": shown},
            ]
            used = f"{session.queries_asked} of {s.max_queries_per_session} questions used."
            label = "" if evidence.grounded else f"**{UNGROUNDED_LABEL}**"
            return history, "", gallery, table, choices, evidence.result_set_id, label, used

        ask_outputs = [
            chat,
            question,
            crops,
            evidence_table,
            jump_to,
            last_result,
            grounding,
            ask_status,
        ]
        send.click(on_ask, [video_sid, question, chat], ask_outputs)
        question.submit(on_ask, [video_sid, question, chat], ask_outputs)
        jump.click(None, jump_to, None, js=JUMP_JS)

        @_guard(2)
        def on_render(sid: str | None, result_id: str | None) -> tuple[Any, ...]:
            session = engine.session(sid)
            if not result_id or result_id not in session.conversation.result_sets:
                raise EngineError("ask a question first")
            return str(engine.annotated_video(session, result_id)), ""

        render.click(on_render, [video_sid, last_result], [player, ask_status])

        def on_trace(image: str | None, video: str | None) -> list[list[Any]]:
            events = engine.trace_events(video) + engine.trace_events(image)
            return trace_rows(events)

        refresh_trace.click(on_trace, [image_sid, video_sid], trace_table)

        @_guard(1)
        def on_feedback(
            image: str | None,
            video: str | None,
            result: str,
            verdict: str,
            track: float | None,
            label: str,
            box: str,
            note: str,
            retain: bool,
        ) -> tuple[Any, ...]:
            parsed = tuple(float(v) for v in box.split(",")) if box.strip() else None
            if parsed is not None and len(parsed) != 4:
                raise EngineError("the box needs four numbers: x1,y1,x2,y2")
            feedback = Feedback(
                result_set_id=result.strip(),
                verdict=verdict,
                track_id=None if track is None else int(track),
                box=parsed,
                label=label.strip(),
                note=note.strip(),
                retain_media=retain,
            )
            stored = engine.submit_feedback(video or image, feedback)
            kept = "with your media" if stored.media_file else "without media"
            return (f"Thank you. Feedback stored {kept}.",)

        fb_send.click(
            on_feedback,
            [
                image_sid,
                video_sid,
                fb_result,
                fb_verdict,
                fb_track,
                fb_label,
                fb_box,
                fb_note,
                fb_retain,
            ],
            fb_status,
        )

        def on_about() -> tuple[str, str]:
            return engine.about_markdown(), engine.monitoring_markdown()

        refresh_about.click(on_about, None, [about, monitor])
        timer = gr.Timer(s.cleanup_interval_seconds)
        timer.tick(engine.cleanup)
        _register_api(DiscernApi(engine))
    return demo


def _register_api(api: DiscernApi) -> None:
    """API-only endpoints for the React frontend. The upload takes a file, so it goes through a
    hidden File component (which makes the client upload the file first); the rest are typed
    functions that need no components."""
    upload_file = gr.File(type="filepath", visible=False)
    upload_result = gr.JSON(visible=False)
    gr.Button(visible=False).click(
        api.upload, upload_file, upload_result, api_name="discern_upload"
    )
    gr.api(api.info, api_name="discern_info")
    gr.api(api.clean, api_name="discern_clean")
    gr.api(api.detect, api_name="discern_detect")
    gr.api(api.ingest, api_name="discern_ingest")
    gr.api(api.ask, api_name="discern_ask")
    gr.api(api.annotated_video, api_name="discern_annotated_video")
    gr.api(api.trace, api_name="discern_trace")
    gr.api(api.feedback, api_name="discern_feedback")
    gr.api(api.cleanup, api_name="discern_cleanup")


def launch_kwargs(engine: Engine) -> dict[str, Any]:
    """`launch(...)` options for the API: serve files from the session directory only, and accept
    browser calls from any origin when `DISCERN_ALLOW_ANY_ORIGIN` is set to 1."""
    return {
        "allowed_paths": [str(engine.store.root.resolve())],
        "strict_cors": os.environ.get(ANY_ORIGIN_ENV_VAR) != "1",
    }


def _whoami(profile: gr.OAuthProfile | None) -> str:
    if profile is None:
        return "Not signed in. Sign in so GPU time is counted against your own quota."
    return f"Signed in as {profile.username}."


def main() -> None:
    """Launch the app for the profile named by DISCERN_PROFILE (default local_lite)."""
    engine = build_engine()
    if on_space():
        engine.warm_up()
    build_app(engine).queue().launch(**launch_kwargs(engine))


if __name__ == "__main__":
    main()
