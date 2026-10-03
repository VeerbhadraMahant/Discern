---
title: Discern
emoji: 🔎
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app.py
hf_oauth: true
pinned: false
---

# Discern

Discern cleans a degraded image or video (fog, rain, low light, noise) with restoration models
chosen per scene, then answers questions about it. Answers that find, count or time objects point
to boxes, tracks and timestamps you can inspect, and every step is listed in the Trace tab.

This is a research demo built on open models. It can be wrong: check the evidence shown next to
each answer. Descriptive answers are labelled as ungrounded.

## Privacy

Uploads live in a temporary session directory and are deleted when the session expires. Media is
kept for training only if you tick the opt-in box when sending feedback. The About tab states the
exact retention period and the limits that apply.

## Licenses

Some restoration weights used here are licensed for non-commercial use only, so this demo is
non-commercial. The About tab lists them with their licenses, read from the model registry.

Sign in with Hugging Face so GPU time is counted against your own quota.
