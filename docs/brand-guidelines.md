# Discern: Brand guidelines

Status: v1, October 2026. Source of truth for voice, messaging and visual identity. Typography is fixed by
`docs/typography.md`; the working palette and component language by `frontend/design-system/MASTER.md`.

## 1. Brand core

**What Discern is.** A video question-answering agent that cleans degraded footage (fog, night, rain, noise),
finds and tracks what is in it, and answers in plain language with evidence you can inspect: boxes, tracks and
timestamps. Built on open models, runnable on an 8 GB laptop GPU.

**Name.** To discern is to perceive clearly and tell things apart. The name is the promise: clear sight, then judgement.

**Mission.** We help people get answers from poor footage that they can check for themselves, by cleaning the image,
detecting with several models, and showing the evidence behind every claim.

**Positioning.** Discern is the video question-answering tool for people who have to be right about what a clip shows,
because every number it states is computed by code and checked, and every claim points at something you can see.

**Differentiator, in one line.** Other tools answer confidently. Discern answers with its evidence and tells you where
it is unsure.

**Promise (primary message).** Ask a video a question and see the evidence.

**Proof points (all measured in this repository, never borrowed):**
1. Answers are verified: every number and timestamp is checked against computed facts before it is shown (verifier pass rate 0.962 on 24 test clips).
2. Fusing the best two detectors matches the best single one; restoration is applied only when it helps, because always restoring lowered F1 on hazy and night images.
3. Open models only, 8 GB GPU, uploads deleted on a time limit, retained only with explicit opt-in.
4. It publishes its failures: experience memory did not help, counts run low, GPU cost is high, the hosted Space is not deployed.

## 2. Audience

Primary: engineers, researchers and analysts who work with imperfect footage and must justify an answer (traffic and
site review, wildlife and field cameras, drone and dashcam, research on detection under degradation).
Secondary: reviewers and hiring managers judging engineering quality. They need to see rigor quickly.
Pain: video models give fluent answers that cannot be audited, and fall apart on fog, rain and night.

## 3. Voice

**Personality:** exact, candid, calm, a little dry.

| Trait | Means | Do | Don't |
|-|-|-|-|
| Exact | Specific numbers, named sources | "3 cars between 0:10 and 0:20." | "Lots of vehicles." |
| Candid | States limits and failures plainly | "Restoration often lowers F1. We only apply it when it helps." | "Industry-leading accuracy." |
| Calm | No hype, no exclamation marks | "Upload a clip and ask." | "Revolutionize your video workflow!" |
| Dry | A little understated wit, never jokes at the user's expense | "It shows its work, which is more than most of us do." | Puns, slang, emoji |

**Rules.** Sentence case. Short sentences. Verbs for buttons ("Upload video", "Ask"). Numbers always carry their source.
No superlatives, no "AI-powered", no "seamless", no "unlock", no "revolutionary". No em dashes, no arrows on links,
no middle-dot meta strings. Never claim anything the repository has not measured; label examples as examples.

**Rewrites.**
- Before: "Unlock powerful insights from your video with AI." After: "Ask a video a question and see the evidence."
- Before: "State-of-the-art accuracy." After: "Measured here: 0.664 F1 on hazy drone frames with one open detector."
- Before: "Seamless privacy." After: "Uploads are deleted after a set time. Nothing is kept unless you opt in."

## 4. Messaging hierarchy (landing page)

1. **Headline (promise):** Ask a video a question and see the evidence.
2. **Subline (how, one sentence):** Discern cleans fog, night and rain from each shot, finds and tracks what is there, and answers with boxes and timestamps you can check.
3. **Three pillars** (each one image, one sentence, one proof):
   - *Clean sight:* it restores a shot only when restoring helps, and says when it chose not to.
   - *Shown evidence:* every answer points at tracks, crops and timestamps.
   - *Checked numbers:* a verifier compares each number in an answer with computed facts before showing it.
4. **Proof strip:** four measured figures written in sentences.
5. **Honesty block:** what did not work.
6. **Closing call:** Open the app, or try it with demo data. Run it locally with three commands.

## 5. Visual identity

**Direction: the evidence frame.** Discern looks like a careful photo desk meets a forensic viewfinder: warm paper,
black ink, one ember mark for "found". Quiet surfaces, precise frames, no gloss.

**Palette** (unchanged, see MASTER.md): parchment, bone, ink, ember (marks and focus only, never text).
Add one rule: **ember means "evidence is here"**. It marks a found object, a stamp, a focus ring, never decoration.

**Signature motif 1, the frame brackets.** Four short corner ticks (like a viewfinder or a contact-sheet crop mark)
around anything that is evidence: a detected object, a result figure, the hero visual. Used consistently as the
product's mark of "this was found and can be inspected".

**Signature motif 2, the focus pull.** Degraded to sharp. Used once on the hero headline (see typography.md) and as an
interactive compare slider in the hero visual (the user drags from fog to clean). Never looped.

**Signature motif 3, the contact sheet.** A horizontal filmstrip of frames with timecodes beneath, used to show a clip
through the pipeline: degraded, cleaned, detected, answered. Frames are drawn in SVG (no dataset photos on the page).

**Wordmark.** "Discern" in Montserrat 700, sentence case, with the frame-bracket mark before it:
a square drawn as four corner ticks with one ember dot at its centre ("the thing found"). The mark works alone at 16 px
(favicon) as ticks plus dot. Clear space equals the height of the "D". Minimum width 96 px for the full wordmark.
One colour at a time: ink on parchment or bone, parchment on ink; the ember dot stays ember on both.

**Illustration.** Line drawings in ink with hatch, stipple and dash patterns (pattern carries meaning, so colour is
never the only signal). Flat, no gradients, no blur, no shadows except the card shadow. Scenes: road, drone view,
night street, rain, built from simple shapes. Every illustration is labelled as an illustration, never as a result.

**UI imagery.** Show the real product: mock-ups built from the actual app components (track card, evidence panel,
trace row, chat answer) filled with clearly labelled demo data.

**Iconography.** Phosphor regular only, 16, 20 and 24 px.

**Layout.** Broadsheet grid, 12 columns, generous gutters, alternating parchment and ink bands for rhythm. Display
type is large and left-aligned. One idea per band.

## 6. What the brand is not

Not an "AI magic" product. Not a dashboard of vanity numbers. Not dark-neon. Not cute. Not a template: no gradient
blobs, no glass cards, no purple, no three-icons-in-a-row without a reason.

## 7. Consistency checklist

- [ ] Promise and subline match section 4 on every surface.
- [ ] Every figure has a source and appears in `frontend/src/landing/facts.ts`.
- [ ] Ember used only for evidence marks, focus rings and stamps.
- [ ] Frame brackets used for evidence only.
- [ ] Fonts only as in typography.md; sentence case; no all-caps labels.
- [ ] Examples carry the "Example, not a recorded result" label.
- [ ] Voice test: would a competitor say this sentence? If yes, rewrite.
