# Discern: Typography

Status: v2, 4 October 2026. The user chose Montserrat for all display type on 4 October 2026; this replaces the earlier Bricolage Grotesque choice. Everything about Plex is unchanged. (v1 was October 2026.) Applies to the landing page, the Gradio app (Milestones 4 to 10), and the React frontend (Milestone 11).
Place this file at `docs/typography.md`.

## 1. Decision

| Role | Typeface | Where it appears |
|-|-|-|
| Display | **Montserrat**, one weight (700) | Landing page hero and section headings, the Discern wordmark. Nowhere inside the app |
| Text and UI | **IBM Plex Sans** (variable, weights 400 to 600) | Everything else: app UI, body copy, answers, tables, timestamps, counts |
| Machine output | **IBM Plex Mono** (400) | Only literal machine output: JSON export, trace payloads, model IDs, config values |

All three are free under the SIL Open Font License.

### Why this pairing

Discern's interface is full of small numbers that have to be read precisely: timestamps (00:01:42.360), counts, track IDs, box coordinates, confidence scores. That drives the text face choice.

1. **IBM Plex Sans has tabular figures by default.** I checked the font file directly: all ten digits share one advance width, so timestamps and counts don't shift as they update, and columns align without any extra CSS. Most alternatives, including Inter, Geist and Montserrat itself, ship proportional digits and need tabular figures switched on everywhere, which is a recurring source of jittery counters.
2. **Plex distinguishes I, l, and 1 clearly** (the capital I has slab serifs), and it has a slashed-zero feature. This matters for track IDs and model identifiers.
3. **Plex has a width axis** (75 to 100), so a narrower cut for dense tables and timeline labels comes from the same family rather than a third typeface.
4. **Montserrat gives the landing page a voice the app doesn't need.** A wide geometric sans at weight 700 reads like a title card or a burned-in caption on footage, which is the visual world Discern lives in. It is clearly distinct from Plex, which is the condition for using two families at all. It was chosen by the user; it is much wider than the narrowed display face it replaces, so the display tokens below are tuned smaller and the headline measure is set in ch.
5. **Plex Mono shares Plex's skeleton**, so it sits next to Plex Sans without clashing. It's restricted to literal machine output, so it doesn't count as a stylistic third family.

### What I rejected and why

1. **Inter (the ui-ux-pro-max design-system recommendation).** It's a good typeface, but it's the default of nearly every developer tool and AI dashboard, so it gives Discern no identity. It also needs tabular figures enabled for every numeric element. The skill matched "developer tool, dark, technical" to the most common answer, which is what a generic query returns.
2. **Space Grotesk and Instrument Serif.** Both are now strongly associated with templated AI landing pages.
3. **JetBrains Mono or another mono for small labels.** Monospaced small data labels are one of the most common generic-UI tells. Plex Sans's default tabular figures already give alignment without the mono look.
4. **Geist.** Solid, but strongly tied to Vercel's own brand, and also ships proportional digits by default.
5. **A single-family system (Plex only).** Viable, and the right fallback if the landing page ends up minimal. But the landing page benefits from one expressive element, and Plex at display sizes reads corporate.

## 2. Font files and loading

### Fallback stacks

| Token | Stack |
|-|-|
| `font.display` | "Montserrat", "Montserrat Fallback" (Arial Bold with metric overrides), system-ui, -apple-system, "Segoe UI", Roboto, sans-serif |
| `font.sans` | "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif |
| `font.mono` | "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace |

### Files and budget

Measured from the Google Fonts source files, Latin subset, WOFF2:

| File | Size | Notes |
|-|-|-|
| Montserrat, static 700, Latin (`@fontsource/montserrat` 5.3.0) | 18.8 KB | **Used.** One file for every display use |
| Montserrat, variable wght, Latin (`@fontsource-variable/montserrat` 5.3.0) | 38 KB | Only if a second display weight is truly needed |
| IBM Plex Sans, variable (wdth, wght) | 64 KB | Covers 400, 500, 600 and the narrow width |
| IBM Plex Mono, Regular | 14 KB | Load lazily; it's never above the fold |

### Loading rules

1. Self-host the WOFF2 files rather than calling the Google Fonts CSS API. It's faster, has no third-party request, and makes the files version-pinned like everything else in the project.
2. Montserrat comes as the unmodified Fontsource Latin static 700 file (OFL, no reserved font name issue). No instancing step is needed.
3. For Plex, use pre-subset WOFF2 files from IBM's official `@ibm/plex` package or Fontsource rather than subsetting it yourself. IBM Plex's license may reserve the name "Plex" for modified versions, so check before generating your own subsets. Verify that the variable Plex Sans build with the width axis is available in whichever package you pick.
4. Preload only two files: Plex Sans (every page) and the Montserrat file (landing page only). Never preload the mono.
5. Use font-display swap for all three. For Montserrat, add a metric-matched fallback ("Montserrat Fallback": Arial Bold with size-adjust 109.7%, ascent-override 88.2%, descent-override 22.9%) so the hero doesn't jump when the font arrives. Landing payload: Plex Sans 64 KB plus Montserrat 18.8 KB, about 84 KB; the mono loads only when machine output is on screen.
6. Specify sizes in rem so browser zoom and user font-size settings work.

## 3. Type scale

The steps follow the classical scale from *The Elements of Typographic Style* (12, 14, 16, 18, 21, 24, 36, 48, 72), with fluid interpolation for the two display roles. Fluid ranges run from a 375px to a 1440px viewport.

| Token | Family | Size | Weight | Line height | Tracking | Use |
|-|-|-|-|-|-|-|
| `type.display.xl` | display | 36 to 68px, fluid: clamp(2.25rem, 1.5rem + 3.6vw, 4.25rem) | 700 | 1.05 | -0.02em | Landing hero headline only. One per page |
| `type.display.l` | display | 32 to 48px, fluid: clamp(2rem, 1.65rem + 1.5vw, 3rem) | 700 | 1.1 | 0 | Landing section headings |
| `type.h1` | sans | 24px (1.5rem) | 600 | 1.25 | 0 | App page and panel titles |
| `type.h2` | sans | 21px (1.3125rem) | 600 | 1.3 | 0 | Sub-sections, the answer heading in the chat |
| `type.h3` | sans | 18px (1.125rem) | 600 | 1.35 | 0 | Card and group titles (shot, track) |
| `type.body.l` | sans | 18px (1.125rem) | 400 | 1.6 | 0 | Landing paragraphs, the answer text in chat |
| `type.body` | sans | 16px (1rem) | 400 | 1.55 | 0 | Default app text. Never smaller on mobile |
| `type.body.s` | sans | 14px (0.875rem) | 400 | 1.45 | 0 | Dense panels: trace, evidence lists, tables |
| `type.label` | sans | 14px (0.875rem) | 500 | 1.3 | 0 | Form labels, buttons, tabs. Sentence case |
| `type.caption` | sans | 12px (0.75rem) | 400 | 1.4 | +0.01em | Timestamps under thumbnails, axis ticks. The floor: nothing smaller |
| `type.data.narrow` | sans, width 85 | 14px (0.875rem) | 400 or 500 | 1.3 | 0 | Table columns with many numbers (box coordinates, per-frame scores) |
| `type.code` | mono | 14px (0.875rem) | 400 | 1.5 | 0 | JSON, trace payloads, model IDs |

Weight hierarchy uses only three steps: 400 (reading), 500 (interactive labels), 600 (headings). Montserrat uses only 700. Nothing uses 300 or lighter: thin weights break down on dark backgrounds and in degraded video thumbnails.

## 4. Numbers, time, and data

Numbers are the product's evidence. Treat them as first-class type.

1. **Tabular figures everywhere in the app.** Plex Sans does this by default. If any other face displays numbers (for example Montserrat in a landing-page figure), enable tabular figures explicitly.
2. **Slashed zero for identifiers only.** Turn on the zero feature for track IDs, model IDs, and hashes, where 0 and O can be confused. Leave it off for timestamps and counts, where it adds noise.
3. **One timestamp format.** mm:ss for clips under an hour, h:mm:ss above. Add milliseconds (mm:ss.mmm) only in the trace and in exports. Never mix formats on one screen.
4. **Counts are written into sentences, not blown up.** "3 cars passed between 0:10 and 0:20," with the number at body size and weight 600. Avoid the oversized-number-with-small-label treatment; it's a template default and it detaches the number from its evidence.
5. **Units and confidence.** Write confidence as a percentage with no decimals in the UI (87%). Show raw scores only in the trace.
6. **Right-align numeric table columns**; left-align text columns.

## 5. Landing page

1. **The hero headline is the one bold element on the page.** Set it in `type.display.xl`, left-aligned, at most two lines on desktop and three on mobile. Everything around it stays quiet: Plex Sans body, generous space, no competing effects.
2. **One orchestrated moment, tied to the product.** The headline renders as if seen through degradation (a brief blur and grain) and resolves to sharp once, on load, in about 500ms. This mirrors what Discern does to footage, so it's grounded in the subject rather than decoration. Requirements:
   - The text stays real, selectable DOM text throughout.
   - It is fully readable even if the animation never runs.
   - With reduced motion enabled, it renders sharp immediately.
   - It runs once and never loops.
   - No other entrance animations elsewhere on the page.
3. **Pair the hero with real output.** A real before/after frame processed by Discern beside or below the headline does more than any type treatment. No fabricated results; this rule is already in `CLAUDE.md`.
4. **Section headings** use `type.display.l`. Body paragraphs use `type.body.l` with a maximum measure of 68 characters (68ch).
5. **No eyebrow labels** above headings, **no all-caps labels**, and **no single word accented** in a headline by color, italic, or weight. These are the most common tells of generated landing pages.

## 6. The app (Gradio now, React later)

1. **The app uses Plex only.** Montserrat appears only in the wordmark in the header. Inside the working interface, type should recede so the footage and evidence are the focus.
2. **Gradio:** set the theme's primary font to IBM Plex Sans and its mono font to IBM Plex Mono, both with the fallback stacks above. Check that Gradio's default text sizes map to this scale; override the theme's text size settings so body text is 16px, not smaller.
3. **Chat answers** use `type.body.l` for the answer sentence, `type.h2` for the answer heading if one is needed, and `type.body.s` for the evidence list beneath (track, time range, confidence).
4. **Ungrounded answers** (describe queries) are marked with a text label in `type.label` ("Not grounded in detections"), never by color or font style alone.
5. **Trace view** uses `type.body.s` for node names and decisions, `type.code` for raw inputs and outputs.
6. **React frontend:** expose the tokens in sections 2 and 3 as CSS custom properties and Tailwind theme values. Components reference tokens, never raw font names or pixel sizes.

## 7. Dark mode

The UI is dark-first, which suits the product, since users are looking at video, often night footage.

1. Light text on dark backgrounds looks heavier than the same weight on light. Evaluate body text at weight 400 against a slightly lighter setting (around 380, available on the variable Plex Sans) and choose by eye on a real screen. Don't change the scale.
2. Primary text contrast at least 4.5:1 against its surface; secondary text at least 4.5:1 too, since it carries timestamps and counts. Don't use low-contrast grey for numbers.
3. Pure white text on near-black causes halation for some readers. Use an off-white foreground, but verify contrast numerically.
4. Text drawn over video (box labels, burned-in timestamps) uses Plex Sans 500 at a minimum of 12px on a solid or high-opacity backing, never directly on the footage.

## 8. Writing style in type

1. Sentence case for every heading, button, tab, and label.
2. Buttons say what happens: "Upload video," "Ask," "Download annotated video." The same action keeps the same name throughout ("Export" produces "Exported").
3. Errors say what went wrong and how to fix it, without apology ("Video is longer than 60 seconds. Trim it or run Discern locally.").
4. No arrows appended to links or buttons, no middle-dot meta strings, no em dashes in UI copy.

## 9. Verification checklist

Before any page or screen ships:

- [ ] Only the three families above are loaded, and Plex Mono loads lazily.
- [ ] Total font payload on the landing page is under about 100 KB.
- [ ] No text smaller than 12px; body text at least 16px on mobile.
- [ ] Hero shows no layout shift when the display font loads (check with Lighthouse).
- [ ] Hero renders sharp immediately with reduced motion enabled.
- [ ] A counter or timestamp updating in place doesn't jitter horizontally.
- [ ] Track and model IDs use slashed zero; timestamps don't.
- [ ] Paragraph measure stays at or under 68 characters at every breakpoint.
- [ ] Text contrast at least 4.5:1 for all text, measured in dark mode.
- [ ] No all-caps labels, eyebrow labels, single-word headline accents, or mono-styled small labels.
- [ ] Page still reads correctly at 200% browser zoom.
