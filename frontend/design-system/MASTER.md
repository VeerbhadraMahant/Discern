# Discern frontend: design system (Master)

Source of truth for the visual language. Reference style: Refero "Miranda"
(https://styles.refero.design/style/3f6e3076-e77f-487e-b212-3b5946a34e87), an old-world broadsheet on
warm parchment: near-black ink, one ember accent, serif display type with tight tracking, flat
borderless components, sharp geometry, no gradients. Quality rules come from the ui-ux-pro-max skill
(accessibility, touch, loading states, motion, responsive). Light theme only, as the reference is.

## Tokens (Tailwind v4 `@theme`)

Colors (never use raw hex in components, only these tokens):
- `parchment` #e2dedb page background
- `bone` #cdc6be card and panel surface, inset areas
- `ink` #1d1d1b text, borders, icon strokes, banner fills
- `charcoal` #69645f decorative only. Measured contrast on parchment is 4.38:1 (below the 4.5:1 text minimum),
  so it is never used for text; secondary text is ink.
- `black` #000000 emphasis outlines
- `ember` #c03f13 the ONLY chromatic accent: focus ring, nav underline, stamp borders, warning icons. Measured
  contrast against parchment is 3.96:1 (fine for large UI marks and borders, not for text), so text on or in ember is ink.
- State is never color alone: success = ink check icon + word, error = ember border + warning icon + message.

Type: superseded by [docs/typography.md](../../docs/typography.md) (the user's typography spec). Display is Montserrat
(static 700, Latin, from Fontsource) for the landing hero, landing section headings and the Discern wordmark only; text
and UI are IBM Plex Sans (variable, 400 to 600); literal machine output is IBM Plex Mono 400. Fonts are self-hosted WOFF2 in
`public/fonts` (see `public/fonts/LICENSES.md`). Tokens live in `src/index.css` under `@theme`: `--font-display`, `--font-sans`,
`--font-mono`, and the scale `--text-display-xl`, `--text-display-l`, `--text-wordmark`, `--text-h1`, `--text-h2`, `--text-h3`,
`--text-body-l`, `--text-body`, `--text-body-s`, `--text-label`, `--text-caption`, `--text-data-narrow`, `--text-code`, each with
`--line-height`, `--letter-spacing` and `--font-weight` companions. Components use the matching `type-*` utilities
(for example `type-h1`, `type-body-l`, `type-code`) and never raw font names or pixel sizes. Rules in force: sentence case
everywhere, no all-caps labels, no eyebrow labels, one timestamp format (m:ss, h:mm:ss from an hour; milliseconds only in the trace),
confidence as whole-number percentages outside the trace, counts written into sentences, right-aligned numeric columns,
nothing under 12px or lighter than 400. Color stays the light Refero palette; every text pair is at least 4.5:1.

Space: 4px base; scale 4, 7, 10, 12, 14, 17, 22, 28, 36, 43, 58, 65. Section gap 43, card padding 24,
element gap 14. Page max-width 1440.
Radius: tags and buttons 2.88px, cards 11.52px, images 0. Nothing above 12px.
Shadow: only on cards, the directional ink shadow `rgba(29,29,27,.2) -4px 4px 6px 0`. No other shadows, no gradients, no blur.

## Components

- Header bar: parchment, 1px ink bottom border, Discern wordmark (mark plus word in Montserrat, links to `#/`) at the left edge of the shared `page` container, profile, connection status and Start over at its right edge, nav below at the right edge as text links with icon + text; the active item is underlined with 2px ember.
- Page band: full-width ink block with a sentence-case Plex h1 (`type-h1`, for example "Clean", "Ask", "Trace") and a one-line lead. No display type and no all-caps inside the app.
- Card: bone surface, 24px padding, radius 11.52px, directional shadow, no border. Images inside bleed to the card edge with 0 radius.
- Button: primary = ink fill, parchment text, radius 2.88, min height 44px, 3px ember focus ring with 2px offset; secondary = text link with 1px underline, offset 3; disabled = opacity .45, not-allowed cursor, `aria-disabled`; loading = disabled + spinner icon + visible label change ("Working").
- Badge (stamp): 2.88px radius, ember outline, ink text 12-14px (an ember fill with parchment text measures 3.96:1, too low), only for result states such as "ungrounded", "accepted" or "rejected".
- Inputs: visible label above, bone fill, 1px ink border, radius 2.88, 44px minimum height, helper text below, error text below in ember with a warning icon and `role="alert"`.
- Drop zone: dashed 1px ink border on bone, large icon, text "Drop an image or video, or choose a file", keyboard-operable (a real `<input type="file">` inside a label), shows allowed types and size limits from the API info.
- Before/after: two images in a 2-column grid with captions; an optional "Compare" range slider must be keyboard accessible (arrow keys) and labelled.
- Decision timeline: ordered list of SAIR decisions (perception, restorer, image selection, super-resolution), each with a one-line rationale.
- Evidence track card: crop thumbnail (0 radius), label, time range `mm:ss.s to mm:ss.s` (tabular), mean score, status stamp, rationale, and a "Jump to time" text link that seeks the video player.
- Chat: user turns flush-left in ink on parchment with a 2px ink left rule, assistant turns on bone cards; ungrounded answers carry the text label "Not grounded in detections" (`type-label`); an `aria-live="polite"` region announces new answers; Enter sends, Shift+Enter inserts a newline.
- Trace table: sticky header, tabular numerals, columns node, decision, duration ms, GPU ms, fallback (icon + word), expandable rows.
- Skeletons: bone blocks with `animate-pulse` for any wait over 300ms; ingest shows a progress bar and step text "Step n of N".
- Toasts: ink on parchment, 4s auto-dismiss, `aria-live="polite"`, never steal focus.
- Icons: Phosphor (`@phosphor-icons/react`), regular weight, sizes 16/20/24 only. No emoji as icons.

## Layout and responsive

Mobile-first with breakpoints 375 / 768 / 1024 / 1440. One column on phones, two on tablets (media + decisions),
three at 1024+ (media, chat, evidence). No horizontal page scroll. Reserve space for the fixed header. Use
`min-h-dvh`. Body measure 60-75 characters.

## Motion

150-250ms ease-out for state changes, exits 60-70% of the enter duration, transform and opacity only, at most
one animated element per view, everything disabled under `prefers-reduced-motion`. Motion must explain a cause
(a new answer arriving, a progress step completing).

## Accessibility (must pass)

Contrast 4.5:1 for text, a visible 3px ember focus ring on every interactive element, a skip link to the main
content, heading order h1 then h2, labelled form controls, `aria-label` on icon-only buttons, focus moved to the
main region on tab change, meaningful alt text on all images (before/after, crops with label and time), color
never the only signal, touch targets at least 44x44 with 8px gaps, no hover-only interactions.

## Anti-patterns for this project

Gradients, glass blur (the one allowed exception is the landing hero headline resolving from soft focus once), shadows other than the card shadow, radius above 12px, sans-serif fonts outside the spec stacks, a second accent
color, centered long body text, emoji icons, placeholder-only labels, invented numbers (the About tab shows only
values returned by the API, or "not measured yet").

## Layout container (one system)

Every band's inner content (landing header, section nav, hero, all sections, footer, app header, nav, page banner and views) uses the
single `page` utility from `src/index.css`: `max-width: 90rem`, centred, 16px gutters (32px from 768px). Full-bleed backgrounds may span
the viewport, but content never sets its own max-width or gutters. Verified at 375, 768, 1024, 1440 and 1920 that the wordmark, the first
heading, the last nav item and the last card share the same left and right edges.
