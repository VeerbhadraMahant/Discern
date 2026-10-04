# Discern frontend

React 19, TypeScript (strict), Tailwind v4, Vite. It talks to the Discern Gradio app through
`@gradio/client`; the contract is in [API.md](API.md) and the visual language in
[design-system/MASTER.md](design-system/MASTER.md).

## Run against a local Gradio app

```bash
cd frontend
npm ci
cp .env.example .env.local      # optional; the default URL is http://127.0.0.1:7860
npm run dev                     # http://localhost:5173
```

Start the backend separately (for example `python space/app.py`) so it listens on the URL in
`VITE_DISCERN_URL`.

## Run against a Hugging Face Space

Either set `VITE_DISCERN_URL=https://<owner>-<space>.hf.space` at build time, or open the page with a
query parameter, which overrides the build-time value at runtime (http and https URLs only):

```
http://localhost:5173/?space=https://<owner>-<space>.hf.space
```

Calls are made from the visitor's browser, so the visitor's own ZeroGPU quota applies.

## Demo mode (no backend)

`VITE_USE_MOCK=1 npm run dev`, or add `?mock=1` to any URL. A built-in `MockDiscernClient` returns canned
responses and the header shows a "Demo data" stamp the whole time. Nothing it shows is a measurement.

## Environment variables

| Variable | Meaning | Default |
| --- | --- | --- |
| `VITE_DISCERN_URL` | Gradio app URL | `http://127.0.0.1:7860` |
| `VITE_USE_MOCK` | `1` selects demo mode | unset |

Query parameters: `?space=<url>` and `?mock=1` (or `?mock=0` to turn a build-time mock off).

## Scripts

| Script | What it does |
| --- | --- |
| `npm run dev` | Vite dev server |
| `npm run build` | type-check, then production build into `dist/` |
| `npm run preview` | serve the production build |
| `npm run typecheck` | `tsc --noEmit` |
| `npm run lint` | ESLint |
| `npm test -- --run` | Vitest (API client, mock conformance, components, routing, contrast and style guardrails) |

The landing page is the default route (`#/`, an empty hash, or an in-page anchor); the tabs are hash-routed
(`#/clean`, `#/ask`, `#/trace`, `#/feedback`, `#/about`), so each is deep-linkable. "Try with demo data" links to
`?mock=1#/clean`. The landing page is code-split (`src/landing/`, loaded with `React.lazy`) and renders every claim from
`src/landing/facts.ts`, which cites the repository README per item.

Typography follows [../docs/typography.md](../docs/typography.md): Bricolage Grotesque (static instance) for the landing hero,
section headings and the wordmark, IBM Plex Sans for everything else, IBM Plex Mono for literal machine output. Fonts are
self-hosted in `public/fonts` with their licenses in `public/fonts/LICENSES.md`.

## Deploying

The repo-root `vercel.json` builds this folder from the repo root (`cd frontend && npm run build`, output
`frontend/dist`, SPA rewrite to `/index.html`). Set `VITE_DISCERN_URL` in the Vercel project's
environment variables.

## Screenshots (demo mode)

Desktop: [clean](screenshots/desktop-clean.png), [ask](screenshots/desktop-ask.png),
[trace](screenshots/desktop-trace.png), [feedback](screenshots/desktop-feedback.png),
[about](screenshots/desktop-about.png).

Phone (375 px): [clean](screenshots/phone-clean.png), [ask](screenshots/phone-ask.png),
[trace](screenshots/phone-trace.png), [feedback](screenshots/phone-feedback.png),
[about](screenshots/phone-about.png).

Landing page, desktop: [hero](screenshots/landing-desktop-hero.png), [numbers](screenshots/landing-desktop-numbers.png),
[results](screenshots/landing-desktop-results.png), [query types](screenshots/landing-desktop-queries.png).
Landing page, phone: [hero](screenshots/landing-phone-hero.png), [numbers](screenshots/landing-phone-numbers.png).

## Deviations from MASTER.md

Measured contrast of the palette: charcoal on parchment is 4.38:1, ember on parchment is 3.96:1 and
parchment on ember is 3.96:1, all under the 4.5:1 text rule. So:

- Secondary text is ink, not charcoal. Charcoal is unused for text.
- Status stamps are outlined in ember with ink text instead of an ember fill with parchment text.
- Ember is used only for focus rings, nav underline, stamp borders and warning icons (all above 3:1).
- Body weight is 400 rather than 300, for legibility on the bone surfaces.
- The ingest step text reads "Step n: message" without a total, because the API gives no step count.
