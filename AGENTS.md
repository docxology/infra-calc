# AGENTS.md — `infra-calc`

> Published copy of the local-only `projects/ongoing/DataTools/infra-calc` lane;
> origin `github.com/docxology/infra-calc` — commit and push here normally.

## What this is

`infra-calc` is a self-contained static SPA plus Python driver for modeling the
economics of running local LLM inference infrastructure: hardware ROI, electricity
cost, utilization, per-minute pricing, and GPU-model comparison. `run.py` computes
the economics, regenerates `visualizations/*.png` + `visualizations/results.json`,
writes precomputed results to `js/precalculated/results.js`, patches `index.html`,
then serves the whole directory over a local HTTP server and opens the browser.

## How to run (derived from run.py source, 2026-08-29)

- `python3 run.py` — full pipeline: compute, generate visualizations, serve on the
  first free port from 8000, open browser. Requires matplotlib + numpy
  (`pip install matplotlib numpy`); everything else is stdlib.
- `python3 run.py --gpu N` — select GPU model 1-5 (1=RTX 5090 default; models are
  defined in `run.py` `GPU_MODELS`). Invalid numbers fall back to 1.
- Alternatively open `index.html` directly in a browser — charts may be missing
  until `run.py` has run once (it creates `js/precalculated/results.js`).
- Tweak `DEFAULT_PARAMS` in `run.py` to change the business/LLM/hardware assumptions.

## Invariants / gotchas

- `run.py` rewrites `index.html` (after backing it up to the gitignored
  `index.html.bak`) and appends its style block to `css/styles.css` once — the
  append is skipped when the block is already present (fixed 2026-10-25) — so
  treat `visualizations/`, `js/precalculated/`, `index.html.bak` as generated
  artifacts, not hand-authored sources.
- `js/main.js` boot: when the module loads while `document.readyState === 'loading'`,
  init is wired to `DOMContentLoaded` (fixed 2026-08-30; a prior non-standard
  'dom-ready' event never fired, leaving the dashboard blank). The `dom-ready`
  shim that dispatched it was removed from `index.html` on 2026-10-25 — nothing
  listens for that event.
- `.gitignore` covers `__pycache__/`; do not commit Python bytecode.
- No package manifest, no tests; `index.html.bak` is a generated, gitignored
  backup, not a second entry point.

## Layout (verified by direct listing, 2026-08-29)

```
css/, data/, docs/, js/, visualizations/, .gitattributes, .gitignore, AGENTS.md, README.md, index.html, run.py
```
