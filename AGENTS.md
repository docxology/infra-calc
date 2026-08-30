# AGENTS.md — `infra-calc`

> Local-only path under `projects/ongoing/DataTools/` — matched by the
> root `.gitignore` rule `projects/*`; never commit. Repo-wide policy:
> see `/Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md`.

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

- `run.py` rewrites `index.html` (after backing it up to `index.html.bak`) and
  appends to `css/` on each run — treat `visualizations/`, `js/precalculated/`,
  `index.html.bak` as generated artifacts, not hand-authored sources.
- No package manifest, no tests; `index.html.bak` is a backup, not a second entry point.

## Layout (verified by direct listing, 2026-08-29)

```
css/, data/, docs/, js/, visualizations/, .gitattributes, .gitignore, README.md, index.html, index.html.bak, run.py
```
