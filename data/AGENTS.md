# AGENTS.md — `infra-calc/data`

> Published copy of the local-only `projects/ongoing/DataTools/infra-calc` lane;
> origin `github.com/docxology/infra-calc` — commit and push here normally.

## What this is

Static defaults for the SPA: `default-configs.js` provides the baseline
hardware/economic parameter sets the UI loads before user edits. Server-side
defaults (including the GPU model catalog) live in `run.py` (`DEFAULT_PARAMS`,
`GPU_MODELS`) and are regenerated into `js/precalculated/results.js` on each
`python3 run.py` run — keep the two in mind as separate layers: this dir is the
hand-authored baseline, `js/precalculated/` is generated.

## Layout (verified by direct listing, 2026-08-29)

```
default-configs.js
```
