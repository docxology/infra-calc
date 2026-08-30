# AGENTS.md — `infra-calc/data`

> Local-only path under `projects/ongoing/DataTools/` — matched by the
> root `.gitignore` rule `projects/*`; never commit. Repo-wide policy:
> see `/Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md`.

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
