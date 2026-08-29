# infra-calc — Agent Guide

## Layout

Static web app + Python driver. `run.py` (~1700 lines) computes the economics,
writes precalculated data and matplotlib visualizations, then serves the UI.
All interactive logic lives in `js/` modules; `index.html` ties them together.

## Conventions observed

- No package manifest or tests; dependencies are documented in the root
  `README.md` (matplotlib, numpy, stdlib http.server).
- Generated artifacts (`visualizations/`, `js/precalculated/`) are recreated by
  `run.py` on each run — treat as disposable.

## How docs here are maintained

Root `README.md` covers purpose, features, and economics model. Keep
`docs/README.md` commands in sync with `run.py`'s argparse options.

## Notes for agents

- `index.html.bak` is a backup of the UI; do not treat as a second entry point.
- Verify GPU model numbering and defaults against `run.py` `main()` before
  documenting them.
