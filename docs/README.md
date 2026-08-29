# infra-calc — Documentation

Infra-Calc is an interactive calculator for the economics of running LLM
inference infrastructure: hardware ROI, electricity costs, break-even points,
price frontiers, and profitability heatmaps, served as a browser UI.

## Repository map

- `run.py` — CLI entry point (`--gpu N`, models 1–5) and built-in HTTP server;
  generates charts/visualizations and serves `index.html`
- `index.html` (+ `index.html.bak` backup) — the calculator UI
- `js/` — `calculator.js`, `roi-calculator.js`, `charts.js`,
  `chart-manager.js`, `chart-plugins.js`, `heatmap.js`, `debug.js`,
  `main.js`, plus `js/precalculated/`
- `css/styles.css` — styling
- `visualizations/` — generated chart output (created by `run.py`)
- `data/` — supporting data files

## Running

```bash
python run.py            # default GPU model
python run.py --gpu 3    # pick a GPU model (1-5; see run.py main())
```

Dependencies per root `README.md`: matplotlib, numpy; the web interface is
served with Python's `http.server`.
