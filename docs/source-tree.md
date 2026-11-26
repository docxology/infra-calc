# Source and generated files

| Path | Role |
| --- | --- |
| [index.html](../index.html) and [styles.css](../css/styles.css) | Browser interface and styles |
| [calculator.js](../js/calculator.js) and [roi-calculator.js](../js/roi-calculator.js) | Economics and ROI computations |
| [main.js](../js/main.js) and [debug.js](../js/debug.js) | Form wiring and diagnostics |
| [charts.js](../js/charts.js), [chart-manager.js](../js/chart-manager.js), [chart-plugins.js](../js/chart-plugins.js), and [heatmap.js](../js/heatmap.js) | Chart rendering, annotations and heatmaps |
| [default-configs.js](../data/default-configs.js) | Bundled browser scenarios |
| [run.py](../run.py) | Python scenario driver, chart generator and local server |
| [results.js](../js/precalculated/results.js) and [visualizations](../visualizations/) | Checked-in generated values and chart images |

The Python driver generates the monthly, cost breakdown, frontier,
profitability heatmap and GPU-capacity views. Follow the [generation commands](getting-started.md)
to rebuild them. The manuscript copies its selected inputs to the render worktree's
`output/`.
