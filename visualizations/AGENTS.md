# AGENTS.md — `infra-calc/visualizations`

> Repository-wide policy for template projects is maintained in the template repository, not restated here.

## What this is

Generated matplotlib output of `run.py` (recreated on every run — disposable):

- `profit_over_time.png` — cumulative profit/revenue/cost, break-even markers.
- `cost_breakdown.png` — donut of electricity/hardware/maintenance contributions.
- `price_frontier.png` — minimum viable price vs utilization, profit/loss zones.
- `heatmap_*.png` — profitability heatmaps (price×utilization, threads×price,
  threads×utilization).
- `gpu_*.png`, `llm_size_threads.png` — GPU model comparison / scaling views.
- `results.json` — machine-readable results for the same parameters.

Regenerate with `python3 run.py` (options in `run.py` argparse: `--gpu 1-5`).

## Layout (verified by direct listing, 2026-08-29)

```
cost_breakdown.png, gpu_cost_efficiency.png, gpu_memory_threads.png, heatmap_price_vs_utilization.png, heatmap_threads_vs_price.png, heatmap_threads_vs_utilization.png, llm_size_threads.png, price_frontier.png, profit_over_time.png, results.json
```
