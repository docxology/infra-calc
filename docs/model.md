# Model and assumptions

The calculator uses a 30-day month and one rate per billed minute per
concurrent thread. The scenario inputs give hours used per day, thread
count, hardware expense, hardware lifespan in years, watts while in use,
electricity price per kilowatt-hour and monthly maintenance.

| Output | Calculation |
| --- | --- |
| Monthly revenue | Rate × hours per day × 60 × threads × 30 |
| Monthly hardware expense | Combined purchase cost divided by the hardware lifespan in months |
| Monthly electricity cost | Power in kilowatts × hours per day × electricity price × 30 |
| Monthly cost | Hardware expense + electricity + maintenance |
| Monthly profit | Revenue less monthly cost |
| Break-even estimate | Purchase cost divided by positive monthly profit |
| Lifespan ROI | Monthly profit × lifespan in months divided by purchase cost, shown as a percentage |

The Python GPU scenarios replace the GPU purchase price and power input with
the chosen model row. They constrain concurrency by the whole number of
per-instance VRAM allocations. The software retains a minimum displayed
capacity of one thread, so a GPU row with memory below the configured
requirement needs a separate feasibility check. The browser form uses
editable thread counts for scenario exploration. GPU and pricing rows are
bundled modeling assumptions.

The reported monthly result includes depreciation. The cumulative plot also
places purchases on its timeline. Use the [development backlog](development/backlog.md)
for the open work on separating capital cash flows from the monthly model.
The result file represents a scenario without finite payback with `Infinity`.
The heatmap sweep uses generic default costs and varies the plotted inputs directly.
The dashboard models billed minutes rather than measured token throughput.
Idle power, workload latency, model loading and system-specific memory
overhead require independent inputs.

[Reproduce the example](reproduction/manuscript.md) · [Documentation index](README.md)
