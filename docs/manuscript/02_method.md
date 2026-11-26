# Modeling method

Each concurrent slot earns the chosen per-minute rate for the scenario's
occupied hours. Monthly revenue scales with each slot's occupied minutes
and the assumed days in a month. Monthly costs combine an allocation of
the purchase price over the hardware lifespan, electricity during the
assumed active hours, and a maintenance allowance. The gap between
revenue and those costs is the reported monthly profit.

The Python generator compares the selected GPU's memory with the
configured per-instance requirement and limits the requested
concurrency using the whole allocation count. The model's one-thread
lower bound still requires the user to check memory feasibility. The
capacity comparison isolates a memory constraint from runtime throughput.
