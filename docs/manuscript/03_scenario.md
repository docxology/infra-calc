# Bundled hardware scenario

Run the generation command in the reproduction guide with the bundled
H100 row. That row constrains the requested concurrency to the
scenario's whole memory allocations. Under the default price and daily
hours, the model uses 2 concurrent slots. It reports USD 864.00 in
monthly revenue and USD 1060.02 in monthly cost, for a monthly
result of USD -196.02. The generated result file records the selected
GPU and calculated quantities.

The thread-versus-price chart holds the generic `DEFAULT_PARAMS`
costs and usage fixed. Its grid varies prices and threads directly, and
the colors encode normalized monthly profit. The selected GPU row and
memory clamp apply to the worked scenario's metrics.

![Monthly result across per-minute prices and simulated thread counts under fixed assumptions.](../../output/figures/heatmap_threads_vs_price.png){#fig:infra-profitability-sweep}
