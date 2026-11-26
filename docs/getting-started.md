# Set up and run Infra-Calc

Install uv and run from the repository root. The checkout selects Python 3.15.2
and locks the NumPy and Matplotlib versions used by the generator.

```sh
uv sync --locked
uv run --locked python run.py
```

The driver generates PNGs and JSON in `visualizations/`, writes the browser
precomputations in `js/precalculated/results.js`, and ensures
`index.html` loads them. It serves the checkout on the first available
loopback port starting at 8000. Open the printed HTTP URL in a browser.

Use `--gpu N` to select one of the bundled scenarios. The entry point lists
the available models. It selects the first row by default and when the
requested number is outside the available range.

For an offline generation run on the bundled H100 scenario, use the
noninteractive Matplotlib backend and generate the inputs without the server.

```sh
MPLBACKEND=Agg uv run --locked python run.py --gpu 3 --generate-only
```

Serve the checked-in dashboard without regenerating its charts by running
`uv run --locked python -m http.server 8000 --bind 127.0.0.1` from the
repository root. Browse to `http://127.0.0.1:8000/`. JavaScript modules load over
HTTP, and the dashboard loads Chart.js from its declared CDN URL.

Use the form for an interactive estimate. Edit `DEFAULT_PARAMS` for a new
default generation scenario. Generated chart and precomputation files live
alongside the app, so use a disposable checkout for a manuscript build.

[Documentation index](README.md)
