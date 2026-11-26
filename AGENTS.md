# Infra-Calc working notes

The Python driver and JavaScript calculator model local inference economics. The [documentation index](docs/README.md) links setup, equations, source layout and verification. Keep both calculation paths in mind when changing modeled assumptions or results.

- Use `uv sync --locked` for Python 3.15.2 with the recorded NumPy and Matplotlib versions.
- Run from the repository root. `uv run --locked python run.py` generates assets and serves the app on loopback. `--generate-only` regenerates the same assets without starting the server.
- Follow the [manuscript procedure](docs/reproduction/manuscript.md) in a detached checkout to keep generated dashboard output separate from the source checkout.
- Treat the GPU, cost and usage values as scenario assumptions. Document which values feed each plotted or reported result.
