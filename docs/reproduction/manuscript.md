# Render the Infra-Calc manuscript

Use a standalone checkout, uv, Pandoc and XeLaTeX. The upstream run guide
lists the shared renderer's LaTeX dependencies. [Template run guide](https://github.com/docxology/template/blob/87704feae0a4490ef7dad9c931d268e8c30273c2/docs/RUN_GUIDE.md)

Create a detached render worktree so the chart generator writes only to the
render copy. Run these commands from the original Infra-Calc checkout.

```sh
infra_source="$(pwd -P)"
git clone https://github.com/docxology/template.git ../infra-calc-template
git -C ../infra-calc-template checkout --detach 87704feae0a4490ef7dad9c931d268e8c30273c2
cd ../infra-calc-template
uv sync --locked --no-dev
mkdir -p projects/working
git -C "$infra_source" worktree add --detach "$PWD/projects/working/infra-calc" HEAD
(
  cd projects/working/infra-calc
  uv sync --locked
  MPLBACKEND=Agg uv run --locked python run.py --gpu 3 --generate-only
  mkdir -p output/figures output/data
  cp visualizations/heatmap_threads_vs_price.png output/figures/
  cp docs/manuscript/figure_registry.json output/figures/
  cp visualizations/results.json output/data/h100-scenario.json
)
uv run --no-sync python scripts/pipeline/stage_03_render.py --project working/infra-calc
uv run --no-sync python scripts/pipeline/stage_04_validate.py --project working/infra-calc
```

Open `projects/working/infra-calc/output/pdf/infra-calc_combined.pdf` and
`projects/working/infra-calc/output/web/index.html` inside the Template
checkout. Verify the heatmap, worked scenario and resolved references.
Validation and rendered provenance are recorded in the render worktree's
`output/reports/`. The scenario inputs and results are in
`output/data/h100-scenario.json`.

[Manuscript source](../manuscript/README.md) · [Documentation index](../README.md)
