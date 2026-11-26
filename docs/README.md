# Infra-Calc documentation

Infra-Calc models per-minute LLM inference service under editable price,
usage, GPU and cost assumptions.

| Guide | Contents |
| --- | --- |
| [Set up and run the calculator](getting-started.md) | Locked Python setup, browser dashboard and headless regeneration |
| [Model and assumptions](model.md) | Units, equations, GPU capacity and interpretation |
| [Source and generated files](source-tree.md) | File roles for the browser, generator and outputs |
| [Manuscript](manuscript/README.md) | Software methods, worked scenario and evidence |
| [Render the manuscript](reproduction/manuscript.md) | Clean checkout, chart preparation and pinned Template verification |
| [Development backlog](development/backlog.md) | Open model and interface work |

The browser entry is [index.html](../index.html) and the generator is [run.py](../run.py).
Calculation code lives in [js/calculator.js](../js/calculator.js). Generated
images and scenario results are under [visualizations/](../visualizations/).
