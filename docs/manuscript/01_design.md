# Software design

The repository pairs a Python scenario generator with a browser calculator
[@infraCalc]. The driver loads the bundled hardware rows, applies the
selected GPU's modeled purchase cost and power, and evaluates revenue and
operating costs. The browser calculator consumes the editable scenario
fields and uses the generated data and images for the default dashboard.

The driver uses NumPy for numerical sweeps and Matplotlib for rendered
charts [@harris2020numpy; @hunter2007matplotlib]. The source release includes
the cost, frontier, thread capacity and utilization chart logic.
