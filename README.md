# Infra-Calc

A sophisticated calculator for understanding the economics of running LLM inference infrastructure with enhanced visualizations and interactive features.

## Purpose

This tool helps answer questions like:
- Can we profitably run a 70B parameter LLM locally if we charge $0.02 per minute of usage?
- What hardware specifications would be needed to run multiple inference threads simultaneously?
- How do electricity costs affect the profitability of local LLM hosting?
- What is the break-even point for purchasing specialized hardware?
- How does utilization rate impact long-term ROI?

## Features

- Calculate hardware ROI based on service pricing
- Compare different hardware configurations
- Analyze electricity costs
- Visualize price frontiers and break-even points
- Estimate profitability over time
- Interactive heatmaps showing profitability across different parameters
- Animated visualizations showing parameter relationships
- Responsive UI with smooth transitions and effects

## Enhanced Visualizations

The calculator now includes:

- **High-resolution charts** with improved color schemes and readability
- **Interactive heatmaps** showing profitability across different parameter combinations
- **Animated GIFs** showing how profitability changes with different thread counts
- **Annotated visualizations** with key points, break-even markers, and trend highlights
- **Smooth UI transitions** for a better user experience

## Getting Started

### Option 1: Using the Python Script (Recommended)

1. Clone this repository
2. Install required dependencies:
   ```
   pip install matplotlib numpy pillow
   ```
3. Run the Python script:
   ```
   python run.py
   ```
4. The script will:
   - Generate comprehensive visualizations in the `/visualizations` folder
   - Start a local web server
   - Open the calculator in your web browser with pre-calculated values and animations

### Option 2: Directly Opening HTML

1. Clone this repository
2. Open `index.html` in your browser

Note: Direct HTML opening may not show charts correctly unless the Python script has been run at least once to generate the visualizations.

## Project Structure

- `/index.html` - Main interface
- `/js/` - JavaScript modules for calculations
  - `/js/precalculated/` - Pre-calculated results and visualization loaders
- `/css/` - Styling and animation effects
- `/data/` - Default data for hardware costs, electricity rates, etc.
- `/run.py` - Python script to generate visualizations and serve the app
- `/visualizations/` - Generated chart images, animations, and calculation results

## Customizing the Calculations

To use different default parameters:

1. Open `run.py` in a text editor
2. Modify the `DEFAULT_PARAMS` dictionary to suit your needs
3. Run the script again to generate new visualizations with your parameters

For example, to analyze a smaller LLM setup:

```python
DEFAULT_PARAMS = {
    "charge_per_minute": 0.01,
    "utilization_rate": 8,
    "concurrent_threads": 2,
    "llm_size": 13,
    "vram_required": 15,
    # ... other parameters
}
```

## Advanced Visualization Options

The enhanced visualizations include:

1. **Profit Projection Over Time**
   - Shows cumulative profit, revenue, and cost over the hardware lifespan
   - Marks break-even points and hardware replacement cycles
   - Uses color-coded regions to distinguish profit and loss periods

2. **Cost Breakdown**
   - Interactive donut chart showing the relative contribution of different costs
   - Enhanced labels with dollar values and percentages
   - Color-coded for easy identification

3. **Price Frontier**
   - Shows the minimum viable price at different utilization levels
   - Highlights profit and loss zones
   - Marks current operating point and profit margin

4. **Profitability Heatmaps**
   - Interactive visualization of profitability across different parameter combinations
   - Break-even contour lines
   - Annotations for maximum profit points

5. **Animated Visualization**
   - Shows how profitability changes with different thread counts
   - Helps understand the impact of scaling operations

## Understanding the Economics

The calculator considers:

- Electricity cost and power consumption rates
- Hardware costs including GPU, CPU, RAM, and other components
- Hardware lifespan and depreciation
- LLM computational resource requirements
- Daily utilization rates and concurrent processing capacity
- Service pricing model (per minute charging)

This helps understand whether a specific LLM service can be run profitably on local infrastructure and identify the optimal operating parameters for maximum ROI.

## Acknowledgements

This project uses the following libraries:
- Matplotlib for generating charts and visualizations
- NumPy for numerical computations
- Python's http.server for serving the web interface