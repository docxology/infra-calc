#!/usr/bin/env python3
import os
import json
import subprocess
import webbrowser
import time
import math
import http.server
import socketserver
import threading
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import shutil
from pathlib import Path
import argparse

# Create visualizations directory if it doesn't exist
vis_dir = Path('visualizations')
vis_dir.mkdir(exist_ok=True)

# Default parameters based on the large LLM configuration
DEFAULT_PARAMS = {
    # Business Model
    "charge_per_minute": 0.02,
    "utilization_rate": 12,
    "concurrent_threads": 4,
    
    # LLM Parameters
    "llm_size": 70,
    "inference_time": 1.0,
    "vram_required": 40,
    
    # Hardware Costs
    "gpu_cost": 3000,
    "cpu_cost": 500,
    "ram_cost": 300,
    "other_hardware": 700,
    "hardware_lifespan": 3,
    
    # Operational Costs
    "electricity_cost": 0.15,
    "power_consumption": 800,
    "maintenance_cost": 50,
}

# GPU Model data
GPU_MODELS = [
    {
        "name": "NVIDIA RTX 5090",
        "cost": 3000,  # Average of range $2,000-$3,999
        "memory_gb": 24,
        "power_w": 394
    },
    {
        "name": "NVIDIA RTX 5080",
        "cost": 1200,  # Average of range $1,000-$1,400
        "memory_gb": 16,
        "power_w": 253
    },
    {
        "name": "NVIDIA H100",
        "cost": 33500,  # Average of range $27,000-$40,000
        "memory_gb": 80,
        "power_w": 700
    },
    {
        "name": "NVIDIA H200 (SXM)",
        "cost": 44000,
        "memory_gb": 141,
        "power_w": 650  # Average of range 600-700
    },
    {
        "name": "NVIDIA GB200 (Blackwell)",
        "cost": 100000,  # Placeholder since price is on request
        "memory_gb": 192,
        "power_w": 1200
    }
]

# Default selected GPU model
DEFAULT_GPU_MODEL = GPU_MODELS[0]  # RTX 5090

def calculate_electricity_cost(power_consumption, hours_per_day, cost_per_kwh):
    """Calculate monthly electricity cost"""
    power_in_kw = power_consumption / 1000
    daily_energy_kwh = power_in_kw * hours_per_day
    monthly_cost = daily_energy_kwh * cost_per_kwh * 30
    return monthly_cost

def calculate_max_concurrent_threads(gpu_memory_gb, vram_required_per_model):
    """Calculate the maximum number of concurrent threads based on GPU memory constraints"""
    if vram_required_per_model <= 0:
        return 1  # Default to 1 if model requirements are invalid
    
    # Calculate raw maximum threads
    max_threads = int(gpu_memory_gb / vram_required_per_model)
    
    # Return at least 1 thread even if the model is too large for the GPU
    # (in a real implementation, we'd raise an error or warning)
    return max(1, max_threads)

def calculate_hardware_depreciation(hardware_cost, lifespan_years):
    """Calculate monthly hardware depreciation cost"""
    lifespan_months = lifespan_years * 12
    return hardware_cost / lifespan_months

def calculate_monthly_revenue(charge_per_minute, hours_per_day, concurrent_threads):
    """Calculate monthly revenue"""
    minutes_per_day = hours_per_day * 60
    daily_revenue = minutes_per_day * charge_per_minute * concurrent_threads
    monthly_revenue = daily_revenue * 30
    return monthly_revenue

def calculate_monthly_costs(params, gpu_model=None):
    """Calculate total monthly costs"""
    # If GPU model is provided, use its parameters
    if gpu_model:
        gpu_cost = gpu_model["cost"]
        power_consumption = gpu_model["power_w"]
    else:
        gpu_cost = params["gpu_cost"]
        power_consumption = params["power_consumption"]
    
    total_hardware_cost = gpu_cost + params["cpu_cost"] + params["ram_cost"] + params["other_hardware"]
    
    # Calculate monthly hardware depreciation
    hardware_depreciation = calculate_hardware_depreciation(total_hardware_cost, params["hardware_lifespan"])
    
    # Calculate monthly electricity cost
    electricity_cost_monthly = calculate_electricity_cost(
        power_consumption, 
        params["utilization_rate"], 
        params["electricity_cost"]
    )
    
    # Total monthly costs
    total_monthly_cost = hardware_depreciation + electricity_cost_monthly + params["maintenance_cost"]
    
    return {
        "hardware_depreciation": hardware_depreciation,
        "electricity_cost_monthly": electricity_cost_monthly,
        "maintenance_cost": params["maintenance_cost"],
        "total_monthly_cost": total_monthly_cost,
        "total_hardware_cost": total_hardware_cost,
        "gpu_cost": gpu_cost,
        "power_consumption": power_consumption
    }

def calculate_profitability(monthly_revenue, monthly_cost, hardware_cost, hardware_lifespan):
    """Calculate profitability metrics"""
    # Calculate monthly profit
    monthly_profit = monthly_revenue - monthly_cost
    
    # Calculate profit margin
    profit_margin = (monthly_profit / monthly_revenue) * 100 if monthly_revenue > 0 else 0
    
    # Calculate break-even time in months
    break_even_time_months = hardware_cost / monthly_profit if monthly_profit > 0 else float('inf')
    
    # Calculate ROI over hardware lifespan
    lifespan_months = hardware_lifespan * 12
    lifetime_profit = monthly_profit * lifespan_months
    roi = (lifetime_profit / hardware_cost) * 100
    
    return {
        "monthly_profit": monthly_profit,
        "profit_margin": profit_margin,
        "break_even_time_months": break_even_time_months,
        "roi": roi
    }

def generate_gpu_thread_constraints_data(llm_sizes, gpu_models):
    """Generate data showing how many concurrent threads can run for each GPU model and LLM size"""
    # Calculate VRAM requirements approximately based on LLM size (in billions of parameters)
    # Using a simple rule of thumb: VRAM in GB ≈ (model_size_in_billions * 2) for 16-bit precision
    
    vram_requirements = {size: size * 2 for size in llm_sizes}
    max_threads_by_model = {}
    
    for model in gpu_models:
        max_threads_by_model[model["name"]] = []
        for size in llm_sizes:
            vram_required = vram_requirements[size]
            max_threads = calculate_max_concurrent_threads(model["memory_gb"], vram_required)
            max_threads_by_model[model["name"]].append(max_threads)
    
    return {
        "llm_sizes": llm_sizes,
        "vram_requirements": vram_requirements,
        "max_threads_by_model": max_threads_by_model
    }

def generate_price_frontier_data(params, costs, gpu_model=None):
    """Generate price frontier data for different utilization rates"""
    utilization_rates = []
    min_prices = []
    break_even_times = []
    
    # Use GPU model data if provided
    if gpu_model:
        total_hardware_cost = gpu_model["cost"] + params["cpu_cost"] + params["ram_cost"] + params["other_hardware"]
        power_consumption = gpu_model["power_w"]
    else:
        total_hardware_cost = params["gpu_cost"] + params["cpu_cost"] + params["ram_cost"] + params["other_hardware"]
        power_consumption = params["power_consumption"]
    
    # Calculate for different utilization rates
    for hours in range(1, 25):
        # Adjusted costs for different utilization rates
        adjusted_electricity_cost = calculate_electricity_cost(
            power_consumption, 
            hours, 
            params["electricity_cost"]
        )
        
        adjusted_cost = costs["hardware_depreciation"] + adjusted_electricity_cost + costs["maintenance_cost"]
        
        # Get constrained concurrent threads
        constrained_threads = params["concurrent_threads"]
        if gpu_model:
            max_threads = calculate_max_concurrent_threads(gpu_model["memory_gb"], params["vram_required"])
            constrained_threads = min(params["concurrent_threads"], max_threads)
        
        # Minutes per month at this utilization
        minutes_per_month = hours * 60 * 30 * constrained_threads
        
        # Minimum price per minute
        min_price = adjusted_cost / minutes_per_month
        
        # Monthly profit at this minimum price
        monthly_profit = (min_price * minutes_per_month) - adjusted_cost
        
        # Break-even time
        break_even_time = total_hardware_cost / monthly_profit if monthly_profit > 0 else float('inf')
        
        utilization_rates.append(hours)
        min_prices.append(min_price)
        break_even_times.append(break_even_time)
    
    return {
        "utilization_rates": utilization_rates,
        "min_prices": min_prices,
        "break_even_times": break_even_times
    }

def generate_profit_over_time_data(params, monthly_revenue, monthly_costs, gpu_model=None):
    """Generate profit over time data"""
    # Use GPU model data if provided
    if gpu_model:
        total_hardware_cost = gpu_model["cost"] + params["cpu_cost"] + params["ram_cost"] + params["other_hardware"]
    else:
        total_hardware_cost = params["gpu_cost"] + params["cpu_cost"] + params["ram_cost"] + params["other_hardware"]
    
    months = []
    cumulative_profit = []
    cumulative_cost = []
    cumulative_revenue = []
    
    # Calculate initial investment
    current_profit = -total_hardware_cost
    current_cost = total_hardware_cost
    current_revenue = 0
    
    # Calculate for 3x hardware lifespan to show long-term trends
    total_months = min(params["hardware_lifespan"] * 36, 60)  # Cap at 5 years (60 months)
    
    for month in range(total_months + 1):
        months.append(month)
        cumulative_profit.append(current_profit)
        cumulative_cost.append(current_cost)
        cumulative_revenue.append(current_revenue)
        
        # Add monthly revenue and costs
        if month > 0:
            current_profit += monthly_revenue - monthly_costs["total_monthly_cost"]
            current_cost += monthly_costs["total_monthly_cost"]
            current_revenue += monthly_revenue
        
        # Handle hardware replacement at end of life
        if month > 0 and month % (params["hardware_lifespan"] * 12) == 0 and month < total_months:
            current_profit -= total_hardware_cost
            current_cost += total_hardware_cost
    
    return {
        "months": months,
        "cumulative_profit": cumulative_profit,
        "cumulative_cost": cumulative_cost,
        "cumulative_revenue": cumulative_revenue
    }

def generate_profitability_heatmap_data(base_params, x_param, y_param):
    """Generate data for a profitability heatmap"""
    # Define ranges for each parameter
    param_ranges = {
        "utilization": list(range(1, 25)),  # 1-24 hours
        "price": [round(0.01 + i * 0.01, 2) for i in range(10)],  # $0.01-$0.10
        "threads": list(range(1, 9))  # 1-8 threads
    }
    
    x_values = param_ranges[x_param]
    y_values = param_ranges[y_param]
    
    profits = []
    
    for y_val in y_values:
        row = []
        for x_val in x_values:
            # Copy base parameters and update with current x and y values
            params = base_params.copy()
            
            if x_param == "utilization":
                params["utilization_rate"] = x_val
            elif x_param == "price":
                params["charge_per_minute"] = x_val
            elif x_param == "threads":
                params["concurrent_threads"] = x_val
                
            if y_param == "utilization":
                params["utilization_rate"] = y_val
            elif y_param == "price":
                params["charge_per_minute"] = y_val
            elif y_param == "threads":
                params["concurrent_threads"] = y_val
            
            # Calculate profitability with these parameters
            monthly_revenue = calculate_monthly_revenue(
                params["charge_per_minute"], 
                params["utilization_rate"], 
                params["concurrent_threads"]
            )
            
            monthly_costs = calculate_monthly_costs(params)
            monthly_profit = monthly_revenue - monthly_costs["total_monthly_cost"]
            
            row.append(monthly_profit)
        
        profits.append(row)
    
    return {
        "x_values": x_values,
        "y_values": y_values,
        "profit_matrix": profits
    }

def create_visualizations(params, selected_gpu_model=None):
    """
    Create high-quality visualizations and save to the visualizations directory
    
    This function generates multiple visualizations based on the provided parameters:
    1. Profit Over Time - Shows cumulative profit, revenue, and cost projections
    2. Cost Breakdown - Pie chart showing relative cost components
    3. Price Frontier - Shows minimum pricing needed at different utilization levels
    4. Profitability Heatmaps - Multiple heatmaps showing profitability across different parameter combinations
    5. GPU Model Comparison - Shows how different GPU models affect concurrent threads and profitability
    
    All visualizations are saved as high-resolution PNG files in the visualizations directory.
    
    Args:
        params: Dictionary containing all calculation parameters
        selected_gpu_model: Optional GPU model data to use instead of generic parameters
        
    Returns:
        Dictionary containing calculated results used for visualizations
    """
    # Use the selected GPU model if provided
    gpu_model = selected_gpu_model
    
    # If a GPU model is selected, update related parameters
    constrained_threads = params["concurrent_threads"]
    if gpu_model:
        # Apply VRAM constraint to concurrent threads
        max_threads = calculate_max_concurrent_threads(gpu_model["memory_gb"], params["vram_required"])
        constrained_threads = min(params["concurrent_threads"], max_threads)
        
        # Update power consumption for accurate electricity cost calculation
        power_consumption = gpu_model["power_w"]
    else:
        power_consumption = params["power_consumption"]
    
    # Create a copy of params with constrained threads
    adjusted_params = params.copy()
    adjusted_params["concurrent_threads"] = constrained_threads
    
    # Calculate core metrics
    monthly_revenue = calculate_monthly_revenue(
        adjusted_params["charge_per_minute"], 
        adjusted_params["utilization_rate"], 
        constrained_threads  # Use constrained threads for revenue
    )
    
    monthly_costs = calculate_monthly_costs(adjusted_params, gpu_model)
    
    profitability = calculate_profitability(
        monthly_revenue, 
        monthly_costs["total_monthly_cost"], 
        monthly_costs["total_hardware_cost"],
        adjusted_params["hardware_lifespan"]
    )
    
    # Generate datasets for charts
    price_frontier_data = generate_price_frontier_data(adjusted_params, monthly_costs, gpu_model)
    profit_over_time_data = generate_profit_over_time_data(adjusted_params, monthly_revenue, monthly_costs, gpu_model)
    
    # GPU-specific visualizations
    llm_sizes = [7, 13, 20, 30, 65, 70, 105, 175]  # Common LLM sizes in billions of parameters
    gpu_thread_data = generate_gpu_thread_constraints_data(llm_sizes, GPU_MODELS)
    
    # Setup better visualization aesthetics
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Configure global font size for better readability
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    # Set consistent colors for different data series
    profit_color = '#2ecc71'    # green
    revenue_color = '#3498db'   # blue
    cost_color = '#e74c3c'      # red
    
    # 1. Enhanced Profit Over Time Chart
    plt.figure(figsize=(12, 7))
    
    # Add a subtle grid background for reference
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot with enhanced line styles and thicker lines
    plt.plot(profit_over_time_data["months"], profit_over_time_data["cumulative_profit"], 
             color=profit_color, linewidth=3, label='Cumulative Profit')
    plt.plot(profit_over_time_data["months"], profit_over_time_data["cumulative_revenue"], 
             color=revenue_color, linewidth=2.5, linestyle='--', label='Cumulative Revenue')
    plt.plot(profit_over_time_data["months"], profit_over_time_data["cumulative_cost"], 
             color=cost_color, linewidth=2.5, linestyle='--', label='Cumulative Cost')
    
    # Mark initial investment with annotation
    plt.scatter(0, profit_over_time_data["cumulative_profit"][0], color='black', s=100, zorder=5)
    plt.annotate(f'Initial Investment: ${-profit_over_time_data["cumulative_profit"][0]:.0f}', 
                 (0, profit_over_time_data["cumulative_profit"][0]), 
                 xytext=(10, -30), textcoords='offset points', 
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))
    
    # Mark break-even point with enhanced visibility
    if profitability["break_even_time_months"] != float('inf'):
        break_even_month = profitability["break_even_time_months"]
        
        # Find the closest index to the break-even month
        closest_index = min(range(len(profit_over_time_data["months"])), 
                           key=lambda i: abs(profit_over_time_data["months"][i] - break_even_month))
        
        plt.axvline(x=break_even_month, color='black', linestyle='--', alpha=0.7)
        plt.scatter(break_even_month, 0, color='black', s=80, zorder=5)
        plt.annotate(f'Break-even: {break_even_month:.1f} months', 
                     (break_even_month, 0), 
                     xytext=(10, 30), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'),
                     fontweight='bold')
    
    # Mark hardware replacement points
    for i, month in enumerate(profit_over_time_data["months"]):
        if month > 0 and month % (params["hardware_lifespan"] * 12) == 0 and month < max(profit_over_time_data["months"]):
            plt.axvline(x=month, color='gray', linestyle='-.', alpha=0.5)
            plt.annotate(f'Hardware Replacement', 
                         (month, profit_over_time_data["cumulative_profit"][i]), 
                         xytext=(5, -20), textcoords='offset points',
                         fontsize=10, alpha=0.8)
    
    # Add final profit annotation
    final_month = profit_over_time_data["months"][-1]
    final_profit = profit_over_time_data["cumulative_profit"][-1]
    plt.scatter(final_month, final_profit, color=profit_color, s=100, zorder=5)
    plt.annotate(f'Final Profit: ${final_profit:.0f}', 
                 (final_month, final_profit), 
                 xytext=(-150, -30), textcoords='offset points')
    
    plt.title('Profit Projection Over Time', fontweight='bold')
    plt.xlabel('Months')
    plt.ylabel('Amount ($)')
    
    # Add shaded regions for profit/loss areas
    plt.axhspan(0, max(profit_over_time_data["cumulative_profit"]) * 1.1, 
                color=profit_color, alpha=0.1)
    plt.axhspan(min(profit_over_time_data["cumulative_profit"]) * 1.1, 0, 
                color=cost_color, alpha=0.1)
    
    # Enhanced legend with shadow
    legend = plt.legend(loc='upper left', framealpha=0.9, shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('gray')
    
    plt.tight_layout()
    plt.savefig(vis_dir / 'profit_over_time.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Enhanced Cost Breakdown Chart
    cost_labels = ['Hardware Depreciation', 'Electricity', 'Maintenance']
    cost_values = [
        monthly_costs["hardware_depreciation"],
        monthly_costs["electricity_cost_monthly"],
        monthly_costs["maintenance_cost"]
    ]
    
    # Calculate percentages for annotations
    total_cost = sum(cost_values)
    percentages = [value / total_cost * 100 for value in cost_values]
    
    # Set better color palette with improved visibility and contrast
    colors = ['#3498db', '#f39c12', '#2ecc71']  # blue, orange, green
    
    plt.figure(figsize=(10, 7))
    
    # Create pie chart with enhanced visual elements
    wedges, texts, autotexts = plt.pie(
        cost_values, 
        labels=None,  # We'll add custom labels outside the pie
        autopct='%1.1f%%', 
        startangle=90, 
        colors=colors,
        wedgeprops={'width': 0.6, 'edgecolor': 'w', 'linewidth': 1.5},  # Donut style
        shadow=True,
        explode=(0.05, 0, 0)  # Slightly emphasize hardware cost
    )
    
    # Customize autopct text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    
    # Add a white circle at the center to create a donut chart
    centre_circle = plt.Circle((0, 0), 0.3, fc='white')
    plt.gca().add_artist(centre_circle)
    
    # Add custom legend with dollar values
    legend_labels = [f"{label}: ${value:.2f} ({percent:.1f}%)" 
                    for label, value, percent in zip(cost_labels, cost_values, percentages)]
    plt.legend(wedges, legend_labels, loc="center", bbox_to_anchor=(0.5, 0.15),
              frameon=True, shadow=True, fontsize=11)
    
    plt.title('Monthly Cost Breakdown', fontweight='bold', pad=20)
    
    # Add total cost text in the center
    plt.annotate(f"Total Monthly Cost:\n${total_cost:.2f}", 
                xy=(0, 0), xycoords='data',
                ha='center', va='center',
                fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(vis_dir / 'cost_breakdown.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Enhanced Price Frontier Chart
    plt.figure(figsize=(12, 7))
    
    # Calculate minimum viable price for different utilization levels
    x_values = price_frontier_data["utilization_rates"]
    y_values = [p * 100 for p in price_frontier_data["min_prices"]]  # Convert to cents
    
    # Add reference line for current price
    current_price_cents = params["charge_per_minute"] * 100
    
    # Add gradient background for different profitability zones
    plt.fill_between(x_values, y_values, 0, 
                    color='#a2d9ff', alpha=0.4, 
                    label='Profit Zone',
                    interpolate=True)
    plt.fill_between(x_values, y_values, [max(y_values) * 1.1] * len(x_values), 
                    color='#ffcccc', alpha=0.3,
                    label='Loss Zone')
    
    # Plot the minimum viable price line with enhanced styling
    plt.plot(x_values, y_values, 
             color='#3498db', linewidth=3, marker='o', 
             markersize=6, markerfacecolor='white', markeredgewidth=2)
    
    # Add horizontal line for current price
    plt.axhline(y=current_price_cents, color='#27ae60', linestyle='--', 
                linewidth=2, label=f'Current Price: {current_price_cents:.1f}¢')
    
    # Add vertical line for current utilization
    plt.axvline(x=params["utilization_rate"], color='#e74c3c', linestyle='--', 
                linewidth=2, label=f'Current Utilization: {params["utilization_rate"]} hrs')
    
    # Mark current operation point
    current_min_price = None
    for i, hours in enumerate(x_values):
        if hours == params["utilization_rate"]:
            current_min_price = y_values[i]
            break
    
    if current_min_price:
        plt.scatter([params["utilization_rate"]], [current_min_price], 
                   s=100, color='black', zorder=5)
        
        # Add annotation for minimum price at current utilization
        plt.annotate(f'Min Price: {current_min_price:.1f}¢', 
                     (params["utilization_rate"], current_min_price),
                     xytext=(10, -30), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))
        
        # Add indication of profit margin
        margin = ((current_price_cents - current_min_price) / current_min_price) * 100 if current_min_price > 0 else float('inf')
        if margin > 0:
            plt.annotate(f'Margin: {margin:.1f}%', 
                        (params["utilization_rate"], (current_price_cents + current_min_price)/2),
                        xytext=(20, 0), textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Minimum Price Required for Profitability', fontweight='bold')
    plt.xlabel('Daily Utilization (hours)')
    plt.ylabel('Minimum Price (cents per minute)')
    
    # Add better annotations to explain the chart
    min_price = min(y_values)
    max_price = max(y_values)
    plt.annotate('Higher utilization\nreduces minimum\nviable price', 
                xy=(20, min_price * 1.1), 
                xytext=(18, min_price * 2), 
                arrowprops=dict(arrowstyle='->'),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Enhanced legend with shadow
    legend = plt.legend(loc='upper right', framealpha=0.9, shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('gray')
    
    plt.ylim(0, max(y_values) * 1.2)
    plt.tight_layout()
    plt.savefig(vis_dir / 'price_frontier.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Enhanced Profitability Heatmaps
    # Define combinations to generate
    heatmap_combinations = [
        ('utilization', 'price'),
        ('utilization', 'threads'),
        ('price', 'threads')
    ]
    
    # Define a custom colormap for better visualization of profits and losses
    from matplotlib.colors import LinearSegmentedColormap
    profit_colors = [(0.8, 0.1, 0.1), (1, 1, 0.8), (0.1, 0.6, 0.1)]
    profit_cmap = LinearSegmentedColormap.from_list('profit_cmap', profit_colors, N=100)
    
    param_labels = {
        "utilization": "Daily Utilization (hours)",
        "price": "Price per Minute ($)",
        "threads": "Concurrent Threads"
    }
    
    for x_param, y_param in heatmap_combinations:
        heatmap_data = generate_profitability_heatmap_data(params, x_param, y_param)
        
        # Create the enhanced heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Normalize profit values to create better visualization
        profit_matrix = np.array(heatmap_data["profit_matrix"])
        max_profit = max(abs(np.min(profit_matrix)), abs(np.max(profit_matrix)))
        normalized_matrix = profit_matrix / max_profit if max_profit > 0 else profit_matrix
        
        # Create heatmap with enhanced styling
        im = ax.imshow(normalized_matrix, cmap=profit_cmap, aspect='auto', vmin=-1, vmax=1, interpolation='nearest')
        
        # Add grid lines for better readability
        ax.set_xticks(np.arange(-.5, len(heatmap_data["x_values"]), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(heatmap_data["y_values"]), 1), minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5, alpha=0.2)
        
        # Set labels for axes
        ax.set_xticks(np.arange(len(heatmap_data["x_values"])))
        ax.set_yticks(np.arange(len(heatmap_data["y_values"])))
        
        # Only show a subset of labels if there are too many
        x_label_indices = np.linspace(0, len(heatmap_data["x_values"])-1, min(10, len(heatmap_data["x_values"]))).astype(int)
        y_label_indices = np.linspace(0, len(heatmap_data["y_values"])-1, min(10, len(heatmap_data["y_values"]))).astype(int)
        
        ax.set_xticks(x_label_indices)
        ax.set_yticks(y_label_indices)
        
        ax.set_xticklabels([heatmap_data["x_values"][i] for i in x_label_indices])
        ax.set_yticklabels([heatmap_data["y_values"][i] for i in y_label_indices])
        
        ax.set_xlabel(param_labels[x_param], fontweight='bold')
        ax.set_ylabel(param_labels[y_param], fontweight='bold')
        
        # Add profit values in the cells with improved visibility
        for i in range(len(heatmap_data["y_values"])):
            for j in range(len(heatmap_data["x_values"])):
                # Only show values for a subset of cells to avoid clutter
                if i % 2 == 0 and j % 2 == 0:
                    profit_val = heatmap_data["profit_matrix"][i][j]
                    # Use black or white text based on background darkness
                    text_color = 'white' if abs(normalized_matrix[i, j]) > 0.4 else 'black'
                    ax.text(j, i, f'${profit_val:.0f}', ha="center", va="center", color=text_color, 
                           fontsize=9, fontweight='bold')
        
        # Mark current parameter settings
        current_x = None
        current_y = None
        
        # Find closest indices for current parameters
        for i, val in enumerate(heatmap_data["x_values"]):
            if x_param == "utilization" and val == params["utilization_rate"]:
                current_x = i
            elif x_param == "price" and val == params["charge_per_minute"]:
                current_x = i
            elif x_param == "threads" and val == params["concurrent_threads"]:
                current_x = i
                
        for i, val in enumerate(heatmap_data["y_values"]):
            if y_param == "utilization" and val == params["utilization_rate"]:
                current_y = i
            elif y_param == "price" and val == params["charge_per_minute"]:
                current_y = i
            elif y_param == "threads" and val == params["concurrent_threads"]:
                current_y = i
        
        # Highlight current settings with a marker
        if current_x is not None and current_y is not None:
            ax.plot(current_x, current_y, 'o', markersize=12, markeredgewidth=2,
                   markerfacecolor='none', markeredgecolor='white')
            ax.plot(current_x, current_y, 'x', markersize=8, markeredgewidth=2,
                   markerfacecolor='none', markeredgecolor='white')
        
        # Add an enhanced colorbar
        cbar = plt.colorbar(im, label='Profitability (normalized)', shrink=0.8)
        cbar.ax.set_ylabel('Profitability (normalized)', fontweight='bold')
        
        # Add break-even contour line
        if np.min(profit_matrix) < 0 and np.max(profit_matrix) > 0:
            # Create a contour at zero profit to show break-even line
            CS = ax.contour(np.arange(len(heatmap_data["x_values"])), 
                           np.arange(len(heatmap_data["y_values"])), 
                           profit_matrix, levels=[0], colors='white', linewidths=2)
            ax.clabel(CS, inline=True, fontsize=10, fmt='Break-even')
        
        # Add annotations for maximum and minimum profit points
        max_i, max_j = np.unravel_index(np.argmax(profit_matrix), profit_matrix.shape)
        min_i, min_j = np.unravel_index(np.argmin(profit_matrix), profit_matrix.shape)
        
        # Annotate max profit point if it's positive
        if profit_matrix[max_i, max_j] > 0:
            ax.annotate(f'Max Profit: ${profit_matrix[max_i, max_j]:.0f}', 
                       xy=(max_j, max_i), xytext=(20, -20), 
                       textcoords='offset points', color='black',
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'),
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.title(f'Profitability Heatmap: {param_labels[y_param]} vs {param_labels[x_param]}', fontweight='bold')
        plt.tight_layout()
        plt.savefig(vis_dir / f'heatmap_{y_param}_vs_{x_param}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 5. Create GPU Model Comparison Charts
        
        # Chart 1: GPU Memory vs Max Concurrent Threads
        plt.figure(figsize=(12, 8))
        
        # Define color map for different GPUs
        gpu_colors = {
            "NVIDIA RTX 5090": "#76b900",  # NVIDIA green
            "NVIDIA RTX 5080": "#76b900",
            "NVIDIA H100": "#1E90FF",      # Blue
            "NVIDIA H200 (SXM)": "#9370DB", # Purple
            "NVIDIA GB200 (Blackwell)": "#FF8C00"  # Orange
        }
        
        # Create a bar chart showing max threads for each GPU model at the current LLM size
        gpu_names = [model["name"] for model in GPU_MODELS]
        gpu_memory = [model["memory_gb"] for model in GPU_MODELS]
        
        # Calculate max threads for each GPU with current model size
        max_threads_per_gpu = [calculate_max_concurrent_threads(mem, params["vram_required"]) for mem in gpu_memory]
        
        # Create paired bars for memory and max threads
        x = np.arange(len(GPU_MODELS))
        width = 0.35
        
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # Primary axis for memory
        bars1 = ax1.bar(x - width/2, gpu_memory, width, label='GPU Memory (GB)', color='#3498db')
        ax1.set_ylabel('Memory (GB)', fontweight='bold', color='#3498db')
        ax1.tick_params(axis='y', labelcolor='#3498db')
        
        # Secondary axis for max threads
        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + width/2, max_threads_per_gpu, width, label='Max Concurrent Threads', color='#e74c3c')
        ax2.set_ylabel('Max Concurrent Threads', fontweight='bold', color='#e74c3c')
        ax2.tick_params(axis='y', labelcolor='#e74c3c')
        
        # Set x-axis labels and title
        ax1.set_xticks(x)
        ax1.set_xticklabels(gpu_names, rotation=45, ha='right')
        plt.title(f'GPU Memory and Max Concurrent Threads (LLM Size: {params["llm_size"]}B parameters)', fontweight='bold')
        
        # Add a note about VRAM requirements
        vram_required = params["vram_required"]
        plt.figtext(0.5, 0.01, f'Model VRAM Requirement: ~{vram_required}GB per instance', 
                   ha='center', fontsize=12, style='italic')
        
        # Add value labels on top of bars
        for i, v in enumerate(gpu_memory):
            ax1.text(i - width/2, v + 5, f"{v}GB", ha='center', fontsize=9, fontweight='bold')
        
        for i, v in enumerate(max_threads_per_gpu):
            ax2.text(i + width/2, v + 0.1, str(v), ha='center', fontsize=9, fontweight='bold')
        
        # Add legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout(pad=3)
        plt.savefig(vis_dir / 'gpu_memory_threads.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Chart 2: Max Threads vs LLM Size
        plt.figure(figsize=(14, 8))
        
        # For each GPU model
        for model_name, threads in gpu_thread_data["max_threads_by_model"].items():
            color = gpu_colors.get(model_name, '#333333')
            plt.plot(gpu_thread_data["llm_sizes"], threads, label=model_name, 
                    marker='o', linewidth=2, markersize=8, color=color)
        
        # Add vertical line at current LLM size
        plt.axvline(x=params["llm_size"], color='red', linestyle='--', 
                    linewidth=2, label=f'Current LLM Size ({params["llm_size"]}B)')
        
        # Highlight the current model size
        y_min, y_max = plt.ylim()
        plt.fill_betweenx([0, y_max], params["llm_size"]-2, params["llm_size"]+2, 
                          color='red', alpha=0.1)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.title('Maximum Concurrent Threads by LLM Size and GPU Model', fontweight='bold')
        plt.xlabel('LLM Size (Billions of Parameters)')
        plt.ylabel('Maximum Concurrent Threads')
        
        # Add annotations showing the VRAM requirement estimate
        for size in llm_sizes[::2]:  # Add annotations for every other size to avoid clutter
            vram_req = gpu_thread_data["vram_requirements"][size]
            plt.annotate(f'~{vram_req}GB', 
                        xy=(size, 0.5), 
                        xytext=(0, -20), 
                        textcoords='offset points',
                        ha='center',
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Enhanced legend with shadow
        legend = plt.legend(loc='upper right', framealpha=0.9, shadow=True)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('gray')
        
        plt.tight_layout()
        plt.savefig(vis_dir / 'llm_size_threads.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Chart 3: GPU Cost vs Performance (Cost efficiency)
        plt.figure(figsize=(12, 8))
        
        # Calculate cost per thread - a measure of cost efficiency
        gpu_costs = [model["cost"] for model in GPU_MODELS]
        cost_per_thread = []
        
        for i, model in enumerate(GPU_MODELS):
            threads = max_threads_per_gpu[i]
            cost = model["cost"]
            # Avoid division by zero
            if threads > 0:
                cost_per_thread.append(cost / threads)
            else:
                cost_per_thread.append(float('inf'))
        
        # Create a horizontal bar chart
        y_pos = np.arange(len(GPU_MODELS))
        
        # Create bars with color gradient based on efficiency
        norm = plt.Normalize(min(cost_per_thread), max(cost_per_thread))
        colors = plt.cm.viridis_r(norm(cost_per_thread))
        
        bars = plt.barh(y_pos, cost_per_thread, color=colors)
        
        plt.yticks(y_pos, gpu_names)
        plt.xlabel('Cost per Concurrent Thread ($)')
        plt.title('GPU Cost Efficiency (Lower is Better)', fontweight='bold')
        
        # Add value labels
        for i, v in enumerate(cost_per_thread):
            if v == float('inf'):
                label = "∞"
            else:
                label = f"${v:,.0f}"
            plt.text(v + 1000, i, label, va='center', fontweight='bold')
            
            # Add thread count and total cost as additional info
            threads = max_threads_per_gpu[i]
            cost = gpu_costs[i]
            plt.text(v/2, i, f"{threads} threads | ${cost:,.0f}", 
                    va='center', ha='center', color='white', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(vis_dir / 'gpu_cost_efficiency.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Create a JSON dump of all the calculated values
    results = {
        "monthly_revenue": monthly_revenue,
        "monthly_costs": monthly_costs,
        "profitability": profitability,
        "price_frontier": price_frontier_data,
        "profit_over_time": profit_over_time_data,
        "gpu_thread_data": gpu_thread_data,
        "selected_gpu_model": gpu_model,
        "constrained_threads": constrained_threads
    }
    
    with open(vis_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def format_breakeven_time(months):
    """Format break-even time in a human-readable format"""
    if months == float('inf'):
        return "Never"
    
    years = int(months / 12)
    remaining_months = int(months % 12)
    
    if years > 0 and remaining_months > 0:
        return f"{years} years, {remaining_months} months"
    elif years > 0:
        return f"{years} years"
    else:
        return f"{remaining_months} months"

class ResultHTMLGenerator:
    """
    Generate HTML and JavaScript with pre-calculated results
    
    This class handles:
    1. Generating JavaScript that loads pre-calculated results into the UI
    2. Setting up image loading for visualizations
    3. Enhancing the web interface with animations and transitions
    4. Modifying the index.html to include required scripts
    """
    
    def __init__(self, results, params):
        self.results = results
        self.params = params
    
    def generate_results_js(self):
        """
        Generate a JavaScript file with pre-calculated results and enhanced visualization loading
        
        This creates a comprehensive JS file that:
        - Loads all pre-calculated data into the UI
        - Sets up visualization display with proper sizing
        - Handles responsive behavior for charts
        - Implements smooth transitions and animations
        """
        js_code = f"""
// Pre-calculated results from Python
const preCalculatedResults = {json.dumps(self.results, indent=2)};

// Function to update UI with pre-calculated results
function loadPreCalculatedResults() {{
    // Add animation classes for smooth transitions
    document.querySelectorAll('.result-item').forEach(item => {{
        item.classList.add('animate-in');
    }});
    
    // Update revenue and costs with animated counting
    animateCounter('monthly-revenue', 0, preCalculatedResults.monthly_revenue);
    animateCounter('monthly-costs', 0, preCalculatedResults.monthly_costs.total_monthly_cost);
    animateCounter('monthly-profit', 0, preCalculatedResults.profitability.monthly_profit);
    animateCounter('profit-margin', 0, preCalculatedResults.profitability.profit_margin);
    
    // Break-even analysis
    const breakEvenTime = preCalculatedResults.profitability.break_even_time_months;
    document.getElementById('breakeven-time').textContent = breakEvenTime === Infinity ? 'Never' : 
        Math.floor(breakEvenTime) + ' months';
    animateCounter('lifetime-roi', 0, preCalculatedResults.profitability.roi);
    
    // Set form values to match the calculation parameters
    document.getElementById('charge-per-minute').value = {self.params["charge_per_minute"]};
    document.getElementById('utilization-rate').value = {self.params["utilization_rate"]};
    document.getElementById('concurrent-threads').value = {self.params["concurrent_threads"]};
    document.getElementById('llm-size').value = {self.params["llm_size"]};
    document.getElementById('inference-time').value = {self.params["inference_time"]};
    document.getElementById('vram-required').value = {self.params["vram_required"]};
    document.getElementById('gpu-cost').value = {self.params["gpu_cost"]};
    document.getElementById('cpu-cost').value = {self.params["cpu_cost"]};
    document.getElementById('ram-cost').value = {self.params["ram_cost"]};
    document.getElementById('other-hardware').value = {self.params["other_hardware"]};
    document.getElementById('hardware-lifespan').value = {self.params["hardware_lifespan"]};
    document.getElementById('electricity-cost').value = {self.params["electricity_cost"]};
    document.getElementById('power-consumption').value = {self.params["power_consumption"]};
    document.getElementById('maintenance-cost').value = {self.params["maintenance_cost"]};
    
    // Update ROI calculator UI elements with animated transitions
    document.getElementById('roi-utilization-value').textContent = {self.params["utilization_rate"]};
    document.getElementById('roi-utilization').value = {self.params["utilization_rate"]};
    document.getElementById('roi-price-value').textContent = {self.params["charge_per_minute"]};
    document.getElementById('roi-price').value = {self.params["charge_per_minute"]};
    document.getElementById('roi-threads-value').textContent = {self.params["concurrent_threads"]};
    document.getElementById('roi-threads').value = {self.params["concurrent_threads"]};
    
    const roi = preCalculatedResults.profitability.roi;
    document.getElementById('roi-value').textContent = roi.toFixed(1);
    document.getElementById('roi-breakeven').textContent = '{format_breakeven_time(self.results["profitability"]["break_even_time_months"])}';
    
    // Set ROI meter fill with animation
    const roiMeterFill = document.getElementById('roi-meter-fill');
    const roiPercentage = Math.min(Math.max(roi / 200, 0), 1) * 100;
    roiMeterFill.style.backgroundColor = roi > 0 ? '#4CAF50' : '#F44336';
    
    // Animate the ROI meter fill
    setTimeout(() => {{
        roiMeterFill.style.width = roiPercentage + '%';
    }}, 300);
    
    // Display GPU model and VRAM constraints info
    if (preCalculatedResults.selected_gpu_model) {{
        const gpuModel = preCalculatedResults.selected_gpu_model;
        
        // Create or update GPU info section
        let gpuInfoDiv = document.getElementById('gpu-info-section');
        if (!gpuInfoDiv) {{
            gpuInfoDiv = document.createElement('div');
            gpuInfoDiv.id = 'gpu-info-section';
            gpuInfoDiv.className = 'result-card';
            
            const resultsPanel = document.querySelector('.results-panel');
            if (resultsPanel) {{
                resultsPanel.insertBefore(gpuInfoDiv, resultsPanel.firstChild);
            }}
        }}
        
        // Update GPU info content
        gpuInfoDiv.innerHTML = `
            <h3>GPU Model Information</h3>
            <div class="gpu-info-container">
                <div class="gpu-specs">
                    <div class="gpu-model-name">${{gpuModel.name}}</div>
                    <div class="gpu-specs-details">
                        <div class="spec-item">
                            <span class="spec-label">Memory:</span>
                            <span class="spec-value">${{gpuModel.memory_gb}} GB</span>
                        </div>
                        <div class="spec-item">
                            <span class="spec-label">Cost:</span>
                            <span class="spec-value">$${{gpuModel.cost.toLocaleString()}}</span>
                        </div>
                        <div class="spec-item">
                            <span class="spec-label">Power:</span>
                            <span class="spec-value">${{gpuModel.power_w}} W</span>
                        </div>
                    </div>
                </div>
                <div class="gpu-constraint-info">
                    <div class="constraint-item">
                        <span class="constraint-label">Model VRAM Requirement:</span>
                        <span class="constraint-value">${{self.params.vram_required}} GB</span>
                    </div>
                    <div class="constraint-item">
                        <span class="constraint-label">Desired Threads:</span>
                        <span class="constraint-value">${{self.params.concurrent_threads}}</span>
                    </div>
                    <div class="constraint-item">
                        <span class="constraint-label">Hardware Constrained Threads:</span>
                        <span class="constraint-value">${{preCalculatedResults.constrained_threads}}</span>
                    </div>
                </div>
            </div>
        `;
    }}
}}

// Animation function for counting up numbers
function animateCounter(elementId, start, end) {{
    const element = document.getElementById(elementId);
    if (!element) return;
    
    const duration = 1500; // milliseconds
    const startTime = performance.now();
    const isPercentage = elementId.includes('roi') || elementId.includes('margin');
    const prefix = elementId.includes('revenue') || elementId.includes('cost') || elementId.includes('profit') ? '$' : '';
    const suffix = isPercentage ? '%' : '';
    const decimals = isPercentage || elementId.includes('roi') ? 1 : 2;
    
    function updateNumber(timestamp) {{
        const elapsed = timestamp - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function for smoother animation
        const eased = progress < 0.5 ? 4 * progress * progress * progress : 
                      1 - Math.pow(-2 * progress + 2, 3) / 2;
                      
        const currentValue = start + (end - start) * eased;
        element.textContent = `${{prefix}}${{currentValue.toFixed(decimals)}}${{suffix}}`;
        
        if (progress < 1) {{
            requestAnimationFrame(updateNumber);
        }}
    }}
    
    requestAnimationFrame(updateNumber);
}}

// Replace chart images with pre-generated PNGs and handle responsive behavior
function loadChartImages() {{
    // Get chart containers
    const chartContainers = document.querySelectorAll('.chart-card');
    
    // Map of canvas IDs to image paths
    const chartImageMap = {{
        'profit-chart': 'visualizations/profit_over_time.png',
        'cost-breakdown-chart': 'visualizations/cost_breakdown.png',
        'price-frontier-chart': 'visualizations/price_frontier.png',
        'profitability-heatmap': 'visualizations/heatmap_price_vs_utilization.png'
    }};
    
    // Add new GPU-related chart images to the map
    const gpuChartImageMap = {{
        'gpu-memory-threads': 'visualizations/gpu_memory_threads.png',
        'llm-size-threads': 'visualizations/llm_size_threads.png',
        'gpu-cost-efficiency': 'visualizations/gpu_cost_efficiency.png'
    }};
    
    // Merge all charts
    Object.assign(chartImageMap, gpuChartImageMap);
    
    // Replace each canvas with its corresponding image
    chartContainers.forEach(container => {{
        const canvas = container.querySelector('canvas');
        if (canvas && chartImageMap[canvas.id]) {{
            replaceCanvasWithImage(canvas, chartImageMap[canvas.id]);
            
            // Add fade-in animation class
            container.classList.add('fade-in');
        }}
    }});
    
    // Add GPU visualization section
    const resultsPanel = document.querySelector('.results-panel');
    if (resultsPanel) {{
        // Create GPU visualizations section header
        const gpuHeader = document.createElement('h2');
        gpuHeader.className = 'section-title fade-in';
        gpuHeader.textContent = 'GPU Model Analysis';
        resultsPanel.appendChild(gpuHeader);
        
        // Add GPU visualization cards
        const gpuVisKeys = Object.keys(gpuChartImageMap);
        const gpuVisTitles = {{
            'gpu-memory-threads': 'GPU Memory & Maximum Threads',
            'llm-size-threads': 'LLM Size vs Available Threads',
            'gpu-cost-efficiency': 'GPU Cost Efficiency'
        }};
        
        const gpuVisDescriptions = {{
            'gpu-memory-threads': 'Comparison of GPU memory capacity and the maximum concurrent threads possible for the current model size.',
            'llm-size-threads': 'How different LLM sizes affect the number of concurrent threads each GPU can handle.',
            'gpu-cost-efficiency': 'Cost per thread analysis showing which GPUs provide the best value for running this workload.'
        }};
        
        gpuVisKeys.forEach(chartId => {{
            const card = document.createElement('div');
            card.className = 'result-card chart-card fade-in';
            card.innerHTML = `
                <h3>${{gpuVisTitles[chartId]}}</h3>
                <div class="chart-container">
                    <img src="${{gpuChartImageMap[chartId]}}" class="pre-generated-chart" alt="${{gpuVisTitles[chartId]}}">
                </div>
                <p class="chart-description">${{gpuVisDescriptions[chartId]}}</p>
            `;
            
            resultsPanel.appendChild(card);
        }});
    }}
    
    // Set up heatmap controls with enhanced behavior
    const generateHeatmapBtn = document.getElementById('generate-heatmap');
    const xAxisSelect = document.getElementById('heatmap-x-axis');
    const yAxisSelect = document.getElementById('heatmap-y-axis');
    
    if (generateHeatmapBtn && xAxisSelect && yAxisSelect) {{
        generateHeatmapBtn.addEventListener('click', function() {{
            const xAxis = xAxisSelect.value;
            const yAxis = yAxisSelect.value;
            
            // Prevent same axis selection
            if (xAxis === yAxis) {{
                alert('Please select different parameters for X and Y axes');
                return;
            }}
            
            // Get the heatmap canvas
            const heatmapCanvas = document.getElementById('profitability-heatmap');
            if (!heatmapCanvas) return;
            
            // Add loading indicator
            const loadingIndicator = document.getElementById('heatmap-loading-indicator');
            if (loadingIndicator) loadingIndicator.style.display = 'block';
            
            // Add fade-out effect to current image
            const currentImg = heatmapCanvas.parentNode.querySelector('img');
            if (currentImg) {{
                currentImg.style.opacity = '0.3';
            }}
            
            // Load the appropriate pre-generated heatmap with a small delay for visual effect
            setTimeout(() => {{
                replaceCanvasWithImage(heatmapCanvas, `visualizations/heatmap_${{yAxis}}_vs_${{xAxis}}.png`);
                
                // Hide loading indicator
                if (loadingIndicator) loadingIndicator.style.display = 'none';
                
                // Apply fade-in effect to new image
                const newImg = heatmapCanvas.parentNode.querySelector('img');
                if (newImg) {{
                    newImg.style.opacity = '0.3';
                    setTimeout(() => {{ newImg.style.opacity = '1'; }}, 50);
                }}
            }}, 500);
        }});
    }}
}}

// Helper function to replace a canvas with an image
function replaceCanvasWithImage(canvas, imagePath) {{
    if (!canvas) return;
    
    const img = new Image();
    img.src = imagePath;
    img.className = 'pre-generated-chart';
    img.alt = canvas.id.replace('-chart', '').replace('-', ' ');
    
    // Set responsive behavior
    img.style.width = '100%';
    img.style.height = 'auto';
    img.style.maxWidth = '100%';
    img.style.maxHeight = '600px';
    img.style.objectFit = 'contain';
    
    // Add transition for smooth loading
    img.style.transition = 'opacity 0.3s ease';
    
    // Handle error if image doesn't exist
    img.onerror = function() {{
        console.warn(`Image ${{imagePath}} not found`);
        // Keep the canvas if image isn't found
    }};
    
    // Apply fade-in effect
    img.style.opacity = '0';
    
    const parent = canvas.parentNode;
    parent.replaceChild(img, canvas);
    
    // Trigger fade-in
    setTimeout(() => {{ img.style.opacity = '1'; }}, 50);
}}

// Helper function to check if a file exists (for animations)
function fileExists(url) {{
    var http = new XMLHttpRequest();
    http.open('HEAD', url, false);
    try {{
        http.send();
        return http.status !== 404;
    }} catch(e) {{
        return false;
    }}
}}

// Add CSS for animations and transitions
function addAnimationStyles() {{
    const style = document.createElement('style');
    style.textContent = `
        .animate-in {{
            animation: fadeIn 0.8s ease-out forwards;
        }}
        
        .fade-in {{
            animation: fadeIn 0.8s ease-out forwards;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .pre-generated-chart {{
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border-radius: 4px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .pre-generated-chart:hover {{
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }}
        
        #roi-meter-fill {{
            transition: width 1.5s cubic-bezier(0.19, 1, 0.22, 1), 
                        background-color 1s ease;
        }}
        
        .section-title {{
            width: 100%;
            text-align: center;
            margin: 40px 0 20px;
            font-size: 24px;
            color: #333;
            position: relative;
        }}
        
        .section-title::after {{
            content: '';
            position: absolute;
            bottom: -10px;
            left: 50%;
            transform: translateX(-50%);
            width: 100px;
            height: 3px;
            background: linear-gradient(90deg, #3498db, #27ae60);
        }}
        
        /* GPU info styling */
        .gpu-info-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 15px;
        }}
        
        .gpu-specs {{
            flex: 1;
            min-width: 250px;
            background: linear-gradient(135deg, #76b900 0%, #5a8e00 100%);
            border-radius: 8px;
            padding: 15px;
            color: white;
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}
        
        .gpu-model-name {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.3);
            padding-bottom: 5px;
        }}
        
        .gpu-specs-details {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        
        .spec-item {{
            display: flex;
            justify-content: space-between;
        }}
        
        .gpu-constraint-info {{
            flex: 1;
            min-width: 250px;
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            border-radius: 8px;
            padding: 15px;
            color: white;
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        
        .constraint-item {{
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }}
        
        .constraint-item:last-child {{
            border-bottom: none;
        }}
    `;
    document.head.appendChild(style);
}}

// Initialize everything when the page loads
document.addEventListener('DOMContentLoaded', function() {{
    // Add animation styles
    addAnimationStyles();
    
    // Load results with slight delay for visual effect
    setTimeout(() => {{
        loadPreCalculatedResults();
        loadChartImages();
    }}, 300);
    
    // Disable the calculate button - we're using pre-calculated results
    const calculateBtn = document.getElementById('calculate-btn');
    if (calculateBtn) {{
        calculateBtn.textContent = 'Results Pre-calculated';
        
        calculateBtn.addEventListener('click', function(e) {{
            e.preventDefault();
            alert('This demo uses pre-calculated results. Modify run.py parameters and re-run to see different calculations.');
        }});
    }}
    
    // Make ROI slider controls work interactively
    const roiSliders = document.querySelectorAll('.roi-calculator input[type="range"]');
    roiSliders.forEach(slider => {{
        slider.addEventListener('input', function() {{
            // Update displayed value
            const valueSpan = document.getElementById(`${{this.id}}-value`);
            if (valueSpan) valueSpan.textContent = this.value;
            
            // Calculate new ROI (simplified - normally would recalculate properly)
            const utilizationValue = parseFloat(document.getElementById('roi-utilization').value);
            const priceValue = parseFloat(document.getElementById('roi-price').value);
            const threadsValue = parseFloat(document.getElementById('roi-threads').value);
            
            // For demo purposes, simulate ROI change based on sliders
            // In a real implementation, this would recalculate properly
            const baseRoi = preCalculatedResults.profitability.roi;
            const utilizationFactor = utilizationValue / {self.params["utilization_rate"]};
            const priceFactor = priceValue / {self.params["charge_per_minute"]};
            const threadsFactor = threadsValue / {self.params["concurrent_threads"]};
            
            const estimatedRoi = baseRoi * utilizationFactor * priceFactor * threadsFactor;
            
            // Update ROI display
            document.getElementById('roi-value').textContent = estimatedRoi.toFixed(1);
            
            // Update meter
            const roiMeterFill = document.getElementById('roi-meter-fill');
            const roiPercentage = Math.min(Math.max(estimatedRoi / 200, 0), 1) * 100;
            roiMeterFill.style.width = roiPercentage + '%';
            roiMeterFill.style.backgroundColor = estimatedRoi > 0 ? '#4CAF50' : '#F44336';
        }});
    }});
}});
"""
        
        # Create the JS directory if it doesn't exist
        js_dir = Path('js/precalculated')
        js_dir.mkdir(exist_ok=True, parents=True)
        
        # Write the JS file
        with open(js_dir / 'results.js', 'w') as f:
            f.write(js_code)
        
        # Create an additional CSS file for enhanced visualizations
        css_dir = Path('css')
        css_dir.mkdir(exist_ok=True)
        
        css_code = """
/* Enhanced visualization styles */
.chart-card {
    transition: all 0.3s ease;
}

.chart-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 16px rgba(0,0,0,0.15);
}

.pre-generated-chart {
    display: block;
    margin: 0 auto;
}

.result-card {
    overflow: hidden;
}

.roi-meter-container {
    position: relative;
    margin-top: 20px;
}

.roi-meter {
    height: 20px;
    background-color: #f3f3f3;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);
    margin-bottom: 10px;
}

.roi-meter-fill {
    height: 100%;
    width: 0;
    border-radius: 10px;
    transition: width 1.5s cubic-bezier(0.19, 1, 0.22, 1);
}

.roi-value, .roi-time {
    font-weight: bold;
    margin-top: 5px;
}

.animate-in {
    opacity: 0;
    animation: fadeIn 0.5s ease-out forwards;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Enhanced loading indicator */
.loading {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 20px;
    font-style: italic;
    color: #666;
}

.loading::after {
    content: '';
    width: 20px;
    height: 20px;
    margin-left: 10px;
    border: 3px solid #ddd;
    border-top: 3px solid #3498db;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Animation container */
.animation-container {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 15px 0;
    overflow: hidden;
    border-radius: 4px;
}

.animated-chart {
    max-width: 100%;
    height: auto;
    border-radius: 4px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
"""
        
        # Write the CSS file if it doesn't exist
        css_file = css_dir / 'styles.css'
        if not css_file.exists():
            with open(css_file, 'w') as f:
                f.write(css_code)
        else:
            # Append our styles to the existing file
            with open(css_file, 'a') as f:
                f.write("\n\n/* Enhanced visualization styles added by generator */\n")
                f.write(css_code)
    
    def modify_index_html(self):
        """
        Modify index.html to include pre-calculated results and enhanced visualization script
        
        This ensures proper placement of visualizations in the HTML document by:
        1. Adding required script references
        2. Setting up CSS for responsive behavior
        3. Creating a backup of the original HTML
        """
        try:
            with open('index.html', 'r') as f:
                html_content = f.read()
            
            # Create a backup of the original HTML
            with open('index.html.bak', 'w') as f:
                f.write(html_content)
            
            # Add our pre-calculated results script before the main.js
            if '<script src="js/precalculated/results.js"></script>' not in html_content:
                new_html_content = html_content.replace(
                    '<script type="module" src="js/main.js"></script>',
                    '<script src="js/precalculated/results.js"></script>\n    <script type="module" src="js/main.js"></script>'
                )
                
                # Write the modified HTML
                with open('index.html', 'w') as f:
                    f.write(new_html_content)
                
                return True
            
            return False  # No changes made - script already exists
        except Exception as e:
            print(f"Error modifying HTML: {e}")
            return False

def find_available_port(start_port=8000, max_attempts=20):
    """
    Find an available port to serve the website
    
    Args:
        start_port: Initial port to try
        max_attempts: Maximum number of ports to try
        
    Returns:
        Available port number or None if no port is available
    """
    port = start_port
    attempts = 0
    
    while attempts < max_attempts:
        try:
            with socketserver.TCPServer(("", port), None) as server:
                server.server_close()
                return port
        except OSError:
            print(f"Port {port} is in use, trying next port...")
            port += 1
            attempts += 1
    
    return None

def start_server(port):
    """
    Start the HTTP server on the specified port
    
    Args:
        port: Port number to start the server on
    """
    class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        # Override log_message to provide more useful logging
        def log_message(self, format, *args):
            # Check if args is a string (GET requests) vs status code object
            if args and isinstance(args[0], str) and args[0].startswith("GET"):
                sys.stdout.write(f"\r\033[KServing: {self.path}\n")
                sys.stdout.write("Press Ctrl+C to stop the server\n")
                sys.stdout.flush()
            else:
                # Default logging for other messages (like errors)
                sys.stdout.write(f"\r\033[K{format % args}\n")
                sys.stdout.flush()
    
    class QuietServer(socketserver.TCPServer):
        allow_reuse_address = True
    
    httpd = QuietServer(("", port), CustomHTTPRequestHandler)
    
    print(f"Starting HTTP server on port {port}...")
    print(f"Access the calculator at: \033[94mhttp://localhost:{port}\033[0m")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        httpd.server_close()

def main():
    """
    Main function to run the calculator and start the server
    
    This function:
    1. Sets up required directories
    2. Generates visualizations with current parameters
    3. Displays key results in the terminal
    4. Creates web interface with pre-calculated results
    5. Starts a local server to view the results
    6. Opens the browser to display the results dashboard
    """
    # Process command line arguments
    parser = argparse.ArgumentParser(description='Infra-Calc: LLM Infrastructure Economics Calculator')
    parser.add_argument('--gpu', type=int, default=1, help='GPU model number (1-5)')
    args = parser.parse_args()
    
    # Validate GPU selection
    selected_index = args.gpu - 1
    if selected_index < 0 or selected_index >= len(GPU_MODELS):
        print(f"Invalid GPU model number, using default (1)")
        selected_index = 0
    
    print("\n" + "="*80)
    print(f"{'Infra-Calc: LLM Infrastructure Economics Calculator':^80}")
    print(f"{'GPU Model Comparison Edition':^80}")
    print("="*80)
    
    # Create required directories
    for directory in ['visualizations', 'css', 'js/precalculated']:
        dir_path = Path(directory)
        dir_path.mkdir(exist_ok=True, parents=True)
        print(f"✓ Ensuring directory exists: {directory}")
    
    # Show GPU model options (for reference only)
    print("\nAvailable GPU Models:")
    for i, model in enumerate(GPU_MODELS):
        memory_color = "\033[92m" if model["memory_gb"] >= DEFAULT_PARAMS["vram_required"] else "\033[91m"
        marker = "→" if i == selected_index else " "
        print(f"{marker} {i+1}. {model['name']} - {memory_color}{model['memory_gb']}GB\033[0m VRAM, ${model['cost']:,}, {model['power_w']}W")

    # Use the selected GPU model
    selected_gpu = GPU_MODELS[selected_index]
    
    # Print selected GPU model in a colored box
    model_name_color = "\033[1;92m" if selected_gpu["memory_gb"] >= DEFAULT_PARAMS["vram_required"] else "\033[1;93m"
    print(f"\n┌{'─'*60}┐")
    print(f"│{f'Selected GPU: {model_name_color}{selected_gpu['name']}\033[0m':^60}│")
    print(f"├{'─'*60}┤")
    print(f"│ {'Memory:':<15} {selected_gpu['memory_gb']} GB{' '*35}│")
    print(f"│ {'Cost:':<15} ${selected_gpu['cost']:,}{' '*(41-len(str(selected_gpu['cost'])))}│")
    print(f"│ {'Power:':<15} {selected_gpu['power_w']} W{' '*35}│")
    print(f"└{'─'*60}┘")
    
    # Calculate max threads based on VRAM constraints
    max_threads = calculate_max_concurrent_threads(selected_gpu["memory_gb"], DEFAULT_PARAMS["vram_required"])
    desired_threads = DEFAULT_PARAMS["concurrent_threads"]
    actual_threads = min(desired_threads, max_threads)
    
    # Show warning if thread count is constrained
    if actual_threads < desired_threads:
        print(f"\n\033[1;93m⚠️  VRAM CONSTRAINT DETECTED\033[0m")
        print(f"   Model requires {DEFAULT_PARAMS['vram_required']}GB per instance")
        print(f"   Desired threads: {desired_threads}, but GPU allows max: {max_threads}")
        print(f"   Using {actual_threads} threads for calculations\n")
    else:
        print(f"\n\033[1;92m✓ No VRAM constraints\033[0m - GPU can run all {desired_threads} desired threads\n")
    
    print("Generating visualizations with selected GPU model...")
    
    try:
        # Calculate results with default parameters and selected GPU
        results = create_visualizations(DEFAULT_PARAMS, selected_gpu)
        
        # Format and display key results
        monthly_revenue = results["monthly_revenue"]
        monthly_cost = results["monthly_costs"]["total_monthly_cost"]
        monthly_profit = results["profitability"]["monthly_profit"]
        break_even_time = results["profitability"]["break_even_time_months"]
        roi = results["profitability"]["roi"]
        
        # Use box drawing characters for a nicer table
        print("\n┌" + "─"*60 + "┐")
        print(f"│{'Key Results':^60}│")
        print("├" + "─"*60 + "┤")
        print(f"│ {'Monthly Revenue:':<30} ${monthly_revenue:,.2f} {'':>15}│")
        print(f"│ {'Monthly Costs:':<30} ${monthly_cost:,.2f} {'':>15}│")
        print(f"│ {'Monthly Profit:':<30} ${monthly_profit:,.2f} {'':>15}│")
        print(f"│ {'Break-even Time:':<30} {format_breakeven_time(break_even_time)} {'':>15}│")
        print(f"│ {'ROI over Hardware Lifespan:':<30} {roi:.1f}% {'':>15}│")
        print("└" + "─"*60 + "┘")
        
        # Add profitability summary with color coding
        if monthly_profit > 0:
            profit_status = "PROFITABLE"
            profit_color = "\033[1;92m"  # Bright green
            
            # Add some emoji and more context based on ROI
            if roi > 150:
                emoji = "🚀"
                context = "Excellent ROI"
            elif roi > 100:
                emoji = "💰"
                context = "Strong ROI"
            elif roi > 50:
                emoji = "✅"
                context = "Good ROI"
            else:
                emoji = "👍"
                context = "Positive ROI"
        else:
            profit_status = "NOT PROFITABLE"
            profit_color = "\033[1;91m"  # Bright red
            emoji = "❌"
            context = "Negative ROI"
            
        reset_color = "\033[0m"
        
        print(f"\nProfitability Status: {profit_color}{emoji} {profit_status} - {context}{reset_color}")
        
        # Add model size and thread analysis
        model_size_gb = DEFAULT_PARAMS["vram_required"]
        model_size_b = DEFAULT_PARAMS["llm_size"]
        
        # Show thread capacity of different models for current LLM size
        print("\nThread capacity for this LLM size across different GPUs:")
        print("┌" + "─"*70 + "┐")
        print(f"│ {'GPU Model':<25} {'Memory':<10} {'Threads':<10} {'Cost per Thread':<20}│")
        print("├" + "─"*70 + "┤")
        
        for model in GPU_MODELS:
            threads = calculate_max_concurrent_threads(model["memory_gb"], model_size_gb)
            
            # Calculate cost per thread
            if threads > 0:
                cost_per_thread = model["cost"] / threads
                cost_str = f"${cost_per_thread:,.0f}"
            else:
                cost_str = "N/A"
                
            # Highlight the selected model
            prefix = "→ " if model["name"] == selected_gpu["name"] else "  "
            
            # Color code based on thread capacity
            if threads >= desired_threads:
                thread_color = "\033[92m"  # Green
            elif threads > 0:
                thread_color = "\033[93m"  # Yellow
            else:
                thread_color = "\033[91m"  # Red
                
            reset = "\033[0m"
            
            print(f"│ {prefix}{model['name']:<23} {model['memory_gb']:<10} {thread_color}{threads:<10}{reset} {cost_str:<20}│")
            
        print("└" + "─"*70 + "┘")
        
        # Generate HTML and JS with pre-calculated results
        print("\nGenerating web interface with pre-calculated results...")
        html_generator = ResultHTMLGenerator(results, DEFAULT_PARAMS)
        html_generator.generate_results_js()
        if html_generator.modify_index_html():
            print("✓ Updated index.html to include pre-calculated results")
        else:
            print("✓ index.html already contains required scripts")
        
        # Count visualization files generated
        vis_files = list(vis_dir.glob('*.png')) + list(vis_dir.glob('*.gif'))
        print(f"✓ Generated {len(vis_files)} visualization files")
        
        # Find an available port and start the server
        port = find_available_port()
        if port is None:
            print("\n❌ Error: Could not find an available port. Please check your network settings.")
            return
        
        # Open the browser
        url = f"http://localhost:{port}"
        print(f"\n✓ Opening calculator in browser: {url}")
        
        # Display helpful instructions
        print("\nPress Ctrl+C to stop the server when finished")
        print("-"*80)
        
        # Start the server in a separate thread
        server_thread = threading.Thread(target=start_server, args=(port,), daemon=True)
        server_thread.start()
        
        # Open the browser after a short delay
        time.sleep(1)
        webbrowser.open(url)
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down server...")
            print("Thank you for using Infra-Calc!")
        
    except Exception as e:
        print(f"\n❌ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 