/**
 * Heatmap visualization module for Infra-Calc
 */

import { calculateAll } from './calculator.js';
import { formatCurrency, formatTime } from './charts.js';
import { createChart } from './chart-manager.js';

/**
 * Generate data for a profitability heatmap
 * @param {Object} baseParams - Base parameters for calculation
 * @param {string} xAxis - Parameter for X axis
 * @param {string} yAxis - Parameter for Y axis
 * @returns {Object} Heatmap data
 */
export function generateHeatmapData(baseParams, xAxis, yAxis) {
    // Define ranges for different parameters
    const ranges = {
        price: { min: 0.01, max: 0.1, step: 0.01, label: 'Price per Minute ($)' },
        utilization: { min: 1, max: 24, step: 1, label: 'Daily Utilization (hours)' },
        threads: { min: 1, max: 8, step: 1, label: 'Concurrent Threads' }
    };
    
    // Generate x and y axis values
    const xValues = [];
    const yValues = [];
    
    for (let x = ranges[xAxis].min; x <= ranges[xAxis].max; x += ranges[xAxis].step) {
        xValues.push(x);
    }
    
    for (let y = ranges[yAxis].min; y <= ranges[yAxis].max; y += ranges[yAxis].step) {
        yValues.push(y);
    }
    
    // Generate data points
    const data = [];
    let minProfit = Infinity;
    let maxProfit = -Infinity;
    
    // Calculate profit for each combination of x and y
    for (let i = 0; i < yValues.length; i++) {
        for (let j = 0; j < xValues.length; j++) {
            // Create parameters by modifying the base parameters
            const params = { ...baseParams };
            
            // Set parameter values for current point
            params[xAxis] = xValues[j];
            params[yAxis] = yValues[i];
            
            // Rename some parameters to match what calculateAll expects
            if (xAxis === 'price') params.chargePerMinute = params.price;
            if (yAxis === 'price') params.chargePerMinute = params.price;
            if (xAxis === 'utilization') params.dailyUtilization = params.utilization;
            if (yAxis === 'utilization') params.dailyUtilization = params.utilization;
            if (xAxis === 'threads') params.concurrentThreads = params.threads;
            if (yAxis === 'threads') params.concurrentThreads = params.threads;
            
            // Calculate results
            const results = calculateAll(params);
            const profit = results.profitability.monthlyProfit;
            
            // Track min and max for color scaling
            minProfit = Math.min(minProfit, profit);
            maxProfit = Math.max(maxProfit, profit);
            
            // Store data point
            data.push({
                x: xValues[j],
                y: yValues[i],
                profit: profit,
                breakEven: results.profitability.breakEvenTimeMonths,
                roi: results.profitability.roi
            });
        }
    }
    
    return {
        xValues,
        yValues,
        data,
        minProfit,
        maxProfit,
        xAxisLabel: ranges[xAxis].label,
        yAxisLabel: ranges[yAxis].label
    };
}

/**
 * Create a heatmap chart
 * @param {string} canvasId - ID of the canvas element
 * @param {Object} heatmapData - Data for the heatmap
 */
export function createHeatmap(canvasId, heatmapData) {
    const { xValues, yValues, data, minProfit, maxProfit, xAxisLabel, yAxisLabel } = heatmapData;
    
    // Create datasets for the heatmap
    const datasets = [];
    
    // Group data by y value
    const groupedData = {};
    data.forEach(point => {
        if (!groupedData[point.y]) {
            groupedData[point.y] = [];
        }
        groupedData[point.y].push(point);
    });
    
    // Create one dataset per y value for the heatmap
    Object.keys(groupedData).forEach(y => {
        const points = groupedData[y].map(point => ({
            x: point.x,
            y: point.y,
            profit: point.profit,
            breakEven: point.breakEven,
            roi: point.roi
        }));
        
        datasets.push({
            label: `${y}`,
            data: points,
            parsing: {
                xAxisKey: 'x',
                yAxisKey: 'y'
            },
            borderColor: 'white',
            borderWidth: 0.5,
            pointBackgroundColor: function(context) {
                const point = context.dataset.data[context.dataIndex];
                return getHeatmapColor(point.profit, minProfit, maxProfit);
            },
            pointRadius: function(context) {
                const y = context.chart.height;
                const xCount = xValues.length;
                const yCount = yValues.length;
                // Calculate optimal point size to fill the chart area
                return Math.min(y / (yCount * 2.5), 20);
            },
            pointHoverRadius: function(context) {
                const baseRadius = this.pointRadius(context);
                return baseRadius * 1.2;
            }
        });
    });
    
    // Set up chart configuration
    const config = {
        type: 'scatter',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: {
                        title: function(context) {
                            const point = context[0].dataset.data[context[0].dataIndex];
                            return `X: ${point.x}, Y: ${point.y}`;
                        },
                        label: function(context) {
                            const point = context.dataset.data[context.dataIndex];
                            const isProfit = point.profit >= 0;
                            const labels = [
                                `Monthly ${isProfit ? 'Profit' : 'Loss'}: ${formatCurrency(Math.abs(point.profit))}`,
                                `Break-even: ${formatTime(point.breakEven)}`,
                                `ROI: ${point.roi.toFixed(2)}%`
                            ];
                            return labels;
                        }
                    }
                },
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: xAxisLabel
                    },
                    type: 'linear',
                    position: 'bottom',
                    ticks: {
                        callback: function(value) {
                            // Format price differently
                            if (xAxisLabel.includes('Price')) {
                                return formatCurrency(value);
                            }
                            return value;
                        }
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: yAxisLabel
                    },
                    ticks: {
                        callback: function(value) {
                            // Format price differently
                            if (yAxisLabel.includes('Price')) {
                                return formatCurrency(value);
                            }
                            return value;
                        }
                    }
                }
            }
        }
    };
    
    // Use the chart manager to create the chart
    return createChart(canvasId, config);
}

/**
 * Generate a color for the heatmap based on profit value
 * @param {number} profit - Profit value
 * @param {number} minProfit - Minimum profit in dataset
 * @param {number} maxProfit - Maximum profit in dataset
 * @returns {string} CSS color
 */
function getHeatmapColor(profit, minProfit, maxProfit) {
    // Red for negative, green for positive
    if (profit < 0) {
        // Range from light red to dark red for losses
        const intensity = Math.min(1, Math.abs(profit) / Math.abs(minProfit));
        const red = 255;
        const green = 100 - (intensity * 100);
        const blue = 100 - (intensity * 100);
        return `rgba(${red}, ${green}, ${blue}, 0.8)`;
    } else {
        // Range from light green to dark green for profits
        const intensity = Math.min(1, profit / maxProfit);
        const red = 100 - (intensity * 100);
        const green = 200;
        const blue = 100 - (intensity * 70);
        return `rgba(${red}, ${green}, ${blue}, 0.8)`;
    }
} 