/**
 * Chart visualization module for Infra-Calc
 */

import { createChart } from './chart-manager.js';

/**
 * Format currency values for display
 * @param {number} value - Value to format
 * @param {number} decimals - Number of decimal places
 * @returns {string} Formatted currency string
 */
export function formatCurrency(value, decimals = 2) {
    return '$' + value.toFixed(decimals);
}

/**
 * Format number values for display
 * @param {number} value - Value to format
 * @param {number} decimals - Number of decimal places
 * @returns {string} Formatted number string
 */
export function formatNumber(value, decimals = 2) {
    return value.toFixed(decimals);
}

/**
 * Format time values in months
 * @param {number} months - Number of months
 * @returns {string} Formatted time string
 */
export function formatTime(months) {
    if (months === Infinity || isNaN(months)) {
        return 'Never';
    }
    
    const years = Math.floor(months / 12);
    const remainingMonths = Math.round(months % 12);
    
    if (years === 0) {
        return `${remainingMonths} month${remainingMonths !== 1 ? 's' : ''}`;
    } else if (remainingMonths === 0) {
        return `${years} year${years !== 1 ? 's' : ''}`;
    } else {
        return `${years} year${years !== 1 ? 's' : ''} and ${remainingMonths} month${remainingMonths !== 1 ? 's' : ''}`;
    }
}

/**
 * Creates a profit over time chart
 * @param {string} canvasId - ID of the canvas element
 * @param {Object} data - Data for the chart
 */
export function createProfitOverTimeChart(canvasId, data) {
    // Prepare data
    const { months, cumulativeProfit, cumulativeCost, cumulativeRevenue } = data;
    
    // Create breakeven point annotation if it exists
    const breakEvenIndex = cumulativeProfit.findIndex(profit => profit >= 0);
    const annotations = {};
    
    if (breakEvenIndex > 0) {
        annotations.breakeven = {
            type: 'line',
            scaleID: 'x',
            value: months[breakEvenIndex],
            borderColor: 'rgba(0, 0, 0, 0.7)',
            borderWidth: 2,
            borderDash: [5, 5],
            label: {
                content: 'Break-even',
                enabled: true,
                position: 'top'
            }
        };
    }
    
    // Create chart configuration
    const config = {
        type: 'line',
        data: {
            labels: months.map(month => `Month ${month}`),
            datasets: [
                {
                    label: 'Cumulative Profit',
                    data: cumulativeProfit,
                    borderColor: '#34d399',
                    backgroundColor: 'rgba(52, 211, 153, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1
                },
                {
                    label: 'Cumulative Revenue',
                    data: cumulativeRevenue,
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.1
                },
                {
                    label: 'Cumulative Cost',
                    data: cumulativeCost,
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${formatCurrency(context.raw)}`;
                        }
                    }
                },
                annotation: {
                    annotations: annotations
                },
                legend: {
                    position: 'top'
                },
                title: {
                    display: false
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time'
                    },
                    ticks: {
                        callback: function(value, index) {
                            const month = months[index];
                            if (month % 12 === 0) {
                                return `Year ${month / 12}`;
                            } else if (index === 0 || index === months.length - 1 || index % 6 === 0) {
                                return `Month ${month}`;
                            }
                            return '';
                        }
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Amount ($)'
                    },
                    ticks: {
                        callback: function(value) {
                            return formatCurrency(value, 0);
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
 * Creates a cost breakdown pie chart
 * @param {string} canvasId - ID of the canvas element
 * @param {Object} data - Data for the chart
 */
export function createCostBreakdownChart(canvasId, data) {
    // Prepare data
    const { labels, values } = data;
    
    // Create chart configuration
    const config = {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: [
                    'rgba(59, 130, 246, 0.7)',  // Blue for Hardware
                    'rgba(245, 158, 11, 0.7)',  // Amber for Electricity
                    'rgba(139, 92, 246, 0.7)'   // Purple for Maintenance
                ],
                borderColor: [
                    'rgba(59, 130, 246, 1)',
                    'rgba(245, 158, 11, 1)',
                    'rgba(139, 92, 246, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = formatCurrency(context.raw);
                            const total = context.dataset.data.reduce((acc, cur) => acc + cur, 0);
                            const percentage = Math.round((context.raw / total) * 100);
                            return `${label}: ${value} (${percentage}%)`;
                        }
                    }
                },
                legend: {
                    position: 'top'
                }
            }
        }
    };
    
    // Use the chart manager to create the chart
    return createChart(canvasId, config);
}

/**
 * Creates a price frontier chart
 * @param {string} canvasId - ID of the canvas element
 * @param {Object} data - Data for the chart
 * @param {number} currentPrice - Current price per minute
 */
export function createPriceFrontierChart(canvasId, data, currentPrice) {
    // Prepare data
    const { utilizationRates, minPrices, breakEvenTimes } = data;
    
    // Create price line (horizontal line at current price)
    const currentPriceData = utilizationRates.map(() => currentPrice);
    
    // Create chart configuration
    const config = {
        type: 'line',
        data: {
            labels: utilizationRates.map(hours => `${hours}h`),
            datasets: [
                {
                    label: 'Minimum Price per Minute',
                    data: minPrices,
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1
                },
                {
                    label: 'Current Price',
                    data: currentPriceData,
                    borderColor: '#ef4444',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            if (context.datasetIndex === 0) {
                                const price = formatCurrency(context.raw);
                                const breakEvenTime = formatTime(breakEvenTimes[context.dataIndex]);
                                return [
                                    `Minimum price: ${price}/minute`,
                                    `Break-even time: ${breakEvenTime}`
                                ];
                            } else {
                                return `Current price: ${formatCurrency(context.raw)}/minute`;
                            }
                        }
                    }
                },
                legend: {
                    position: 'top'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Daily Utilization (hours)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Price per Minute ($)'
                    },
                    ticks: {
                        callback: function(value) {
                            return formatCurrency(value);
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
 * Updates all charts with new data
 * @param {Object} results - Calculation results
 */
export function updateAllCharts(results) {
    // Update profit over time chart
    createProfitOverTimeChart('profit-chart', results.profitOverTime);
    
    // Update cost breakdown chart
    createCostBreakdownChart('cost-breakdown-chart', results.costBreakdown);
    
    // Update price frontier chart
    createPriceFrontierChart('price-frontier-chart', results.priceFrontier, results.params.chargePerMinute);
}

/**
 * Updates the results display with calculation results
 * @param {Object} results - Calculation results
 */
export function updateResults(results) {
    // Update revenue and cost numbers
    document.getElementById('monthly-revenue').textContent = formatCurrency(results.monthlyRevenue);
    document.getElementById('monthly-costs').textContent = formatCurrency(results.monthlyCosts.totalMonthlyCost);
    document.getElementById('monthly-profit').textContent = formatCurrency(results.profitability.monthlyProfit);
    document.getElementById('profit-margin').textContent = formatNumber(results.profitability.profitMargin) + '%';
    
    // Update break-even and ROI numbers
    document.getElementById('breakeven-time').textContent = formatTime(results.profitability.breakEvenTimeMonths);
    document.getElementById('lifetime-roi').textContent = formatNumber(results.profitability.roi) + '%';
    
    // Update all charts
    updateAllCharts(results);
    
    // Highlight profit/loss with colors
    const profitElement = document.getElementById('monthly-profit');
    if (results.profitability.monthlyProfit >= 0) {
        profitElement.style.color = '#34d399'; // Green for profit
    } else {
        profitElement.style.color = '#ef4444'; // Red for loss
    }
} 