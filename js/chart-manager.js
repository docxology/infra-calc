/**
 * Chart manager module to handle Chart.js instances consistently
 */

// Store chart instances
const chartInstances = {};

/**
 * Create or update a Chart.js instance
 * @param {string} canvasId - Canvas element ID
 * @param {object} config - Chart.js configuration
 * @returns {Chart} The Chart.js instance
 */
export function createChart(canvasId, config) {
    const canvas = document.getElementById(canvasId);
    
    if (!canvas) {
        console.error(`Canvas element with ID ${canvasId} not found`);
        return null;
    }
    
    // Get the rendering context
    const ctx = canvas.getContext('2d');
    
    if (!ctx) {
        console.error(`Unable to get 2D context for canvas ${canvasId}`);
        return null;
    }
    
    // Destroy existing chart if it exists
    if (chartInstances[canvasId]) {
        chartInstances[canvasId].destroy();
        delete chartInstances[canvasId];
    }
    
    try {
        // Create new chart
        const chart = new Chart(ctx, config);
        
        // Store the instance
        chartInstances[canvasId] = chart;
        
        return chart;
    } catch (error) {
        console.error(`Error creating chart for ${canvasId}:`, error);
        return null;
    }
}

/**
 * Get an existing chart instance
 * @param {string} canvasId - Canvas element ID
 * @returns {Chart|null} The Chart.js instance or null if not found
 */
export function getChart(canvasId) {
    return chartInstances[canvasId] || null;
}

/**
 * Destroy a chart instance
 * @param {string} canvasId - Canvas element ID
 * @returns {boolean} Success status
 */
export function destroyChart(canvasId) {
    if (chartInstances[canvasId]) {
        chartInstances[canvasId].destroy();
        delete chartInstances[canvasId];
        return true;
    }
    return false;
}

/**
 * Destroy all chart instances
 */
export function destroyAllCharts() {
    Object.keys(chartInstances).forEach(canvasId => {
        chartInstances[canvasId].destroy();
        delete chartInstances[canvasId];
    });
}

/**
 * Apply a specific sizing strategy to charts
 * This helps ensure charts render properly in their containers
 */
export function resizeCharts() {
    // Force a resize on all charts
    Object.keys(chartInstances).forEach(canvasId => {
        if (chartInstances[canvasId]) {
            chartInstances[canvasId].resize();
        }
    });
}

/**
 * Check if Chart.js is properly loaded and available
 * @returns {boolean} True if Chart.js is available
 */
export function isChartJsAvailable() {
    return typeof Chart !== 'undefined';
} 