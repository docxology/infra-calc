/**
 * Main JavaScript module for Infra-Calc
 * Manages UI interactions and ties together the calculator and chart modules
 */

import { calculateAll } from './calculator.js';
import { updateResults } from './charts.js';
import { defaultConfigs, electricityRates } from '../data/default-configs.js';
import { registerChartPlugins } from './chart-plugins.js';
import { generateHeatmapData, createHeatmap } from './heatmap.js';
import { initRoiCalculator, updateRoiDisplay } from './roi-calculator.js';
import { diagnosticCheck } from './debug.js';
import { resizeCharts, isChartJsAvailable } from './chart-manager.js';

// Default form values based on the question
const defaultValues = {
    chargePerMinute: 0.02,
    dailyUtilization: 12,
    concurrentThreads: 4,
    llmSize: 70,
    inferenceTime: 1,
    vramRequired: 40,
    gpuCost: 3000,
    cpuCost: 500,
    ramCost: 300,
    otherHardwareCost: 700,
    hardwareLifespan: 3,
    electricityCost: 0.15,
    powerConsumption: 800,
    maintenanceCost: 50
};

/**
 * Get all input values from the form
 * @returns {Object} Input values
 */
function getInputValues() {
    return {
        chargePerMinute: parseFloat(document.getElementById('charge-per-minute').value),
        dailyUtilization: parseFloat(document.getElementById('utilization-rate').value),
        concurrentThreads: parseInt(document.getElementById('concurrent-threads').value),
        llmSize: parseInt(document.getElementById('llm-size').value),
        inferenceTime: parseFloat(document.getElementById('inference-time').value),
        vramRequired: parseInt(document.getElementById('vram-required').value),
        gpuCost: parseFloat(document.getElementById('gpu-cost').value),
        cpuCost: parseFloat(document.getElementById('cpu-cost').value),
        ramCost: parseFloat(document.getElementById('ram-cost').value),
        otherHardwareCost: parseFloat(document.getElementById('other-hardware').value),
        hardwareLifespan: parseFloat(document.getElementById('hardware-lifespan').value),
        electricityCost: parseFloat(document.getElementById('electricity-cost').value),
        powerConsumption: parseFloat(document.getElementById('power-consumption').value),
        maintenanceCost: parseFloat(document.getElementById('maintenance-cost').value)
    };
}

/**
 * Set all form input values
 * @param {Object} values - Values to set
 */
function setInputValues(values) {
    document.getElementById('charge-per-minute').value = values.chargePerMinute;
    document.getElementById('utilization-rate').value = values.dailyUtilization;
    document.getElementById('concurrent-threads').value = values.concurrentThreads;
    document.getElementById('llm-size').value = values.llmSize;
    document.getElementById('inference-time').value = values.inferenceTime;
    document.getElementById('vram-required').value = values.vramRequired;
    document.getElementById('gpu-cost').value = values.gpuCost;
    document.getElementById('cpu-cost').value = values.cpuCost;
    document.getElementById('ram-cost').value = values.ramCost;
    document.getElementById('other-hardware').value = values.otherHardwareCost;
    document.getElementById('hardware-lifespan').value = values.hardwareLifespan;
    document.getElementById('electricity-cost').value = values.electricityCost;
    document.getElementById('power-consumption').value = values.powerConsumption;
    document.getElementById('maintenance-cost').value = values.maintenanceCost;
}

/**
 * Calculate and display results
 */
function calculateAndDisplayResults() {
    const inputValues = getInputValues();
    
    // Calculate results
    const results = calculateAll(inputValues);
    
    // Add original parameters to results for use in charts
    results.params = inputValues;
    
    // Update the results display and charts
    updateResults(results);
    
    // Update ROI calculator with new base params
    updateRoiDisplay(inputValues);
}

/**
 * Apply a hardware configuration preset
 * @param {string} configKey - Key of the configuration to use
 */
function applyHardwareConfig(configKey) {
    const config = defaultConfigs[configKey];
    
    if (!config) {
        console.error(`Configuration "${configKey}" not found`);
        return;
    }
    
    // Get current input values
    const currentValues = getInputValues();
    
    // Apply configuration to relevant inputs
    const updatedValues = {
        ...currentValues,
        gpuCost: config.hardwareCosts.gpu,
        cpuCost: config.hardwareCosts.cpu,
        ramCost: config.hardwareCosts.ram,
        otherHardwareCost: config.hardwareCosts.other,
        powerConsumption: config.performance.powerConsumption,
        vramRequired: config.performance.vramRequired,
        concurrentThreads: config.performance.maxThreads,
        inferenceTime: config.performance.inferenceTime
    };
    
    // Update the form
    setInputValues(updatedValues);
    
    // Calculate with new values
    calculateAndDisplayResults();
}

/**
 * Reset form to default values
 */
function resetToDefaults() {
    setInputValues(defaultValues);
    calculateAndDisplayResults();
}

/**
 * Generate and display the heatmap
 */
function generateAndDisplayHeatmap() {
    // Get current input values as base parameters
    const baseParams = getInputValues();
    
    // Get selected axes
    const xAxis = document.getElementById('heatmap-x-axis').value;
    const yAxis = document.getElementById('heatmap-y-axis').value;
    
    // Check if axes are different
    if (xAxis === yAxis) {
        alert('Please select different parameters for X and Y axes');
        return;
    }
    
    // Show loading indicator
    const loadingIndicator = document.getElementById('heatmap-loading-indicator');
    loadingIndicator.style.display = 'block';
    
    // Generate heatmap data with slight delay to allow UI update
    setTimeout(() => {
        try {
            const heatmapData = generateHeatmapData(baseParams, xAxis, yAxis);
            createHeatmap('profitability-heatmap', heatmapData);
            // Hide loading indicator
            loadingIndicator.style.display = 'none';
        } catch (error) {
            console.error('Error generating heatmap:', error);
            loadingIndicator.textContent = 'Error generating heatmap. Please try different parameters.';
            loadingIndicator.classList.remove('loading');
            loadingIndicator.classList.add('error');
        }
    }, 100);
}

/**
 * Initialize the application
 */
function init() {
    // Check if Chart.js is available
    if (!isChartJsAvailable()) {
        console.error('Chart.js is not loaded. Please check your script includes.');
        document.body.innerHTML = '<div style="text-align: center; margin-top: 100px; color: red; font-size: 24px;">Error: Chart.js library could not be loaded. Please check your internet connection.</div>';
        return;
    }
    
    // Register Chart.js plugins
    registerChartPlugins();
    
    // Run a diagnostic check
    diagnosticCheck(defaultValues);
    
    // Set initial input values
    setInputValues(defaultValues);
    
    // Add window resize listener to handle chart resizing
    window.addEventListener('resize', debounce(() => {
        resizeCharts();
    }, 250));
    
    // Give DOM elements time to render before calculating
    setTimeout(() => {
        // Calculate initial results
        calculateAndDisplayResults();
        
        // Initialize the ROI calculator
        initRoiCalculator(defaultValues);
        
        // Set up event listeners after initial calculation
        setupEventListeners();
        
        // Generate initial heatmap after a short delay
        setTimeout(() => {
            // Set default axes that make sense
            document.getElementById('heatmap-x-axis').value = 'utilization';
            document.getElementById('heatmap-y-axis').value = 'price';
            
            // Generate initial heatmap
            generateAndDisplayHeatmap();
        }, 500);
    }, 100);
}

/**
 * Set up all event listeners
 */
function setupEventListeners() {
    // Add event listeners
    document.getElementById('calculate-btn').addEventListener('click', calculateAndDisplayResults);
    document.getElementById('reset-btn').addEventListener('click', resetToDefaults);
    
    // Add heatmap generation event listener
    document.getElementById('generate-heatmap').addEventListener('click', generateAndDisplayHeatmap);
    
    // Add hardware preset button event listeners
    const presetButtons = document.querySelectorAll('.preset-btn');
    presetButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Remove active class from all preset buttons
            presetButtons.forEach(btn => btn.classList.remove('active'));
            
            // Add active class to clicked button
            this.classList.add('active');
            
            // Apply the hardware configuration
            const presetKey = this.dataset.preset;
            applyHardwareConfig(presetKey);
        });
    });
    
    // Add change event listeners to all input fields
    const inputs = document.querySelectorAll('input[type="number"]');
    inputs.forEach(input => {
        input.addEventListener('change', () => {
            // When an input changes, highlight the calculate button
            const calculateBtn = document.getElementById('calculate-btn');
            calculateBtn.classList.add('highlight');
            setTimeout(() => calculateBtn.classList.remove('highlight'), 2000);
            
            // Remove active class from preset buttons as values have been customized
            document.querySelectorAll('.preset-btn').forEach(btn => btn.classList.remove('active'));
        });
        
        // Also add input event for immediate feedback
        input.addEventListener('input', debounce(calculateAndDisplayResults, 500));
    });
}

/**
 * Debounce function to limit how often a function can be called
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @returns {Function} Debounced function
 */
function debounce(func, wait) {
    let timeout;
    return function() {
        const context = this;
        const args = arguments;
        clearTimeout(timeout);
        timeout = setTimeout(() => {
            func.apply(context, args);
        }, wait);
    };
}

// Initialize the application when DOM is loaded
if (document.readyState === 'loading') {
    // Add event listener if document is still loading
    document.addEventListener('dom-ready', init);
} else {
    // Otherwise, run init now
    init();
}

// Export functions for potential future use
export {
    calculateAndDisplayResults,
    resetToDefaults,
    applyHardwareConfig
}; 