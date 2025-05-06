/**
 * ROI Calculator module for Infra-Calc
 */

import { calculateAll } from './calculator.js';
import { formatTime } from './charts.js';

/**
 * Initialize the ROI calculator with sliders and event listeners
 * @param {Object} baseParams - Base parameters for calculations
 */
export function initRoiCalculator(baseParams) {
    const utilizationSlider = document.getElementById('roi-utilization');
    const utilizationValueDisplay = document.getElementById('roi-utilization-value');
    
    const priceSlider = document.getElementById('roi-price');
    const priceValueDisplay = document.getElementById('roi-price-value');
    
    const threadsSlider = document.getElementById('roi-threads');
    const threadsValueDisplay = document.getElementById('roi-threads-value');
    
    const roiValueDisplay = document.getElementById('roi-value');
    const breakEvenDisplay = document.getElementById('roi-breakeven');
    const meterFill = document.getElementById('roi-meter-fill');
    
    if (!utilizationSlider || !priceSlider || !threadsSlider || 
        !utilizationValueDisplay || !priceValueDisplay || !threadsValueDisplay ||
        !roiValueDisplay || !breakEvenDisplay || !meterFill) {
        console.error('ROI Calculator: Some elements not found');
        return;
    }
    
    // Set initial values
    utilizationSlider.value = baseParams.dailyUtilization;
    priceSlider.value = baseParams.chargePerMinute;
    threadsSlider.value = baseParams.concurrentThreads;
    
    // Update displays
    utilizationValueDisplay.textContent = utilizationSlider.value;
    priceValueDisplay.textContent = priceSlider.value;
    threadsValueDisplay.textContent = threadsSlider.value;
    
    // Calculate and update ROI
    updateRoiDisplay(baseParams);
    
    // Add event listeners
    utilizationSlider.addEventListener('input', () => {
        utilizationValueDisplay.textContent = utilizationSlider.value;
        updateRoiDisplay(baseParams);
    });
    
    priceSlider.addEventListener('input', () => {
        priceValueDisplay.textContent = priceSlider.value;
        updateRoiDisplay(baseParams);
    });
    
    threadsSlider.addEventListener('input', () => {
        threadsValueDisplay.textContent = threadsSlider.value;
        updateRoiDisplay(baseParams);
    });
}

/**
 * Update the ROI display based on current slider values
 * @param {Object} baseParams - Base parameters for calculations
 */
export function updateRoiDisplay(baseParams) {
    const utilizationSlider = document.getElementById('roi-utilization');
    const priceSlider = document.getElementById('roi-price');
    const threadsSlider = document.getElementById('roi-threads');
    
    const roiValueDisplay = document.getElementById('roi-value');
    const breakEvenDisplay = document.getElementById('roi-breakeven');
    const meterFill = document.getElementById('roi-meter-fill');
    
    // Create params object from base params
    const params = { ...baseParams };
    
    // Update with slider values
    params.dailyUtilization = parseFloat(utilizationSlider.value);
    params.chargePerMinute = parseFloat(priceSlider.value);
    params.concurrentThreads = parseInt(threadsSlider.value);
    
    // Calculate results
    const results = calculateAll(params);
    
    // Get ROI and break-even values
    const roi = results.profitability.roi;
    const breakEvenMonths = results.profitability.breakEvenTimeMonths;
    
    // Update displays
    roiValueDisplay.textContent = roi.toFixed(2);
    breakEvenDisplay.textContent = formatTime(breakEvenMonths);
    
    // Update meter fill width based on ROI
    const meterWidth = calculateMeterWidth(roi);
    meterFill.style.width = `${meterWidth}%`;
    
    // Update meter fill color based on ROI
    updateMeterColor(meterFill, roi);
    
    // Add animation to highlight changes
    meterFill.classList.remove('animate');
    void meterFill.offsetWidth; // Trigger reflow to restart animation
    meterFill.classList.add('animate');
}

/**
 * Calculate meter width based on ROI value
 * @param {number} roi - ROI percentage
 * @returns {number} Width percentage (0-100)
 */
function calculateMeterWidth(roi) {
    // Define thresholds for meter
    const minRoi = -50;
    const maxRoi = 200;
    
    // Limit ROI to range for visualization
    const limitedRoi = Math.max(minRoi, Math.min(maxRoi, roi));
    
    // Map ROI to width percentage (0% = minRoi, 100% = maxRoi)
    return ((limitedRoi - minRoi) / (maxRoi - minRoi)) * 100;
}

/**
 * Update meter color based on ROI value
 * @param {HTMLElement} meterElement - Meter fill element
 * @param {number} roi - ROI percentage
 */
function updateMeterColor(meterElement, roi) {
    if (roi < 0) {
        // Red for negative ROI
        meterElement.style.backgroundColor = 'rgba(255, 100, 100, 0.7)';
    } else if (roi < 50) {
        // Yellow for low positive ROI
        meterElement.style.backgroundColor = 'rgba(255, 205, 86, 0.7)';
    } else if (roi < 100) {
        // Light green for good ROI
        meterElement.style.backgroundColor = 'rgba(100, 230, 132, 0.7)';
    } else {
        // Strong green for excellent ROI
        meterElement.style.backgroundColor = 'rgba(52, 211, 153, 0.7)';
    }
} 