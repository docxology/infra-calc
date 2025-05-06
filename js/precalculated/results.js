
// Pre-calculated results from Python
const preCalculatedResults = {
  "monthly_revenue": 1728.0,
  "monthly_costs": {
    "hardware_depreciation": 2819.4444444444443,
    "electricity_cost_monthly": 64.8,
    "maintenance_cost": 50,
    "total_monthly_cost": 2934.2444444444445,
    "total_hardware_cost": 101500,
    "gpu_cost": 100000,
    "power_consumption": 1200
  },
  "profitability": {
    "monthly_profit": -1206.2444444444445,
    "profit_margin": -69.80581275720165,
    "break_even_time_months": Infinity,
    "roi": -42.78305418719212
  },
  "price_frontier": {
    "utilization_rates": [
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24
    ],
    "min_prices": [
      0.39928395061728394,
      0.200016975308642,
      0.1335946502057613,
      0.10038348765432098,
      0.08045679012345679,
      0.06717232510288065,
      0.05768342151675485,
      0.05056674382716049,
      0.045031550068587105,
      0.040603395061728394,
      0.03698035914702581,
      0.03396116255144033,
      0.03140645773979107,
      0.029216710758377423,
      0.027318930041152264,
      0.025658371913580245,
      0.024193173565722585,
      0.02289077503429355,
      0.021725471085120206,
      0.020676697530864197,
      0.019727807172251616,
      0.018865179573512907,
      0.01807756307031669,
      0.017355581275720164
    ],
    "break_even_times": [
      Infinity,
      Infinity,
      Infinity,
      Infinity,
      Infinity,
      Infinity,
      Infinity,
      Infinity,
      Infinity,
      Infinity,
      Infinity,
      Infinity,
      Infinity,
      Infinity,
      Infinity,
      Infinity,
      Infinity,
      Infinity,
      Infinity,
      Infinity,
      Infinity,
      Infinity,
      Infinity,
      Infinity
    ]
  },
  "profit_over_time": {
    "months": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      39,
      40,
      41,
      42,
      43,
      44,
      45,
      46,
      47,
      48,
      49,
      50,
      51,
      52,
      53,
      54,
      55,
      56,
      57,
      58,
      59,
      60
    ],
    "cumulative_profit": [
      -101500,
      -101500,
      -102706.24444444444,
      -103912.48888888888,
      -105118.73333333332,
      -106324.97777777776,
      -107531.2222222222,
      -108737.46666666665,
      -109943.71111111109,
      -111149.95555555553,
      -112356.19999999997,
      -113562.44444444441,
      -114768.68888888885,
      -115974.93333333329,
      -117181.17777777773,
      -118387.42222222217,
      -119593.66666666661,
      -120799.91111111105,
      -122006.1555555555,
      -123212.39999999994,
      -124418.64444444438,
      -125624.88888888882,
      -126831.13333333326,
      -128037.3777777777,
      -129243.62222222214,
      -130449.86666666658,
      -131656.11111111104,
      -132862.3555555555,
      -134068.59999999995,
      -135274.8444444444,
      -136481.08888888886,
      -137687.3333333333,
      -138893.57777777777,
      -140099.82222222222,
      -141306.06666666668,
      -142512.31111111114,
      -143718.5555555556,
      -246424.80000000005,
      -247631.0444444445,
      -248837.28888888896,
      -250043.5333333334,
      -251249.77777777787,
      -252456.02222222232,
      -253662.26666666678,
      -254868.51111111123,
      -256074.7555555557,
      -257281.00000000015,
      -258487.2444444446,
      -259693.48888888906,
      -260899.7333333335,
      -262105.97777777797,
      -263312.2222222224,
      -264518.46666666685,
      -265724.7111111113,
      -266930.9555555557,
      -268137.2000000001,
      -269343.44444444455,
      -270549.688888889,
      -271755.9333333334,
      -272962.17777777783,
      -274168.42222222226
    ],
    "cumulative_cost": [
      101500,
      101500,
      104434.24444444444,
      107368.48888888888,
      110302.73333333332,
      113236.97777777776,
      116171.2222222222,
      119105.46666666665,
      122039.71111111109,
      124973.95555555553,
      127908.19999999997,
      130842.44444444441,
      133776.68888888886,
      136710.93333333332,
      139645.17777777778,
      142579.42222222223,
      145513.6666666667,
      148447.91111111114,
      151382.1555555556,
      154316.40000000005,
      157250.6444444445,
      160184.88888888896,
      163119.13333333342,
      166053.37777777787,
      168987.62222222233,
      171921.86666666679,
      174856.11111111124,
      177790.3555555557,
      180724.60000000015,
      183658.8444444446,
      186593.08888888906,
      189527.33333333352,
      192461.57777777797,
      195395.82222222243,
      198330.06666666688,
      201264.31111111134,
      204198.5555555558,
      308632.8000000003,
      311567.0444444447,
      314501.28888888913,
      317435.53333333356,
      320369.777777778,
      323304.0222222224,
      326238.26666666684,
      329172.51111111126,
      332106.7555555557,
      335041.0000000001,
      337975.24444444454,
      340909.48888888897,
      343843.7333333334,
      346777.9777777778,
      349712.22222222225,
      352646.4666666667,
      355580.7111111111,
      358514.9555555555,
      361449.19999999995,
      364383.4444444444,
      367317.6888888888,
      370251.93333333323,
      373186.17777777766,
      376120.4222222221
    ],
    "cumulative_revenue": [
      0,
      0,
      1728.0,
      3456.0,
      5184.0,
      6912.0,
      8640.0,
      10368.0,
      12096.0,
      13824.0,
      15552.0,
      17280.0,
      19008.0,
      20736.0,
      22464.0,
      24192.0,
      25920.0,
      27648.0,
      29376.0,
      31104.0,
      32832.0,
      34560.0,
      36288.0,
      38016.0,
      39744.0,
      41472.0,
      43200.0,
      44928.0,
      46656.0,
      48384.0,
      50112.0,
      51840.0,
      53568.0,
      55296.0,
      57024.0,
      58752.0,
      60480.0,
      62208.0,
      63936.0,
      65664.0,
      67392.0,
      69120.0,
      70848.0,
      72576.0,
      74304.0,
      76032.0,
      77760.0,
      79488.0,
      81216.0,
      82944.0,
      84672.0,
      86400.0,
      88128.0,
      89856.0,
      91584.0,
      93312.0,
      95040.0,
      96768.0,
      98496.0,
      100224.0,
      101952.0
    ]
  },
  "gpu_thread_data": {
    "llm_sizes": [
      7,
      13,
      20,
      30,
      65,
      70,
      105,
      175
    ],
    "vram_requirements": {
      "7": 14,
      "13": 26,
      "20": 40,
      "30": 60,
      "65": 130,
      "70": 140,
      "105": 210,
      "175": 350
    },
    "max_threads_by_model": {
      "NVIDIA RTX 5090": [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1
      ],
      "NVIDIA RTX 5080": [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1
      ],
      "NVIDIA H100": [
        5,
        3,
        2,
        1,
        1,
        1,
        1,
        1
      ],
      "NVIDIA H200 (SXM)": [
        10,
        5,
        3,
        2,
        1,
        1,
        1,
        1
      ],
      "NVIDIA GB200 (Blackwell)": [
        13,
        7,
        4,
        3,
        1,
        1,
        1,
        1
      ]
    }
  },
  "selected_gpu_model": {
    "name": "NVIDIA GB200 (Blackwell)",
    "cost": 100000,
    "memory_gb": 192,
    "power_w": 1200
  },
  "constrained_threads": 4
};

// Function to update UI with pre-calculated results
function loadPreCalculatedResults() {
    // Add animation classes for smooth transitions
    document.querySelectorAll('.result-item').forEach(item => {
        item.classList.add('animate-in');
    });
    
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
    document.getElementById('charge-per-minute').value = 0.02;
    document.getElementById('utilization-rate').value = 12;
    document.getElementById('concurrent-threads').value = 4;
    document.getElementById('llm-size').value = 70;
    document.getElementById('inference-time').value = 1.0;
    document.getElementById('vram-required').value = 40;
    document.getElementById('gpu-cost').value = 3000;
    document.getElementById('cpu-cost').value = 500;
    document.getElementById('ram-cost').value = 300;
    document.getElementById('other-hardware').value = 700;
    document.getElementById('hardware-lifespan').value = 3;
    document.getElementById('electricity-cost').value = 0.15;
    document.getElementById('power-consumption').value = 800;
    document.getElementById('maintenance-cost').value = 50;
    
    // Update ROI calculator UI elements with animated transitions
    document.getElementById('roi-utilization-value').textContent = 12;
    document.getElementById('roi-utilization').value = 12;
    document.getElementById('roi-price-value').textContent = 0.02;
    document.getElementById('roi-price').value = 0.02;
    document.getElementById('roi-threads-value').textContent = 4;
    document.getElementById('roi-threads').value = 4;
    
    const roi = preCalculatedResults.profitability.roi;
    document.getElementById('roi-value').textContent = roi.toFixed(1);
    document.getElementById('roi-breakeven').textContent = 'Never';
    
    // Set ROI meter fill with animation
    const roiMeterFill = document.getElementById('roi-meter-fill');
    const roiPercentage = Math.min(Math.max(roi / 200, 0), 1) * 100;
    roiMeterFill.style.backgroundColor = roi > 0 ? '#4CAF50' : '#F44336';
    
    // Animate the ROI meter fill
    setTimeout(() => {
        roiMeterFill.style.width = roiPercentage + '%';
    }, 300);
    
    // Display GPU model and VRAM constraints info
    if (preCalculatedResults.selected_gpu_model) {
        const gpuModel = preCalculatedResults.selected_gpu_model;
        
        // Create or update GPU info section
        let gpuInfoDiv = document.getElementById('gpu-info-section');
        if (!gpuInfoDiv) {
            gpuInfoDiv = document.createElement('div');
            gpuInfoDiv.id = 'gpu-info-section';
            gpuInfoDiv.className = 'result-card';
            
            const resultsPanel = document.querySelector('.results-panel');
            if (resultsPanel) {
                resultsPanel.insertBefore(gpuInfoDiv, resultsPanel.firstChild);
            }
        }
        
        // Update GPU info content
        gpuInfoDiv.innerHTML = `
            <h3>GPU Model Information</h3>
            <div class="gpu-info-container">
                <div class="gpu-specs">
                    <div class="gpu-model-name">${gpuModel.name}</div>
                    <div class="gpu-specs-details">
                        <div class="spec-item">
                            <span class="spec-label">Memory:</span>
                            <span class="spec-value">${gpuModel.memory_gb} GB</span>
                        </div>
                        <div class="spec-item">
                            <span class="spec-label">Cost:</span>
                            <span class="spec-value">$${gpuModel.cost.toLocaleString()}</span>
                        </div>
                        <div class="spec-item">
                            <span class="spec-label">Power:</span>
                            <span class="spec-value">${gpuModel.power_w} W</span>
                        </div>
                    </div>
                </div>
                <div class="gpu-constraint-info">
                    <div class="constraint-item">
                        <span class="constraint-label">Model VRAM Requirement:</span>
                        <span class="constraint-value">${self.params.vram_required} GB</span>
                    </div>
                    <div class="constraint-item">
                        <span class="constraint-label">Desired Threads:</span>
                        <span class="constraint-value">${self.params.concurrent_threads}</span>
                    </div>
                    <div class="constraint-item">
                        <span class="constraint-label">Hardware Constrained Threads:</span>
                        <span class="constraint-value">${preCalculatedResults.constrained_threads}</span>
                    </div>
                </div>
            </div>
        `;
    }
}

// Animation function for counting up numbers
function animateCounter(elementId, start, end) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    const duration = 1500; // milliseconds
    const startTime = performance.now();
    const isPercentage = elementId.includes('roi') || elementId.includes('margin');
    const prefix = elementId.includes('revenue') || elementId.includes('cost') || elementId.includes('profit') ? '$' : '';
    const suffix = isPercentage ? '%' : '';
    const decimals = isPercentage || elementId.includes('roi') ? 1 : 2;
    
    function updateNumber(timestamp) {
        const elapsed = timestamp - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function for smoother animation
        const eased = progress < 0.5 ? 4 * progress * progress * progress : 
                      1 - Math.pow(-2 * progress + 2, 3) / 2;
                      
        const currentValue = start + (end - start) * eased;
        element.textContent = `${prefix}${currentValue.toFixed(decimals)}${suffix}`;
        
        if (progress < 1) {
            requestAnimationFrame(updateNumber);
        }
    }
    
    requestAnimationFrame(updateNumber);
}

// Replace chart images with pre-generated PNGs and handle responsive behavior
function loadChartImages() {
    // Get chart containers
    const chartContainers = document.querySelectorAll('.chart-card');
    
    // Map of canvas IDs to image paths
    const chartImageMap = {
        'profit-chart': 'visualizations/profit_over_time.png',
        'cost-breakdown-chart': 'visualizations/cost_breakdown.png',
        'price-frontier-chart': 'visualizations/price_frontier.png',
        'profitability-heatmap': 'visualizations/heatmap_price_vs_utilization.png'
    };
    
    // Add new GPU-related chart images to the map
    const gpuChartImageMap = {
        'gpu-memory-threads': 'visualizations/gpu_memory_threads.png',
        'llm-size-threads': 'visualizations/llm_size_threads.png',
        'gpu-cost-efficiency': 'visualizations/gpu_cost_efficiency.png'
    };
    
    // Merge all charts
    Object.assign(chartImageMap, gpuChartImageMap);
    
    // Replace each canvas with its corresponding image
    chartContainers.forEach(container => {
        const canvas = container.querySelector('canvas');
        if (canvas && chartImageMap[canvas.id]) {
            replaceCanvasWithImage(canvas, chartImageMap[canvas.id]);
            
            // Add fade-in animation class
            container.classList.add('fade-in');
        }
    });
    
    // Add GPU visualization section
    const resultsPanel = document.querySelector('.results-panel');
    if (resultsPanel) {
        // Create GPU visualizations section header
        const gpuHeader = document.createElement('h2');
        gpuHeader.className = 'section-title fade-in';
        gpuHeader.textContent = 'GPU Model Analysis';
        resultsPanel.appendChild(gpuHeader);
        
        // Add GPU visualization cards
        const gpuVisKeys = Object.keys(gpuChartImageMap);
        const gpuVisTitles = {
            'gpu-memory-threads': 'GPU Memory & Maximum Threads',
            'llm-size-threads': 'LLM Size vs Available Threads',
            'gpu-cost-efficiency': 'GPU Cost Efficiency'
        };
        
        const gpuVisDescriptions = {
            'gpu-memory-threads': 'Comparison of GPU memory capacity and the maximum concurrent threads possible for the current model size.',
            'llm-size-threads': 'How different LLM sizes affect the number of concurrent threads each GPU can handle.',
            'gpu-cost-efficiency': 'Cost per thread analysis showing which GPUs provide the best value for running this workload.'
        };
        
        gpuVisKeys.forEach(chartId => {
            const card = document.createElement('div');
            card.className = 'result-card chart-card fade-in';
            card.innerHTML = `
                <h3>${gpuVisTitles[chartId]}</h3>
                <div class="chart-container">
                    <img src="${gpuChartImageMap[chartId]}" class="pre-generated-chart" alt="${gpuVisTitles[chartId]}">
                </div>
                <p class="chart-description">${gpuVisDescriptions[chartId]}</p>
            `;
            
            resultsPanel.appendChild(card);
        });
    }
    
    // Set up heatmap controls with enhanced behavior
    const generateHeatmapBtn = document.getElementById('generate-heatmap');
    const xAxisSelect = document.getElementById('heatmap-x-axis');
    const yAxisSelect = document.getElementById('heatmap-y-axis');
    
    if (generateHeatmapBtn && xAxisSelect && yAxisSelect) {
        generateHeatmapBtn.addEventListener('click', function() {
            const xAxis = xAxisSelect.value;
            const yAxis = yAxisSelect.value;
            
            // Prevent same axis selection
            if (xAxis === yAxis) {
                alert('Please select different parameters for X and Y axes');
                return;
            }
            
            // Get the heatmap canvas
            const heatmapCanvas = document.getElementById('profitability-heatmap');
            if (!heatmapCanvas) return;
            
            // Add loading indicator
            const loadingIndicator = document.getElementById('heatmap-loading-indicator');
            if (loadingIndicator) loadingIndicator.style.display = 'block';
            
            // Add fade-out effect to current image
            const currentImg = heatmapCanvas.parentNode.querySelector('img');
            if (currentImg) {
                currentImg.style.opacity = '0.3';
            }
            
            // Load the appropriate pre-generated heatmap with a small delay for visual effect
            setTimeout(() => {
                replaceCanvasWithImage(heatmapCanvas, `visualizations/heatmap_${yAxis}_vs_${xAxis}.png`);
                
                // Hide loading indicator
                if (loadingIndicator) loadingIndicator.style.display = 'none';
                
                // Apply fade-in effect to new image
                const newImg = heatmapCanvas.parentNode.querySelector('img');
                if (newImg) {
                    newImg.style.opacity = '0.3';
                    setTimeout(() => { newImg.style.opacity = '1'; }, 50);
                }
            }, 500);
        });
    }
}

// Helper function to replace a canvas with an image
function replaceCanvasWithImage(canvas, imagePath) {
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
    img.onerror = function() {
        console.warn(`Image ${imagePath} not found`);
        // Keep the canvas if image isn't found
    };
    
    // Apply fade-in effect
    img.style.opacity = '0';
    
    const parent = canvas.parentNode;
    parent.replaceChild(img, canvas);
    
    // Trigger fade-in
    setTimeout(() => { img.style.opacity = '1'; }, 50);
}

// Helper function to check if a file exists (for animations)
function fileExists(url) {
    var http = new XMLHttpRequest();
    http.open('HEAD', url, false);
    try {
        http.send();
        return http.status !== 404;
    } catch(e) {
        return false;
    }
}

// Add CSS for animations and transitions
function addAnimationStyles() {
    const style = document.createElement('style');
    style.textContent = `
        .animate-in {
            animation: fadeIn 0.8s ease-out forwards;
        }
        
        .fade-in {
            animation: fadeIn 0.8s ease-out forwards;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .pre-generated-chart {
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border-radius: 4px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .pre-generated-chart:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        
        #roi-meter-fill {
            transition: width 1.5s cubic-bezier(0.19, 1, 0.22, 1), 
                        background-color 1s ease;
        }
        
        .section-title {
            width: 100%;
            text-align: center;
            margin: 40px 0 20px;
            font-size: 24px;
            color: #333;
            position: relative;
        }
        
        .section-title::after {
            content: '';
            position: absolute;
            bottom: -10px;
            left: 50%;
            transform: translateX(-50%);
            width: 100px;
            height: 3px;
            background: linear-gradient(90deg, #3498db, #27ae60);
        }
        
        /* GPU info styling */
        .gpu-info-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 15px;
        }
        
        .gpu-specs {
            flex: 1;
            min-width: 250px;
            background: linear-gradient(135deg, #76b900 0%, #5a8e00 100%);
            border-radius: 8px;
            padding: 15px;
            color: white;
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        
        .gpu-model-name {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.3);
            padding-bottom: 5px;
        }
        
        .gpu-specs-details {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .spec-item {
            display: flex;
            justify-content: space-between;
        }
        
        .gpu-constraint-info {
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
        }
        
        .constraint-item {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }
        
        .constraint-item:last-child {
            border-bottom: none;
        }
    `;
    document.head.appendChild(style);
}

// Initialize everything when the page loads
document.addEventListener('DOMContentLoaded', function() {
    // Add animation styles
    addAnimationStyles();
    
    // Load results with slight delay for visual effect
    setTimeout(() => {
        loadPreCalculatedResults();
        loadChartImages();
    }, 300);
    
    // Disable the calculate button - we're using pre-calculated results
    const calculateBtn = document.getElementById('calculate-btn');
    if (calculateBtn) {
        calculateBtn.textContent = 'Results Pre-calculated';
        
        calculateBtn.addEventListener('click', function(e) {
            e.preventDefault();
            alert('This demo uses pre-calculated results. Modify run.py parameters and re-run to see different calculations.');
        });
    }
    
    // Make ROI slider controls work interactively
    const roiSliders = document.querySelectorAll('.roi-calculator input[type="range"]');
    roiSliders.forEach(slider => {
        slider.addEventListener('input', function() {
            // Update displayed value
            const valueSpan = document.getElementById(`${this.id}-value`);
            if (valueSpan) valueSpan.textContent = this.value;
            
            // Calculate new ROI (simplified - normally would recalculate properly)
            const utilizationValue = parseFloat(document.getElementById('roi-utilization').value);
            const priceValue = parseFloat(document.getElementById('roi-price').value);
            const threadsValue = parseFloat(document.getElementById('roi-threads').value);
            
            // For demo purposes, simulate ROI change based on sliders
            // In a real implementation, this would recalculate properly
            const baseRoi = preCalculatedResults.profitability.roi;
            const utilizationFactor = utilizationValue / 12;
            const priceFactor = priceValue / 0.02;
            const threadsFactor = threadsValue / 4;
            
            const estimatedRoi = baseRoi * utilizationFactor * priceFactor * threadsFactor;
            
            // Update ROI display
            document.getElementById('roi-value').textContent = estimatedRoi.toFixed(1);
            
            // Update meter
            const roiMeterFill = document.getElementById('roi-meter-fill');
            const roiPercentage = Math.min(Math.max(estimatedRoi / 200, 0), 1) * 100;
            roiMeterFill.style.width = roiPercentage + '%';
            roiMeterFill.style.backgroundColor = estimatedRoi > 0 ? '#4CAF50' : '#F44336';
        });
    });
});
