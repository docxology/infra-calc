/**
 * Debug and diagnostic utilities for Infra-Calc
 */

/**
 * Check if calculations are working properly
 * @param {Object} params - Parameters for a test calculation
 * @returns {boolean} True if calculations appear to be working
 */
export function diagnosticCheck(params) {
    console.log("Running diagnostic check with params:", params);
    
    try {
        // Check that DOM is ready
        const calculateBtn = document.getElementById('calculate-btn');
        console.log("DOM readiness check:", calculateBtn ? "PASS" : "FAIL");
        
        // Check Chart.js is loaded
        console.log("Chart.js availability check:", window.Chart ? "PASS" : "FAIL");
        
        // Check canvas contexts
        const canvasIds = ['profit-chart', 'cost-breakdown-chart', 'price-frontier-chart', 'profitability-heatmap'];
        canvasIds.forEach(id => {
            const canvas = document.getElementById(id);
            const context = canvas ? canvas.getContext('2d') : null;
            console.log(`Canvas ${id} context check:`, context ? "PASS" : "FAIL");
        });
        
        // Import required modules
        import('./calculator.js')
            .then(calculatorModule => {
                console.log("Calculator module import check: PASS");
                
                // Try a sample calculation
                try {
                    const results = calculatorModule.calculateAll(params);
                    console.log("Calculation check:", results ? "PASS" : "FAIL");
                    console.log("Sample calculation results:", {
                        monthlyRevenue: results.monthlyRevenue,
                        monthlyCosts: results.monthlyCosts.totalMonthlyCost,
                        monthlyProfit: results.profitability.monthlyProfit,
                        breakEvenMonths: results.profitability.breakEvenTimeMonths
                    });
                    return true;
                } catch (calcError) {
                    console.error("Calculation error:", calcError);
                    return false;
                }
            })
            .catch(err => {
                console.error("Module import error:", err);
                return false;
            });
    } catch (e) {
        console.error("Diagnostic error:", e);
        return false;
    }
} 