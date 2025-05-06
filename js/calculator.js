/**
 * Core calculation module for LLM infrastructure economics
 */

/**
 * Calculate monthly electricity cost
 * @param {number} powerConsumption - Power consumption in watts
 * @param {number} hoursPerDay - Daily utilization in hours
 * @param {number} costPerKwh - Electricity cost per kWh
 * @returns {number} Monthly electricity cost in dollars
 */
export function calculateElectricityCost(powerConsumption, hoursPerDay, costPerKwh) {
    // Convert watts to kilowatts
    const powerInKw = powerConsumption / 1000;
    
    // Calculate daily energy usage in kWh
    const dailyEnergyKwh = powerInKw * hoursPerDay;
    
    // Calculate monthly energy cost (assuming 30 days)
    const monthlyCost = dailyEnergyKwh * costPerKwh * 30;
    
    return monthlyCost;
}

/**
 * Calculate monthly hardware depreciation cost
 * @param {number} hardwareCost - Total hardware cost
 * @param {number} lifespanYears - Expected hardware lifespan in years
 * @returns {number} Monthly hardware depreciation cost
 */
export function calculateHardwareDepreciation(hardwareCost, lifespanYears) {
    // Convert years to months and calculate monthly depreciation
    const lifespanMonths = lifespanYears * 12;
    return hardwareCost / lifespanMonths;
}

/**
 * Calculate monthly revenue
 * @param {number} chargePerMinute - Charge per minute of LLM usage
 * @param {number} hoursPerDay - Daily utilization in hours
 * @param {number} concurrentThreads - Number of concurrent threads
 * @returns {number} Monthly revenue in dollars
 */
export function calculateMonthlyRevenue(chargePerMinute, hoursPerDay, concurrentThreads) {
    // Convert hours to minutes
    const minutesPerDay = hoursPerDay * 60;
    
    // Calculate daily revenue
    const dailyRevenue = minutesPerDay * chargePerMinute * concurrentThreads;
    
    // Calculate monthly revenue (assuming 30 days)
    const monthlyRevenue = dailyRevenue * 30;
    
    return monthlyRevenue;
}

/**
 * Calculate total monthly costs
 * @param {Object} params - Cost parameters
 * @returns {Object} Breakdown of monthly costs
 */
export function calculateMonthlyCosts(params) {
    const {
        gpuCost,
        cpuCost,
        ramCost,
        otherHardwareCost,
        hardwareLifespan,
        powerConsumption,
        dailyUtilization,
        electricityCost,
        maintenanceCost
    } = params;
    
    // Calculate total hardware cost
    const totalHardwareCost = gpuCost + cpuCost + ramCost + otherHardwareCost;
    
    // Calculate monthly hardware depreciation
    const hardwareDepreciation = calculateHardwareDepreciation(totalHardwareCost, hardwareLifespan);
    
    // Calculate monthly electricity cost
    const electricityCostMonthly = calculateElectricityCost(powerConsumption, dailyUtilization, electricityCost);
    
    // Total monthly costs
    const totalMonthlyCost = hardwareDepreciation + electricityCostMonthly + maintenanceCost;
    
    return {
        hardwareDepreciation,
        electricityCostMonthly,
        maintenanceCost,
        totalMonthlyCost
    };
}

/**
 * Calculate profitability metrics
 * @param {number} monthlyRevenue - Monthly revenue
 * @param {number} monthlyCost - Monthly costs
 * @param {number} hardwareCost - Total hardware cost
 * @param {number} hardwareLifespan - Hardware lifespan in years
 * @returns {Object} Profitability metrics
 */
export function calculateProfitability(monthlyRevenue, monthlyCost, hardwareCost, hardwareLifespan) {
    // Calculate monthly profit
    const monthlyProfit = monthlyRevenue - monthlyCost;
    
    // Calculate profit margin
    const profitMargin = monthlyRevenue > 0 ? (monthlyProfit / monthlyRevenue) * 100 : 0;
    
    // Calculate break-even time in months
    const breakEvenTimeMonths = monthlyProfit > 0 ? hardwareCost / monthlyProfit : Infinity;
    
    // Calculate ROI over hardware lifespan
    const lifespanMonths = hardwareLifespan * 12;
    const lifetimeProfit = monthlyProfit * lifespanMonths;
    const roi = (lifetimeProfit / hardwareCost) * 100;
    
    return {
        monthlyProfit,
        profitMargin,
        breakEvenTimeMonths,
        roi
    };
}

/**
 * Calculate minimum charge per minute needed for profitability
 * @param {Object} params - Parameters for calculation
 * @returns {number} Minimum charge per minute for profitability
 */
export function calculateMinimumCharge(params) {
    const {
        costs,
        concurrentThreads,
        hoursPerDay
    } = params;
    
    // Calculate total minutes of service per month
    const minutesPerMonth = hoursPerDay * 60 * 30 * concurrentThreads;
    
    // Calculate minimum charge per minute
    return costs.totalMonthlyCost / minutesPerMonth;
}

/**
 * Generate price frontier data for different utilization rates
 * @param {Object} params - Parameters for calculation
 * @returns {Array} Array of data points for price frontier chart
 */
export function generatePriceFrontierData(params) {
    const { 
        costs,
        concurrentThreads,
        hardwareCost,
        hardwareLifespan
    } = params;
    
    const utilizationRates = [];
    const minPrices = [];
    const breakEvenTimes = [];
    
    // Calculate minimum price and break-even time for different utilization rates
    for (let hours = 1; hours <= 24; hours += 1) {
        // Adjusted costs for different utilization rates
        const adjustedElectricityCost = calculateElectricityCost(
            params.powerConsumption, 
            hours, 
            params.electricityCost
        );
        
        const adjustedCost = costs.hardwareDepreciation + adjustedElectricityCost + costs.maintenanceCost;
        
        // Minutes per month at this utilization
        const minutesPerMonth = hours * 60 * 30 * concurrentThreads;
        
        // Minimum price per minute
        const minPrice = adjustedCost / minutesPerMonth;
        
        // Monthly profit at this minimum price
        const monthlyProfit = (minPrice * minutesPerMonth) - adjustedCost;
        
        // Break-even time
        const breakEvenTime = monthlyProfit > 0 ? hardwareCost / monthlyProfit : Infinity;
        
        utilizationRates.push(hours);
        minPrices.push(minPrice);
        breakEvenTimes.push(breakEvenTime);
    }
    
    return {
        utilizationRates,
        minPrices,
        breakEvenTimes
    };
}

/**
 * Generate profit over time data
 * @param {Object} params - Parameters for calculation
 * @returns {Array} Array of data points for profit over time chart
 */
export function generateProfitOverTimeData(params) {
    const {
        monthlyRevenue,
        monthlyCosts,
        hardwareCost,
        hardwareLifespan
    } = params;
    
    const months = [];
    const cumulativeProfit = [];
    const cumulativeCost = [];
    const cumulativeRevenue = [];
    
    // Calculate initial investment
    let currentProfit = -hardwareCost;
    let currentCost = hardwareCost;
    let currentRevenue = 0;
    
    // Calculate for 3x hardware lifespan to show long-term trends
    const totalMonths = Math.min(hardwareLifespan * 36, 60); // Cap at 5 years (60 months)
    
    for (let month = 0; month <= totalMonths; month++) {
        months.push(month);
        cumulativeProfit.push(currentProfit);
        cumulativeCost.push(currentCost);
        cumulativeRevenue.push(currentRevenue);
        
        // Add monthly revenue and costs
        if (month > 0) {
            currentProfit += monthlyRevenue - monthlyCosts.totalMonthlyCost;
            currentCost += monthlyCosts.totalMonthlyCost;
            currentRevenue += monthlyRevenue;
        }
        
        // Handle hardware replacement at end of life
        if (month > 0 && month % (hardwareLifespan * 12) === 0 && month < totalMonths) {
            currentProfit -= hardwareCost;
            currentCost += hardwareCost;
        }
    }
    
    return {
        months,
        cumulativeProfit,
        cumulativeCost,
        cumulativeRevenue
    };
}

/**
 * Perform all calculations and return complete results
 * @param {Object} params - All input parameters
 * @returns {Object} Complete calculation results
 */
export function calculateAll(params) {
    // Extract parameters
    const {
        chargePerMinute,
        dailyUtilization,
        concurrentThreads,
        gpuCost,
        cpuCost,
        ramCost,
        otherHardwareCost,
        hardwareLifespan,
        electricityCost,
        powerConsumption,
        maintenanceCost
    } = params;
    
    // Calculate total hardware cost
    const hardwareCost = gpuCost + cpuCost + ramCost + otherHardwareCost;
    
    // Calculate monthly revenue
    const monthlyRevenue = calculateMonthlyRevenue(
        chargePerMinute,
        dailyUtilization,
        concurrentThreads
    );
    
    // Calculate monthly costs
    const monthlyCosts = calculateMonthlyCosts({
        gpuCost,
        cpuCost,
        ramCost,
        otherHardwareCost,
        hardwareLifespan,
        powerConsumption,
        dailyUtilization,
        electricityCost,
        maintenanceCost
    });
    
    // Calculate profitability metrics
    const profitability = calculateProfitability(
        monthlyRevenue,
        monthlyCosts.totalMonthlyCost,
        hardwareCost,
        hardwareLifespan
    );
    
    // Calculate minimum charge for profitability
    const minimumCharge = calculateMinimumCharge({
        costs: monthlyCosts,
        concurrentThreads,
        hoursPerDay: dailyUtilization
    });
    
    // Generate price frontier data
    const priceFrontier = generatePriceFrontierData({
        costs: monthlyCosts,
        concurrentThreads,
        powerConsumption,
        electricityCost,
        hardwareCost,
        hardwareLifespan
    });
    
    // Generate profit over time data
    const profitOverTime = generateProfitOverTimeData({
        monthlyRevenue,
        monthlyCosts,
        hardwareCost,
        hardwareLifespan
    });
    
    // Monthly cost breakdown for pie chart
    const costBreakdown = {
        labels: ['Hardware Depreciation', 'Electricity', 'Maintenance'],
        values: [
            monthlyCosts.hardwareDepreciation,
            monthlyCosts.electricityCostMonthly,
            monthlyCosts.maintenanceCost
        ]
    };
    
    return {
        hardwareCost,
        monthlyRevenue,
        monthlyCosts,
        profitability,
        minimumCharge,
        priceFrontier,
        profitOverTime,
        costBreakdown
    };
} 