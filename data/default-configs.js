/**
 * Default hardware configurations for different LLM sizes
 */
export const defaultConfigs = {
    small: {
        name: "Small LLM Setup (7-13B)",
        description: "Suitable for running smaller models like Llama2 7B, Mistral 7B, etc.",
        hardwareCosts: {
            gpu: 1000, // Consumer GPU like RTX 3090
            cpu: 300,  // Mid-range CPU
            ram: 200,  // 32GB RAM
            other: 500 // Motherboard, PSU, cooling, case, etc.
        },
        performance: {
            vramRequired: 16,      // GB
            powerConsumption: 450, // Watts
            maxThreads: 1,         // Concurrent inference threads
            inferenceTime: 0.5     // Minutes per average request
        }
    },
    medium: {
        name: "Medium LLM Setup (30-40B)",
        description: "Suitable for running medium-sized models like Llama2 34B, CodeLlama 34B, etc.",
        hardwareCosts: {
            gpu: 2000, // Higher-end consumer GPU like RTX 4090
            cpu: 400,  // Higher-end consumer CPU
            ram: 250,  // 64GB RAM
            other: 600 // Motherboard, PSU, cooling, case, etc.
        },
        performance: {
            vramRequired: 24,      // GB
            powerConsumption: 600, // Watts
            maxThreads: 1,         // Concurrent inference threads
            inferenceTime: 0.75    // Minutes per average request
        }
    },
    large: {
        name: "Large LLM Setup (65-70B)",
        description: "Suitable for running large models like Llama2 70B, Falcon 40B, etc.",
        hardwareCosts: {
            gpu: 3000, // Pro GPU like RTX A5000 or multiple consumer GPUs
            cpu: 500,  // High-end CPU
            ram: 300,  // 128GB RAM
            other: 700 // Motherboard, PSU, cooling, case, etc.
        },
        performance: {
            vramRequired: 40,      // GB
            powerConsumption: 800, // Watts
            maxThreads: 1,         // Concurrent inference threads
            inferenceTime: 1.0     // Minutes per average request
        }
    },
    server: {
        name: "Server LLM Setup (Multiple models)",
        description: "For running multiple models or serving many users concurrently",
        hardwareCosts: {
            gpu: 10000, // Multiple professional GPUs or server GPUs
            cpu: 1200,  // Server-grade CPU(s)
            ram: 1000,  // 256GB+ RAM
            other: 2000 // Server chassis, enterprise cooling, redundant PSUs, etc.
        },
        performance: {
            vramRequired: 80,       // GB total
            powerConsumption: 1500, // Watts
            maxThreads: 4,          // Concurrent inference threads
            inferenceTime: 0.8      // Minutes per average request
        }
    }
};

/**
 * Electricity costs by region ($/kWh)
 */
export const electricityRates = {
    us: {
        average: 0.15,
        low: 0.10,    // States like Idaho, Washington
        high: 0.30,   // States like Hawaii, California
    },
    europe: {
        average: 0.25,
        low: 0.15,    // Countries like Bulgaria
        high: 0.40,   // Countries like Germany, Denmark
    },
    asia: {
        average: 0.12,
        low: 0.05,    // Countries like India
        high: 0.30,   // Countries like Japan
    }
};

/**
 * LLM parameter to VRAM requirements mapping (approximate)
 */
export const llmVramRequirements = {
    "7B": 14,
    "13B": 28,
    "30B": 60,
    "65B": 130,
    "70B": 140
}; 