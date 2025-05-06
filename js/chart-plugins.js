/**
 * Additional Chart.js plugin setup
 * 
 * This module implements a simple annotation plugin for Chart.js
 * based on the concept of the chartjs-plugin-annotation, but with 
 * only the minimal functionality we need for our break-even lines.
 */

/**
 * Simple annotation plugin implementation
 * This avoids having to include the full chartjs-plugin-annotation package
 */
const annotationPlugin = {
    id: 'simpleAnnotation',
    
    afterDraw: function(chart) {
        const options = chart.config.options;
        if (!options.plugins || !options.plugins.annotation || !options.plugins.annotation.annotations) {
            return;
        }
        
        const annotations = options.plugins.annotation.annotations;
        const ctx = chart.ctx;
        
        Object.keys(annotations).forEach(key => {
            const annotation = annotations[key];
            
            if (annotation.type === 'line') {
                drawLineAnnotation(chart, ctx, annotation);
            }
        });
    }
};

/**
 * Draw a line annotation
 */
function drawLineAnnotation(chart, ctx, annotation) {
    const scaleId = annotation.scaleID || 'x';
    const scale = chart.scales[scaleId];
    
    if (!scale) return;
    
    const value = annotation.value;
    const isVertical = scaleId === 'x';
    
    ctx.save();
    
    // Set up line style
    ctx.lineWidth = annotation.borderWidth || 1;
    ctx.strokeStyle = annotation.borderColor || 'rgba(0,0,0,0.5)';
    
    if (annotation.borderDash && annotation.borderDash.length) {
        ctx.setLineDash(annotation.borderDash);
    }
    
    // Draw the line
    ctx.beginPath();
    if (isVertical) {
        const x = scale.getPixelForValue(value);
        ctx.moveTo(x, chart.chartArea.top);
        ctx.lineTo(x, chart.chartArea.bottom);
    } else {
        const y = scale.getPixelForValue(value);
        ctx.moveTo(chart.chartArea.left, y);
        ctx.lineTo(chart.chartArea.right, y);
    }
    ctx.stroke();
    
    // Add label if required
    if (annotation.label && annotation.label.enabled) {
        drawAnnotationLabel(chart, ctx, annotation, isVertical);
    }
    
    ctx.restore();
}

/**
 * Draw an annotation label
 */
function drawAnnotationLabel(chart, ctx, annotation, isVertical) {
    const label = annotation.label;
    const scaleId = annotation.scaleID || 'x';
    const scale = chart.scales[scaleId];
    const value = annotation.value;
    
    // Prepare label style
    ctx.fillStyle = label.backgroundColor || 'rgba(0,0,0,0.8)';
    ctx.font = label.font || '12px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    // Calculate label position
    let x, y;
    const textWidth = ctx.measureText(label.content).width + 10;
    const textHeight = 20;
    
    if (isVertical) {
        x = scale.getPixelForValue(value);
        
        if (label.position === 'top') {
            y = chart.chartArea.top + 10;
        } else if (label.position === 'bottom') {
            y = chart.chartArea.bottom - 10;
        } else {
            y = (chart.chartArea.top + chart.chartArea.bottom) / 2;
        }
    } else {
        y = scale.getPixelForValue(value);
        
        if (label.position === 'left') {
            x = chart.chartArea.left + 30;
        } else if (label.position === 'right') {
            x = chart.chartArea.right - 30;
        } else {
            x = (chart.chartArea.left + chart.chartArea.right) / 2;
        }
    }
    
    // Draw label background
    ctx.fillStyle = 'rgba(255,255,255,0.8)';
    ctx.fillRect(x - textWidth / 2, y - textHeight / 2, textWidth, textHeight);
    
    // Draw label border
    ctx.strokeStyle = annotation.borderColor || 'rgba(0,0,0,0.5)';
    ctx.lineWidth = 1;
    ctx.strokeRect(x - textWidth / 2, y - textHeight / 2, textWidth, textHeight);
    
    // Draw label text
    ctx.fillStyle = 'rgba(0,0,0,0.8)';
    ctx.fillText(label.content, x, y);
}

/**
 * Register the annotation plugin with Chart.js
 */
export function registerChartPlugins() {
    if (window.Chart) {
        window.Chart.register(annotationPlugin);
    } else {
        console.error('Chart.js not found. Make sure it is loaded before this script.');
    }
} 