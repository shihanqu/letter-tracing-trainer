/**
 * Canvas Module
 * WebGPU/Canvas2D drawing with preprocessing for ML inference
 */

let canvas = null;
let ctx = null;
let traceCanvas = null;
let traceCtx = null;
let isDrawing = false;
let lastX = 0;
let lastY = 0;
let hasDrawing = false;

// Drawing settings
const settings = {
    lineWidth: 20,
    lineColor: '#FFFFFF',
    lineCap: 'round',
    lineJoin: 'round',
    smoothing: true
};

/**
 * Initialize canvases
 */
export function initCanvas(canvasId, traceCanvasId) {
    canvas = document.getElementById(canvasId);
    traceCanvas = document.getElementById(traceCanvasId);

    if (!canvas) {
        console.error('Drawing canvas not found:', canvasId);
        return false;
    }

    // Set up main drawing canvas
    ctx = canvas.getContext('2d');
    setupCanvas(canvas, ctx);

    // Set up trace overlay canvas
    if (traceCanvas) {
        traceCtx = traceCanvas.getContext('2d');
        setupCanvas(traceCanvas, traceCtx);
    }

    // Set up event listeners
    setupEventListeners();

    return true;
}

/**
 * Set up canvas size and styling
 */
function setupCanvas(canvasEl, context) {
    const container = canvasEl.parentElement;
    const rect = container.getBoundingClientRect();

    // Set actual size in memory
    const dpr = window.devicePixelRatio || 1;
    canvasEl.width = rect.width * dpr;
    canvasEl.height = rect.height * dpr;

    // Scale for HiDPI displays
    context.scale(dpr, dpr);

    // Set display size
    canvasEl.style.width = rect.width + 'px';
    canvasEl.style.height = rect.height + 'px';
}

/**
 * Set up mouse/touch event listeners
 */
function setupEventListeners() {
    // Mouse events
    canvas.addEventListener('mousedown', handleStart);
    canvas.addEventListener('mousemove', handleMove);
    canvas.addEventListener('mouseup', handleEnd);
    canvas.addEventListener('mouseout', handleEnd);

    // Touch events
    canvas.addEventListener('touchstart', handleTouchStart, { passive: false });
    canvas.addEventListener('touchmove', handleTouchMove, { passive: false });
    canvas.addEventListener('touchend', handleEnd);
    canvas.addEventListener('touchcancel', handleEnd);

    // Prevent context menu on long press
    canvas.addEventListener('contextmenu', e => e.preventDefault());

    // Handle resize
    window.addEventListener('resize', handleResize);
}

/**
 * Handle resize
 */
function handleResize() {
    // Save current drawing
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    tempCtx.drawImage(canvas, 0, 0);

    // Resize canvases
    setupCanvas(canvas, ctx);
    if (traceCanvas && traceCtx) {
        setupCanvas(traceCanvas, traceCtx);
    }

    // Restore drawing
    const dpr = window.devicePixelRatio || 1;
    ctx.drawImage(tempCanvas, 0, 0, canvas.width / dpr, canvas.height / dpr);
}

/**
 * Get position from mouse event
 */
function getMousePos(e) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
    };
}

/**
 * Get position from touch event
 */
function getTouchPos(e) {
    const rect = canvas.getBoundingClientRect();
    const touch = e.touches[0];
    return {
        x: touch.clientX - rect.left,
        y: touch.clientY - rect.top
    };
}

/**
 * Start drawing
 */
function handleStart(e) {
    isDrawing = true;
    const pos = getMousePos(e);
    lastX = pos.x;
    lastY = pos.y;

    // Draw a dot for single clicks
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, settings.lineWidth / 2, 0, Math.PI * 2);
    ctx.fillStyle = settings.lineColor;
    ctx.fill();

    hasDrawing = true;
}

/**
 * Handle touch start
 */
function handleTouchStart(e) {
    e.preventDefault();
    isDrawing = true;
    const pos = getTouchPos(e);
    lastX = pos.x;
    lastY = pos.y;

    // Draw a dot for single taps
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, settings.lineWidth / 2, 0, Math.PI * 2);
    ctx.fillStyle = settings.lineColor;
    ctx.fill();

    hasDrawing = true;
}

/**
 * Draw while moving
 */
function handleMove(e) {
    if (!isDrawing) return;

    const pos = getMousePos(e);
    draw(pos.x, pos.y);
}

/**
 * Handle touch move
 */
function handleTouchMove(e) {
    e.preventDefault();
    if (!isDrawing) return;

    const pos = getTouchPos(e);
    draw(pos.x, pos.y);
}

/**
 * Draw line segment
 */
function draw(x, y) {
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.strokeStyle = settings.lineColor;
    ctx.lineWidth = settings.lineWidth;
    ctx.lineCap = settings.lineCap;
    ctx.lineJoin = settings.lineJoin;
    ctx.stroke();

    lastX = x;
    lastY = y;
    hasDrawing = true;
}

/**
 * End drawing
 */
function handleEnd() {
    isDrawing = false;
}

/**
 * Clear the drawing canvas
 */
export function clearCanvas() {
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    ctx.clearRect(0, 0, canvas.width / dpr, canvas.height / dpr);
    hasDrawing = false;
}

/**
 * Clear the trace overlay
 */
export function clearTraceOverlay() {
    if (!traceCtx) return;

    const dpr = window.devicePixelRatio || 1;
    traceCtx.clearRect(0, 0, traceCanvas.width / dpr, traceCanvas.height / dpr);
}

/**
 * Draw trace template on overlay
 */
export function drawTrace(char) {
    if (!traceCtx) return;

    clearTraceOverlay();

    // Import templates dynamically to avoid circular deps
    import('./templates.js').then(templates => {
        const dpr = window.devicePixelRatio || 1;
        const width = traceCanvas.width / dpr;
        const height = traceCanvas.height / dpr;

        templates.drawTraceOnCanvas(traceCtx, char, width, height);
    });
}

/**
 * Check if canvas has any drawing
 */
export function hasContent() {
    return hasDrawing;
}

/**
 * Get canvas data as ImageData
 */
export function getImageData() {
    if (!ctx) return null;

    const dpr = window.devicePixelRatio || 1;
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

/**
 * Preprocess canvas for ML model input
 * Returns a 28x28 grayscale Float32Array normalized [0, 1]
 * 
 * IMPORTANT: EMNIST images are horizontally flipped compared to natural drawing.
 * We must mirror our drawing left-right to match the training data orientation.
 */
export function preprocessForModel() {
    if (!ctx) return null;

    const dpr = window.devicePixelRatio || 1;
    const width = canvas.width;
    const height = canvas.height;

    // Get original image data
    const imageData = ctx.getImageData(0, 0, width, height);

    // Find bounding box of drawing
    const bounds = findBoundingBox(imageData);
    if (!bounds) {
        // No drawing found
        return new Float32Array(28 * 28).fill(0);
    }

    // Create temporary canvas for processing
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = 28;
    tempCanvas.height = 28;

    // Calculate scaling to fit in 20x20 (with 4px padding like MNIST)
    const { x, y, w, h } = bounds;
    const scale = Math.min(20 / w, 20 / h);
    const scaledW = w * scale;
    const scaledH = h * scale;
    const offsetX = (28 - scaledW) / 2;
    const offsetY = (28 - scaledH) / 2;

    // Draw scaled and centered
    tempCtx.fillStyle = 'black';
    tempCtx.fillRect(0, 0, 28, 28);
    tempCtx.drawImage(
        canvas,
        x, y, w, h,
        offsetX, offsetY, scaledW, scaledH
    );

    // Get pixel data and convert to grayscale
    const processedData = tempCtx.getImageData(0, 0, 28, 28);
    const pixels = processedData.data;

    // Create result array - HORIZONTALLY FLIPPED to match EMNIST format
    // EMNIST images are mirrored left-right compared to how we naturally draw
    const result = new Float32Array(28 * 28);

    for (let row = 0; row < 28; row++) {
        for (let col = 0; col < 28; col++) {
            // Original position
            const srcIdx = (row * 28 + col) * 4;

            // Horizontally flipped position (mirror left-right)
            const flippedCol = 27 - col;
            const dstIdx = row * 28 + flippedCol;

            const r = pixels[srcIdx];
            const g = pixels[srcIdx + 1];
            const b = pixels[srcIdx + 2];
            const a = pixels[srcIdx + 3];

            // Calculate luminance
            const luminance = (0.299 * r + 0.587 * g + 0.114 * b) * (a / 255);

            // Normalize to [0, 1]
            result[dstIdx] = luminance / 255;
        }
    }

    return result;
}

/**
 * Get debug visualization data
 * Returns an object with canvas data URL and histogram for debugging
 */
export function getDebugData() {
    if (!ctx) return null;

    const dpr = window.devicePixelRatio || 1;
    const width = canvas.width;
    const height = canvas.height;

    // Get original image data
    const imageData = ctx.getImageData(0, 0, width, height);
    const bounds = findBoundingBox(imageData);

    if (!bounds) {
        return {
            originalDataUrl: null,
            processedDataUrl: null,
            transposedDataUrl: null,
            message: 'No drawing found'
        };
    }

    // Step 1: Create 28x28 scaled version
    const scaledCanvas = document.createElement('canvas');
    const scaledCtx = scaledCanvas.getContext('2d');
    scaledCanvas.width = 28;
    scaledCanvas.height = 28;

    const { x, y, w, h } = bounds;
    const scale = Math.min(20 / w, 20 / h);
    const scaledW = w * scale;
    const scaledH = h * scale;
    const offsetX = (28 - scaledW) / 2;
    const offsetY = (28 - scaledH) / 2;

    scaledCtx.fillStyle = 'black';
    scaledCtx.fillRect(0, 0, 28, 28);
    scaledCtx.drawImage(canvas, x, y, w, h, offsetX, offsetY, scaledW, scaledH);

    const processedDataUrl = scaledCanvas.toDataURL();

    // Step 2: Create horizontally flipped version (what model sees)
    const flippedCanvas = document.createElement('canvas');
    const flippedCtx = flippedCanvas.getContext('2d');
    flippedCanvas.width = 28;
    flippedCanvas.height = 28;

    const srcData = scaledCtx.getImageData(0, 0, 28, 28);
    const dstData = flippedCtx.createImageData(28, 28);

    for (let row = 0; row < 28; row++) {
        for (let col = 0; col < 28; col++) {
            const srcIdx = (row * 28 + col) * 4;
            // Horizontal flip: mirror left-right
            const flippedCol = 27 - col;
            const dstIdx = (row * 28 + flippedCol) * 4;

            dstData.data[dstIdx] = srcData.data[srcIdx];
            dstData.data[dstIdx + 1] = srcData.data[srcIdx + 1];
            dstData.data[dstIdx + 2] = srcData.data[srcIdx + 2];
            dstData.data[dstIdx + 3] = srcData.data[srcIdx + 3];
        }
    }

    flippedCtx.putImageData(dstData, 0, 0);
    const flippedDataUrl = flippedCanvas.toDataURL();

    return {
        processedDataUrl,
        transposedDataUrl: flippedDataUrl,  // Keep same key name for compatibility
        bounds,
        message: 'Flipped image is what the model sees'
    };
}

/**
 * Find bounding box of non-empty pixels
 */
function findBoundingBox(imageData) {
    const { width, height, data } = imageData;
    let minX = width, minY = height, maxX = 0, maxY = 0;
    let found = false;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const offset = (y * width + x) * 4;
            const alpha = data[offset + 3];
            const luminance = data[offset] + data[offset + 1] + data[offset + 2];

            // Check if pixel is drawn (has content)
            if (alpha > 10 && luminance > 30) {
                found = true;
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
    }

    if (!found) return null;

    // Add small margin
    const margin = 10;
    minX = Math.max(0, minX - margin);
    minY = Math.max(0, minY - margin);
    maxX = Math.min(width - 1, maxX + margin);
    maxY = Math.min(height - 1, maxY + margin);

    return {
        x: minX,
        y: minY,
        w: maxX - minX + 1,
        h: maxY - minY + 1
    };
}

/**
 * Set line width
 */
export function setLineWidth(width) {
    settings.lineWidth = width;
}

/**
 * Set line color
 */
export function setLineColor(color) {
    settings.lineColor = color;
}

/**
 * Get canvas dimensions
 */
export function getDimensions() {
    if (!canvas) return { width: 0, height: 0 };

    const dpr = window.devicePixelRatio || 1;
    return {
        width: canvas.width / dpr,
        height: canvas.height / dpr
    };
}
