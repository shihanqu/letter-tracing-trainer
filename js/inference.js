/**
 * Inference Module
 * ONNX Runtime Web for ML model inference
 */

let session = null;
let isModelLoaded = false;
let useWebGPU = false;

// Model configuration
const MODEL_PATH = 'models/emnist_balanced.onnx';
const INPUT_NAME = 'input';
const OUTPUT_NAME = 'output';

// EMNIST Balanced labels (47 classes)
const LABELS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e',
    'f', 'g', 'h', 'n', 'q', 'r', 't'
];

/**
 * Check WebGPU support
 */
async function checkWebGPU() {
    if (!navigator.gpu) {
        console.log('WebGPU not supported, will use WASM backend');
        return false;
    }

    try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            console.log('No WebGPU adapter found, will use WASM backend');
            return false;
        }
        console.log('WebGPU is supported');
        return true;
    } catch (e) {
        console.log('WebGPU check failed:', e);
        return false;
    }
}

/**
 * Initialize ONNX Runtime and load model
 */
export async function initInference(onProgress) {
    if (isModelLoaded) return true;

    try {
        if (onProgress) onProgress('Checking browser capabilities...');

        // Configure ONNX Runtime
        if (typeof ort === 'undefined') {
            throw new Error('ONNX Runtime not loaded. Make sure ort.min.js is included.');
        }

        // Set WASM paths for local files
        ort.env.wasm.wasmPaths = 'lib/';

        if (onProgress) onProgress('Loading AI model...');

        // Try WebGPU first if available, fallback to WASM
        const hasWebGPU = await checkWebGPU();

        if (hasWebGPU) {
            try {
                if (onProgress) onProgress('Trying WebGPU backend...');
                session = await ort.InferenceSession.create(MODEL_PATH, {
                    executionProviders: ['webgpu'],
                    graphOptimizationLevel: 'all'
                });
                useWebGPU = true;
                console.log('Model loaded with WebGPU backend');
            } catch (webgpuError) {
                console.log('WebGPU failed, falling back to WASM:', webgpuError.message);
                useWebGPU = false;
            }
        }

        // If WebGPU failed or not available, use WASM
        if (!session) {
            if (onProgress) onProgress('Loading with WASM backend...');
            session = await ort.InferenceSession.create(MODEL_PATH, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            });
            console.log('Model loaded with WASM backend');
        }

        console.log('Model loaded successfully');
        console.log('Input names:', session.inputNames);
        console.log('Output names:', session.outputNames);
        console.log('Using backend:', useWebGPU ? 'WebGPU' : 'WASM');

        isModelLoaded = true;

        if (onProgress) onProgress('Ready!');

        return true;
    } catch (error) {
        console.error('Failed to load model:', error);

        if (onProgress) onProgress('Model loading failed: ' + error.message);

        return false;
    }
}

/**
 * Run inference on preprocessed image data
 * @param {Float32Array} imageData - 28x28 normalized grayscale image
 * @returns {Object} - { label, confidence, allPredictions }
 */
export async function predict(imageData) {
    if (!isModelLoaded || !session) {
        throw new Error('Model not loaded. Call initInference() first.');
    }

    try {
        // Create input tensor [1, 1, 28, 28] - batch, channels, height, width
        const inputTensor = new ort.Tensor('float32', imageData, [1, 1, 28, 28]);

        // Run inference
        const feeds = { [session.inputNames[0]]: inputTensor };
        const results = await session.run(feeds);

        // Get output
        const output = results[session.outputNames[0]];
        const probabilities = softmax(Array.from(output.data));

        // Find top prediction
        let maxIdx = 0;
        let maxProb = probabilities[0];

        for (let i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                maxIdx = i;
            }
        }

        // Get top 5 predictions
        const allPredictions = probabilities
            .map((prob, idx) => ({ label: LABELS[idx], probability: prob, index: idx }))
            .sort((a, b) => b.probability - a.probability)
            .slice(0, 5);

        return {
            label: LABELS[maxIdx],
            confidence: maxProb,
            index: maxIdx,
            allPredictions
        };
    } catch (error) {
        console.error('Inference error:', error);
        throw error;
    }
}

/**
 * Softmax function to convert logits to probabilities
 */
function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const scores = logits.map(l => Math.exp(l - maxLogit));
    const sum = scores.reduce((a, b) => a + b, 0);
    return scores.map(s => s / sum);
}

/**
 * Check if model is loaded
 */
export function isReady() {
    return isModelLoaded;
}

/**
 * Get backend type
 */
export function getBackend() {
    return useWebGPU ? 'WebGPU' : 'WASM';
}

/**
 * Get label for prediction index
 */
export function getLabel(index) {
    return LABELS[index] || '?';
}

/**
 * Get all labels
 */
export function getAllLabels() {
    return [...LABELS];
}

/**
 * Dispose of the session
 */
export async function dispose() {
    if (session) {
        await session.release();
        session = null;
        isModelLoaded = false;
    }
}
