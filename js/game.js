/**
 * Game Module
 * Core game logic, modes, timer, and character selection
 */

import * as storage from './storage.js';

// Character ranges
const RANGES = {
    '1-10': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
    '0-99': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    'A-Z': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split(''),
    '0-Z': [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z'
    ]
};

// Game state
let currentState = {
    range: '1-10',
    mode: 'both',     // 'trace', 'audible', 'both'
    timerDuration: 0, // 0 = unlimited
    currentCharacter: null,
    timerValue: 0,
    timerInterval: null,
    isTimerRunning: false,
    isPaused: false
};

// Event callbacks
let onTimerTick = null;
let onTimerEnd = null;
let onCharacterChange = null;

/**
 * Initialize game with saved settings
 */
export function initGame(callbacks = {}) {
    onTimerTick = callbacks.onTimerTick;
    onTimerEnd = callbacks.onTimerEnd;
    onCharacterChange = callbacks.onCharacterChange;

    // Load saved settings
    const settings = storage.getSettings();
    currentState.range = settings.range;
    currentState.mode = settings.mode;
    currentState.timerDuration = settings.timer;

    return currentState;
}

/**
 * Set character range
 */
export function setRange(range) {
    if (RANGES[range]) {
        currentState.range = range;
        storage.updateSettings({ range });
    }
}

/**
 * Set display mode
 */
export function setMode(mode) {
    if (['trace', 'audible', 'both'].includes(mode)) {
        currentState.mode = mode;
        storage.updateSettings({ mode });
    }
}

/**
 * Set timer duration (seconds, 0 = unlimited)
 */
export function setTimerDuration(seconds) {
    currentState.timerDuration = seconds;
    storage.updateSettings({ timer: seconds });
}

/**
 * Get current settings
 */
export function getSettings() {
    return { ...currentState };
}

/**
 * Get characters for current range
 */
export function getCurrentCharacters() {
    return RANGES[currentState.range] || RANGES['1-10'];
}

/**
 * Select a random character from current range
 * Weighted towards less-practiced characters
 */
export function selectNextCharacter() {
    const characters = getCurrentCharacters();
    const charStats = storage.getCharacterStats();

    // Weight characters by inverse of practice (less practiced = higher weight)
    const weights = characters.map(char => {
        const stats = charStats[char];
        if (!stats || stats.attempts === 0) {
            return 10; // High weight for unpracticed
        }
        const accuracy = stats.correct / stats.attempts;
        // Lower accuracy = higher weight (more practice needed)
        return Math.max(1, 10 - (accuracy * 10));
    });

    // Weighted random selection
    const totalWeight = weights.reduce((a, b) => a + b, 0);
    let random = Math.random() * totalWeight;

    for (let i = 0; i < characters.length; i++) {
        random -= weights[i];
        if (random <= 0) {
            currentState.currentCharacter = characters[i];
            if (onCharacterChange) {
                onCharacterChange(characters[i]);
            }
            return characters[i];
        }
    }

    // Fallback to first character
    currentState.currentCharacter = characters[0];
    return characters[0];
}

/**
 * Get current character
 */
export function getCurrentCharacter() {
    return currentState.currentCharacter;
}

/**
 * Check if trace should be shown
 */
export function shouldShowTrace() {
    return currentState.mode === 'trace' || currentState.mode === 'both';
}

/**
 * Check if audio should be played
 */
export function shouldPlayAudio() {
    return currentState.mode === 'audible' || currentState.mode === 'both';
}

/**
 * Start the timer
 */
export function startTimer() {
    if (currentState.timerDuration === 0) {
        // Unlimited mode - no timer
        currentState.isTimerRunning = false;
        return;
    }

    stopTimer(); // Clear any existing timer

    currentState.timerValue = currentState.timerDuration;
    currentState.isTimerRunning = true;
    currentState.timerStartTime = performance.now();
    currentState.timerEndTime = currentState.timerStartTime + (currentState.timerDuration * 1000);

    if (onTimerTick) {
        onTimerTick(currentState.timerValue, 1.0); // progress = 1.0 at start
    }

    // Use requestAnimationFrame for smooth updates
    function updateTimer() {
        if (!currentState.isTimerRunning) return;
        if (currentState.isPaused) {
            currentState.timerAnimationFrame = requestAnimationFrame(updateTimer);
            return;
        }

        const now = performance.now();
        const elapsed = now - currentState.timerStartTime;
        const total = currentState.timerDuration * 1000;
        const remaining = Math.max(0, total - elapsed);
        const progress = remaining / total;

        currentState.timerValue = Math.ceil(remaining / 1000);

        if (onTimerTick) {
            onTimerTick(currentState.timerValue, progress);
        }

        if (remaining <= 0) {
            stopTimer();
            if (onTimerEnd) {
                onTimerEnd();
            }
            return;
        }

        currentState.timerAnimationFrame = requestAnimationFrame(updateTimer);
    }

    currentState.timerAnimationFrame = requestAnimationFrame(updateTimer);
}

/**
 * Stop the timer
 */
export function stopTimer() {
    if (currentState.timerAnimationFrame) {
        cancelAnimationFrame(currentState.timerAnimationFrame);
        currentState.timerAnimationFrame = null;
    }
    if (currentState.timerInterval) {
        clearInterval(currentState.timerInterval);
        currentState.timerInterval = null;
    }
    currentState.isTimerRunning = false;
}

/**
 * Pause the timer
 */
export function pauseTimer() {
    currentState.isPaused = true;
}

/**
 * Resume the timer
 */
export function resumeTimer() {
    currentState.isPaused = false;
}

/**
 * Get remaining time
 */
export function getRemainingTime() {
    return currentState.timerValue;
}

/**
 * Check if timer is active (not unlimited mode)
 */
export function isTimerActive() {
    return currentState.timerDuration > 0;
}

/**
 * Reset game (for new session)
 */
export function resetGame() {
    stopTimer();
    currentState.currentCharacter = null;
    currentState.isPaused = false;
}

/**
 * Get all available ranges
 */
export function getAvailableRanges() {
    return Object.keys(RANGES);
}

/**
 * Map model prediction index to character label
 */
export function predictionToLabel(index) {
    // EMNIST Balanced label mapping
    const labels = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e',
        'f', 'g', 'h', 'n', 'q', 'r', 't'
    ];
    return labels[index] || '?';
}

/**
 * Check if prediction matches target (accounting for case)
 */
export function isPredictionCorrect(prediction, target) {
    // Direct match
    if (prediction === target) return true;

    // Case-insensitive for letters (since EMNIST merges some cases)
    if (prediction.toLowerCase() === target.toLowerCase()) return true;

    return false;
}
