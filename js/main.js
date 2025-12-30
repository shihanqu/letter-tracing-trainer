/**
 * Main Application Module
 * Coordinates all modules and handles UI interactions
 */

import * as storage from './storage.js';
import * as canvas from './canvas.js';
import * as inference from './inference.js';
import * as speech from './speech.js';
import * as game from './game.js';
import * as achievements from './achievements.js';
import * as dashboard from './dashboard.js';
import * as templates from './templates.js';

// DOM Elements
let elements = {};

// Application state
let appState = {
    currentView: 'game', // 'game', 'settings', or 'dashboard'
    isLoading: true,
    modelReady: false,
    timerTotal: 0
};

/**
 * Initialize the application
 */
async function init() {
    console.log('Initializing Letter Tracing App...');

    // Cache DOM elements
    cacheElements();

    // Initialize storage
    storage.initStorage();

    // Initialize speech
    await speech.initSpeech();

    // Initialize canvas
    canvas.initCanvas('drawing-canvas', 'trace-overlay');

    // Initialize game with callbacks
    game.initGame({
        onTimerTick: handleTimerTick,
        onTimerEnd: handleTimerEnd,
        onCharacterChange: handleCharacterChange
    });

    // Set up event listeners
    setupEventListeners();

    // Load saved settings into UI
    loadSettingsToUI();

    // Update UI with current stats
    updateStatsDisplay();

    // Initialize dashboard
    dashboard.initDashboard();

    // Load ML model
    await loadModel();

    // Start first round
    startNewRound();

    console.log('App initialized!');
}

/**
 * Cache frequently used DOM elements
 */
function cacheElements() {
    elements = {
        // Views
        gameView: document.getElementById('game-view'),
        settingsView: document.getElementById('settings-view'),
        dashboardView: document.getElementById('dashboard-view'),

        // Loading
        loadingOverlay: document.getElementById('loading-overlay'),
        loadingMessage: document.getElementById('loading-message'),

        // Game elements
        targetCharacter: document.getElementById('target-character'),

        // Timer border
        timerBorder: document.getElementById('timer-border'),
        timerProgress: document.getElementById('timer-progress'),

        // Stats
        streakDisplay: document.getElementById('streak-display'),
        streakCount: document.getElementById('streak-count'),
        scoreDisplay: document.getElementById('score-display'),

        // Buttons
        speakBtn: document.getElementById('speak-btn'),
        clearBtn: document.getElementById('clear-btn'),
        submitBtn: document.getElementById('submit-btn'),
        skipBtn: document.getElementById('skip-btn'),
        settingsBtn: document.getElementById('settings-btn'),
        dashboardBtn: document.getElementById('dashboard-btn'),
        backToGameBtn: document.getElementById('back-to-game'),
        backToGameFromSettingsBtn: document.getElementById('back-to-game-from-settings'),
        nextBtn: document.getElementById('next-btn'),
        resetAllBtn: document.getElementById('reset-all-btn'),
        resetConfirmBtn: document.getElementById('reset-confirm-btn'),
        resetCancelBtn: document.getElementById('reset-cancel-btn'),

        // Controls
        rangeOptions: document.getElementById('range-options'),
        modeOptions: document.getElementById('mode-options'),
        timerOptions: document.getElementById('timer-options'),

        // Modals
        resultModal: document.getElementById('result-modal'),
        resultTitle: document.getElementById('result-title'),
        resultIcon: document.getElementById('result-icon'),
        resultMessage: document.getElementById('result-message'),
        resetConfirmModal: document.getElementById('reset-confirm-modal'),

        // Toast
        toast: document.getElementById('toast'),
        toastMessage: document.getElementById('toast-message'),

        // Celebration
        celebrationOverlay: document.getElementById('celebration-overlay'),

        // Debug
        debugBtn: document.getElementById('debug-btn'),
        debugModal: document.getElementById('debug-modal'),
        debugClose: document.getElementById('debug-close'),
        debugMessage: document.getElementById('debug-message'),
        debugContent: document.getElementById('debug-content'),
        debugProcessed: document.getElementById('debug-processed'),
        debugTransposed: document.getElementById('debug-transposed'),
        debugHistogram: document.getElementById('debug-histogram')
    };
}

/**
 * Set up event listeners
 */
function setupEventListeners() {
    // Action buttons
    elements.speakBtn?.addEventListener('click', handleSpeak);
    elements.clearBtn?.addEventListener('click', handleClear);
    elements.submitBtn?.addEventListener('click', handleSubmit);
    elements.skipBtn?.addEventListener('click', handleSkip);
    elements.nextBtn?.addEventListener('click', handleNext);

    // Navigation
    elements.settingsBtn?.addEventListener('click', () => switchView('settings'));
    elements.dashboardBtn?.addEventListener('click', () => switchView('dashboard'));
    elements.backToGameBtn?.addEventListener('click', () => switchView('game'));
    elements.backToGameFromSettingsBtn?.addEventListener('click', () => switchView('game'));

    // Reset functionality
    elements.resetAllBtn?.addEventListener('click', showResetConfirm);
    elements.resetConfirmBtn?.addEventListener('click', handleResetAll);
    elements.resetCancelBtn?.addEventListener('click', hideResetConfirm);

    // Debug
    elements.debugBtn?.addEventListener('click', showDebugModal);
    elements.debugClose?.addEventListener('click', hideDebugModal);

    // Control options
    setupControlOptions();

    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyboard);
}

/**
 * Handle keyboard shortcuts
 */
function handleKeyboard(e) {
    if (appState.currentView !== 'game') return;

    switch (e.key) {
        case 'Enter':
            e.preventDefault();
            handleSubmit();
            break;
        case 'Escape':
            handleClear();
            break;
        case ' ':
            e.preventDefault();
            handleSkip();
            break;
    }
}

/**
 * Set up control option buttons
 */
function setupControlOptions() {
    // Range options
    elements.rangeOptions?.querySelectorAll('.setting-option').forEach(btn => {
        btn.addEventListener('click', (e) => {
            selectOption(elements.rangeOptions, e.target.closest('.setting-option'));
            game.setRange(e.target.closest('.setting-option').dataset.range);
        });
    });

    // Mode options
    elements.modeOptions?.querySelectorAll('.setting-option').forEach(btn => {
        btn.addEventListener('click', (e) => {
            selectOption(elements.modeOptions, e.target.closest('.setting-option'));
            game.setMode(e.target.closest('.setting-option').dataset.mode);
        });
    });

    // Timer options
    elements.timerOptions?.querySelectorAll('.setting-option').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const option = e.target.closest('.setting-option');
            selectOption(elements.timerOptions, option);
            const duration = parseInt(option.dataset.timer) || 0;
            game.setTimerDuration(duration);
            appState.timerTotal = duration;
        });
    });
}

/**
 * Select an option button and deselect others
 */
function selectOption(container, selectedBtn) {
    container.querySelectorAll('.setting-option').forEach(btn => {
        btn.classList.remove('active');
    });
    selectedBtn.classList.add('active');
}

/**
 * Load settings into UI
 */
function loadSettingsToUI() {
    const settings = game.getSettings();

    // Set active options based on saved settings
    setActiveOption(elements.rangeOptions, 'range', settings.range);
    setActiveOption(elements.modeOptions, 'mode', settings.mode);
    setActiveOption(elements.timerOptions, 'timer', settings.timerDuration.toString());

    appState.timerTotal = settings.timerDuration;
}

/**
 * Set active option by data attribute
 */
function setActiveOption(container, dataAttr, value) {
    if (!container) return;

    container.querySelectorAll('.setting-option').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset[dataAttr] === value) {
            btn.classList.add('active');
        }
    });
}

/**
 * Load ML model
 */
async function loadModel() {
    showLoading(true, 'Loading AI model...');

    try {
        const success = await inference.initInference((message) => {
            updateLoadingMessage(message);
        });

        appState.modelReady = success;

        if (!success) {
            showToast('Model failed to load. Running in demo mode.', 'error');
        }
    } catch (error) {
        console.error('Model loading error:', error);
        showToast('Model error: ' + error.message, 'error');
    }

    showLoading(false);
}

/**
 * Start a new round
 */
function startNewRound() {
    // Clear canvas
    canvas.clearCanvas();

    // Select next character
    const char = game.selectNextCharacter();

    // Update display
    updateCharacterDisplay(char);

    // Update trace visibility
    updateTraceVisibility();

    // Play audio if in audible mode
    if (game.shouldPlayAudio()) {
        speech.speakPrompt(char);
    }

    // Start timer
    game.startTimer();
    updateTimerBorder();
}

/**
 * Update character display
 */
function updateCharacterDisplay(char) {
    if (elements.targetCharacter) {
        elements.targetCharacter.textContent = char;
        elements.targetCharacter.classList.add('animate-bounce-in');
        setTimeout(() => {
            elements.targetCharacter.classList.remove('animate-bounce-in');
        }, 500);
    }
}

/**
 * Update trace visibility based on mode
 */
function updateTraceVisibility() {
    const char = game.getCurrentCharacter();

    if (game.shouldShowTrace() && char) {
        canvas.drawTrace(char);
    } else {
        canvas.clearTraceOverlay();
    }
}

/**
 * Handle character change
 */
function handleCharacterChange(char) {
    updateCharacterDisplay(char);
}

/**
 * Handle speak button
 */
function handleSpeak() {
    const char = game.getCurrentCharacter();
    if (char) {
        speech.speakPrompt(char);
    }
}

/**
 * Handle clear button
 */
function handleClear() {
    canvas.clearCanvas();
}

/**
 * Handle submit button
 */
async function handleSubmit(autoSubmit = false) {
    if (!canvas.hasContent()) {
        if (!autoSubmit) {
            showToast('Draw something first!', 'info');
        }
        return { submitted: false };
    }

    game.stopTimer();

    const targetChar = game.getCurrentCharacter();

    // Check if model is ready
    if (!appState.modelReady) {
        // Demo mode - random result
        const isCorrect = Math.random() > 0.3;
        processResult(isCorrect, targetChar, isCorrect ? targetChar : 'X', 0, false);
        return { submitted: true };
    }

    try {
        // Preprocess canvas
        const imageData = canvas.preprocessForModel();

        // Run inference
        const prediction = await inference.predict(imageData);

        console.log('Prediction:', prediction);

        // Use loose matching for confusable characters
        const looseResult = game.isPredictionCorrectLoose(prediction.allPredictions, targetChar);

        processResult(looseResult.isCorrect, targetChar, looseResult.matchedPrediction || prediction.label, prediction.confidence, looseResult.isLooseMatch);
        return { submitted: true };
    } catch (error) {
        console.error('Prediction error:', error);
        showToast('Prediction failed: ' + error.message, 'error');
        return { submitted: false };
    }
}

/**
 * Process the result of a submission
 */
function processResult(isCorrect, targetChar, predictedChar, confidence = 0, isLooseMatch = false) {
    // Record attempt
    const result = achievements.recordAttempt(targetChar, isCorrect);

    // Update UI
    updateStatsDisplay();

    // Show result
    if (isCorrect) {
        showCorrectResult(result, isLooseMatch);
    } else {
        showIncorrectResult(predictedChar, result);
    }

    // Check for new achievements
    if (result.newAchievements.length > 0) {
        setTimeout(() => {
            result.newAchievements.forEach(achievement => {
                dashboard.showBadgeUnlock(achievement);
            });
        }, 1500);
    }
}

/**
 * Show correct result
 */
function showCorrectResult(result, isLooseMatch = false) {
    // Flash effect
    showFlash('success');

    // Create confetti
    createConfetti();

    // Speech feedback
    speech.speakCorrect();

    // Show modal - slightly different message for loose matches
    const message = isLooseMatch
        ? 'Close enough! ' + getStreakMessage(result.streakResult.currentStreak)
        : getStreakMessage(result.streakResult.currentStreak);

    showResultModal({
        title: '🎉 Correct!',
        icon: '⭐',
        message: message
    });
}

/**
 * Show incorrect result
 */
function showIncorrectResult(predictedChar, result) {
    // Flash effect
    showFlash('error');

    // Shake canvas
    document.getElementById('canvas-container')?.classList.add('animate-shake');
    setTimeout(() => {
        document.getElementById('canvas-container')?.classList.remove('animate-shake');
    }, 500);

    // Speech feedback
    speech.speakIncorrect(predictedChar);

    // Show modal
    showResultModal({
        title: 'Try Again!',
        icon: '💪',
        message: `That looked like "${predictedChar}". Keep practicing!`
    });
}

/**
 * Get streak message
 */
function getStreakMessage(streak) {
    if (streak >= 10) return `Amazing! ${streak} in a row! 🔥🔥🔥`;
    if (streak >= 5) return `On fire! ${streak} streak! 🔥`;
    if (streak >= 3) return `Great streak: ${streak}! Keep going!`;
    return 'Nice job! Keep it up!';
}

/**
 * Handle skip button
 */
function handleSkip() {
    speech.stopSpeaking();
    game.stopTimer();
    startNewRound();
}

/**
 * Handle next button (after result)
 */
function handleNext() {
    hideResultModal();
    startNewRound();
}

/**
 * Handle timer tick - update border
 */
function handleTimerTick(seconds, progress) {
    updateTimerBorder(seconds, progress);
}

/**
 * Update timer border visualization
 */
function updateTimerBorder(seconds, progress) {
    if (!elements.timerProgress) return;

    const total = appState.timerTotal;

    if (!game.isTimerActive() || total === 0) {
        elements.timerProgress.classList.add('hidden');
        return;
    }

    elements.timerProgress.classList.remove('hidden');

    // Use progress directly for smooth animation
    const p = progress !== undefined ? progress : (seconds / total);

    // Calculate stroke-dashoffset (392 is perimeter of the rect)
    const perimeter = 392;
    const offset = perimeter * (1 - p);
    elements.timerProgress.style.strokeDashoffset = offset;

    // Update color based on time
    elements.timerProgress.classList.remove('warning', 'danger');
    if (seconds <= 3) {
        elements.timerProgress.classList.add('danger');
    } else if (seconds <= 5) {
        elements.timerProgress.classList.add('warning');
    }
}

/**
 * Handle timer end
 */
async function handleTimerEnd() {
    // Time's up - auto-submit if there's content
    speech.speak("Time's up!");

    if (canvas.hasContent()) {
        // Auto-submit the drawing
        showToast("Time's up! Checking your drawing...", 'info');
        await handleSubmit(true);
    } else {
        // No content - treat as skip
        showToast("Time's up! Try again.", 'info');
        const targetChar = game.getCurrentCharacter();
        achievements.recordAttempt(targetChar, false);
        updateStatsDisplay();
        setTimeout(startNewRound, 1500);
    }
}

/**
 * Update stats display (streak, score)
 */
function updateStatsDisplay() {
    const progress = storage.getProgress();

    if (elements.streakCount) {
        elements.streakCount.textContent = progress.currentStreak;
    }

    if (elements.scoreDisplay) {
        elements.scoreDisplay.textContent = progress.score;
    }

    // Also update dashboard if visible
    dashboard.updateDashboardStats();
}

/**
 * Switch between views
 */
function switchView(view) {
    // Stop timer when leaving game
    if (appState.currentView === 'game' && view !== 'game') {
        game.stopTimer();
    }

    appState.currentView = view;

    // Hide all views
    elements.gameView?.classList.remove('active');
    elements.settingsView?.classList.remove('active');
    elements.dashboardView?.classList.remove('active');

    // Show selected view
    if (view === 'game') {
        elements.gameView?.classList.add('active');
        startNewRound();
    } else if (view === 'settings') {
        elements.settingsView?.classList.add('active');
    } else if (view === 'dashboard') {
        elements.dashboardView?.classList.add('active');
        dashboard.initDashboard();
    }
}

/**
 * Show/hide loading overlay
 */
function showLoading(show, message = 'Loading...') {
    appState.isLoading = show;

    if (elements.loadingOverlay) {
        if (show) {
            elements.loadingOverlay.classList.add('visible');
        } else {
            elements.loadingOverlay.classList.remove('visible');
        }
    }

    updateLoadingMessage(message);
}

/**
 * Update loading message
 */
function updateLoadingMessage(message) {
    if (elements.loadingMessage) {
        elements.loadingMessage.textContent = message;
    }
}

/**
 * Show result modal
 */
function showResultModal({ title, icon, message }) {
    if (elements.resultTitle) elements.resultTitle.textContent = title;
    if (elements.resultIcon) elements.resultIcon.textContent = icon;
    if (elements.resultMessage) elements.resultMessage.textContent = message;

    elements.resultModal?.classList.add('visible');
}

/**
 * Hide result modal
 */
function hideResultModal() {
    elements.resultModal?.classList.remove('visible');
}

/**
 * Show reset confirmation modal
 */
function showResetConfirm() {
    elements.resetConfirmModal?.classList.add('visible');
}

/**
 * Hide reset confirmation modal
 */
function hideResetConfirm() {
    elements.resetConfirmModal?.classList.remove('visible');
}

/**
 * Handle reset all stats
 */
function handleResetAll() {
    storage.resetAll();
    hideResetConfirm();
    showToast('All progress has been reset!', 'info');
    updateStatsDisplay();
    dashboard.initDashboard();
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
    if (!elements.toast || !elements.toastMessage) return;

    elements.toastMessage.textContent = message;
    elements.toast.className = `toast toast-${type} visible`;

    setTimeout(() => {
        elements.toast.classList.remove('visible');
    }, 3000);
}

/**
 * Show flash effect
 */
function showFlash(type) {
    const flash = document.createElement('div');
    flash.className = `${type}-flash`;
    document.body.appendChild(flash);

    setTimeout(() => flash.remove(), 500);
}

/**
 * Create confetti effect
 */
function createConfetti() {
    const overlay = elements.celebrationOverlay;
    if (!overlay) return;

    // Clear previous confetti
    overlay.innerHTML = '';

    // Create confetti pieces
    for (let i = 0; i < 50; i++) {
        const confetti = document.createElement('div');
        confetti.className = 'confetti';
        confetti.style.left = Math.random() * 100 + 'vw';
        confetti.style.animationDelay = Math.random() * 0.5 + 's';
        confetti.style.animationDuration = (2 + Math.random() * 2) + 's';
        overlay.appendChild(confetti);
    }

    // Clean up after animation
    setTimeout(() => {
        overlay.innerHTML = '';
    }, 4000);
}

/**
 * Show debug modal and immediately run analysis
 */
async function showDebugModal() {
    elements.debugModal?.classList.add('visible');
    // Immediately run debug analysis
    await runDebug();
}

/**
 * Hide debug modal
 */
function hideDebugModal() {
    elements.debugModal?.classList.remove('visible');
}

/**
 * Run debug - show preprocessed image and predictions
 */
async function runDebug() {
    // Show message and hide content if no canvas content
    if (!canvas.hasContent()) {
        if (elements.debugMessage) {
            elements.debugMessage.textContent = 'Draw something on the canvas first, then click the debug button again.';
            elements.debugMessage.style.display = 'block';
        }
        // Hide the actual debug content
        const contentParts = elements.debugContent?.querySelectorAll('h3, div');
        contentParts?.forEach(el => el.style.display = 'none');
        return;
    }

    // Show content, hide message
    if (elements.debugMessage) {
        elements.debugMessage.style.display = 'none';
    }
    const contentParts = elements.debugContent?.querySelectorAll('h3, div');
    contentParts?.forEach(el => el.style.display = '');

    // Get debug visualization data
    const debugData = canvas.getDebugData();

    if (!debugData || !debugData.processedDataUrl) {
        if (elements.debugMessage) {
            elements.debugMessage.textContent = 'Unable to process drawing.';
            elements.debugMessage.style.display = 'block';
        }
        return;
    }

    // Show images
    if (elements.debugProcessed) {
        elements.debugProcessed.src = debugData.processedDataUrl;
    }
    if (elements.debugTransposed) {
        elements.debugTransposed.src = debugData.transposedDataUrl;
    }

    // Run inference and show histogram
    if (appState.modelReady) {
        try {
            const imageData = canvas.preprocessForModel();
            const prediction = await inference.predict(imageData);

            // Render histogram of top 10 predictions
            renderPredictionHistogram(prediction.allPredictions);
        } catch (error) {
            console.error('Debug inference error:', error);
            if (elements.debugHistogram) {
                elements.debugHistogram.innerHTML = `<p style="color: var(--color-danger);">Error: ${error.message}</p>`;
            }
        }
    } else {
        if (elements.debugHistogram) {
            elements.debugHistogram.innerHTML = '<p style="color: var(--color-warning);">Model not loaded - histogram unavailable</p>';
        }
    }

    // Show debug content
    elements.debugContent.style.display = 'block';
}

/**
 * Render prediction histogram
 */
function renderPredictionHistogram(predictions) {
    if (!elements.debugHistogram) return;

    // Get top 10 (or however many we have)
    const top10 = predictions.slice(0, 10);
    const maxProb = top10[0]?.probability || 1;

    const html = top10.map((pred, idx) => {
        const barWidth = (pred.probability / maxProb) * 100;
        const percent = (pred.probability * 100).toFixed(1);
        const isTop = idx === 0;

        return `
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="width: 30px; font-weight: ${isTop ? 'bold' : 'normal'}; color: ${isTop ? 'var(--color-primary)' : 'var(--color-text)'};">${pred.label}</span>
                <div style="flex: 1; background: var(--color-surface); border-radius: 4px; overflow: hidden; margin: 0 0.5rem;">
                    <div style="width: ${barWidth}%; height: 20px; background: ${isTop ? 'var(--color-primary)' : 'var(--color-secondary)'}; transition: width 0.3s;"></div>
                </div>
                <span style="width: 50px; text-align: right; font-size: 0.8rem; color: var(--color-text-muted);">${percent}%</span>
            </div>
        `;
    }).join('');

    elements.debugHistogram.innerHTML = html;
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
