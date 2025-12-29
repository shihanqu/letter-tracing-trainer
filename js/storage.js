/**
 * Storage Module
 * LocalStorage wrapper for persistence
 */

const STORAGE_PREFIX = 'tracing_app_';

const defaultData = {
    // User progress
    progress: {
        totalAttempts: 0,
        correctAttempts: 0,
        currentStreak: 0,
        bestStreak: 0,
        score: 0
    },

    // Per-character stats: { 'A': { attempts: 0, correct: 0 }, ... }
    characterStats: {},

    // Unlocked achievements
    achievements: [],

    // Settings
    settings: {
        range: '1-10',
        mode: 'both',
        timer: 0,
        volume: 1.0
    },

    // Session history
    sessions: []
};

/**
 * Get item from storage
 */
export function getItem(key) {
    try {
        const data = localStorage.getItem(STORAGE_PREFIX + key);
        return data ? JSON.parse(data) : null;
    } catch (e) {
        console.error('Storage get error:', e);
        return null;
    }
}

/**
 * Set item in storage
 */
export function setItem(key, value) {
    try {
        localStorage.setItem(STORAGE_PREFIX + key, JSON.stringify(value));
        return true;
    } catch (e) {
        console.error('Storage set error:', e);
        return false;
    }
}

/**
 * Get all app data
 */
export function getAllData() {
    const data = {};
    for (const key of Object.keys(defaultData)) {
        data[key] = getItem(key) ?? defaultData[key];
    }
    return data;
}

/**
 * Initialize storage with defaults
 */
export function initStorage() {
    for (const [key, value] of Object.entries(defaultData)) {
        if (getItem(key) === null) {
            setItem(key, value);
        }
    }
}

/**
 * Get user progress
 */
export function getProgress() {
    return getItem('progress') ?? defaultData.progress;
}

/**
 * Update user progress
 */
export function updateProgress(updates) {
    const current = getProgress();
    const updated = { ...current, ...updates };
    setItem('progress', updated);
    return updated;
}

/**
 * Get character stats
 */
export function getCharacterStats() {
    return getItem('characterStats') ?? {};
}

/**
 * Update stats for a specific character
 */
export function updateCharacterStat(char, correct) {
    const stats = getCharacterStats();
    if (!stats[char]) {
        stats[char] = { attempts: 0, correct: 0 };
    }
    stats[char].attempts++;
    if (correct) {
        stats[char].correct++;
    }
    setItem('characterStats', stats);
    return stats[char];
}

/**
 * Get character mastery level (0-100)
 */
export function getCharacterMastery(char) {
    const stats = getCharacterStats();
    if (!stats[char] || stats[char].attempts === 0) {
        return 0;
    }
    return Math.round((stats[char].correct / stats[char].attempts) * 100);
}

/**
 * Get settings
 */
export function getSettings() {
    return getItem('settings') ?? defaultData.settings;
}

/**
 * Update settings
 */
export function updateSettings(updates) {
    const current = getSettings();
    const updated = { ...current, ...updates };
    setItem('settings', updated);
    return updated;
}

/**
 * Get unlocked achievements
 */
export function getAchievements() {
    return getItem('achievements') ?? [];
}

/**
 * Add achievement
 */
export function addAchievement(achievementId) {
    const achievements = getAchievements();
    if (!achievements.includes(achievementId)) {
        achievements.push(achievementId);
        setItem('achievements', achievements);
        return true; // New achievement unlocked
    }
    return false; // Already had it
}

/**
 * Add session to history
 */
export function addSession(sessionData) {
    const sessions = getItem('sessions') ?? [];
    sessions.push({
        ...sessionData,
        timestamp: Date.now()
    });
    // Keep only last 100 sessions
    if (sessions.length > 100) {
        sessions.shift();
    }
    setItem('sessions', sessions);
}

/**
 * Export all data as JSON
 */
export function exportData() {
    return JSON.stringify(getAllData(), null, 2);
}

/**
 * Import data from JSON
 */
export function importData(jsonString) {
    try {
        const data = JSON.parse(jsonString);
        for (const [key, value] of Object.entries(data)) {
            if (key in defaultData) {
                setItem(key, value);
            }
        }
        return true;
    } catch (e) {
        console.error('Import error:', e);
        return false;
    }
}

/**
 * Clear all data
 */
export function clearAllData() {
    for (const key of Object.keys(defaultData)) {
        localStorage.removeItem(STORAGE_PREFIX + key);
    }
    initStorage();
}

/**
 * Reset all progress (alias for clearAllData)
 */
export function resetAll() {
    clearAllData();
}
