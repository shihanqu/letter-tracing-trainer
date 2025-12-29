/**
 * Achievements Module
 * Badge definitions and unlock logic
 */

import * as storage from './storage.js';

// Achievement definitions
export const ACHIEVEMENTS = {
    firstSteps: {
        id: 'firstSteps',
        name: 'First Steps',
        description: 'Complete your first character',
        icon: '🌟',
        condition: (progress) => progress.correctAttempts >= 1
    },
    onFire: {
        id: 'onFire',
        name: 'On Fire!',
        description: 'Get a 5 streak',
        icon: '🔥',
        condition: (progress) => progress.bestStreak >= 5
    },
    lightning: {
        id: 'lightning',
        name: 'Lightning Fast',
        description: 'Get a 10 streak',
        icon: '⚡',
        condition: (progress) => progress.bestStreak >= 10
    },
    numberNinja: {
        id: 'numberNinja',
        name: 'Number Ninja',
        description: 'Master all digits 0-9',
        icon: '🔢',
        condition: (progress, charStats) => {
            const digits = '0123456789'.split('');
            return digits.every(d => (charStats[d]?.correct ?? 0) >= 3);
        }
    },
    letterLegend: {
        id: 'letterLegend',
        name: 'Letter Legend',
        description: 'Master A-Z letters',
        icon: '🔤',
        condition: (progress, charStats) => {
            const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
            return letters.every(l => (charStats[l]?.correct ?? 0) >= 3);
        }
    },
    grandMaster: {
        id: 'grandMaster',
        name: 'Grand Master',
        description: 'Master all 47 characters',
        icon: '🏆',
        condition: (progress, charStats) => {
            const allChars = getAllCharacters();
            return allChars.every(c => (charStats[c]?.correct ?? 0) >= 3);
        }
    },
    persistent: {
        id: 'persistent',
        name: 'Never Give Up',
        description: 'Attempt 100 characters',
        icon: '💪',
        condition: (progress) => progress.totalAttempts >= 100
    },
    perfectTen: {
        id: 'perfectTen',
        name: 'Perfect 10',
        description: 'Get 10 correct in a row',
        icon: '🎯',
        condition: (progress) => progress.bestStreak >= 10
    }
};

/**
 * Get all EMNIST Balanced characters
 */
function getAllCharacters() {
    return [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e',
        'f', 'g', 'h', 'n', 'q', 'r', 't'
    ];
}

/**
 * Check for newly unlocked achievements
 * Returns array of newly unlocked achievement IDs
 */
export function checkAchievements() {
    const unlockedIds = storage.getAchievements();
    const progress = storage.getProgress();
    const charStats = storage.getCharacterStats();
    const newlyUnlocked = [];

    for (const [id, achievement] of Object.entries(ACHIEVEMENTS)) {
        if (!unlockedIds.includes(id)) {
            if (achievement.condition(progress, charStats)) {
                const isNew = storage.addAchievement(id);
                if (isNew) {
                    newlyUnlocked.push(achievement);
                }
            }
        }
    }

    return newlyUnlocked;
}

/**
 * Get all achievements with unlock status
 */
export function getAllAchievements() {
    const unlockedIds = storage.getAchievements();

    return Object.values(ACHIEVEMENTS).map(achievement => ({
        ...achievement,
        unlocked: unlockedIds.includes(achievement.id)
    }));
}

/**
 * Get streak status and update
 */
export function handleStreak(correct) {
    const progress = storage.getProgress();
    let { currentStreak, bestStreak } = progress;

    if (correct) {
        currentStreak++;
        if (currentStreak > bestStreak) {
            bestStreak = currentStreak;
        }
    } else {
        currentStreak = 0;
    }

    storage.updateProgress({ currentStreak, bestStreak });

    return { currentStreak, bestStreak, streakBroken: !correct && progress.currentStreak > 0 };
}

/**
 * Record an attempt
 */
export function recordAttempt(character, correct) {
    // Update progress
    const progress = storage.getProgress();
    storage.updateProgress({
        totalAttempts: progress.totalAttempts + 1,
        correctAttempts: progress.correctAttempts + (correct ? 1 : 0),
        score: progress.score + (correct ? 10 : 0)
    });

    // Update character stats
    storage.updateCharacterStat(character, correct);

    // Update streak
    const streakResult = handleStreak(correct);

    // Check for new achievements
    const newAchievements = checkAchievements();

    return {
        streakResult,
        newAchievements
    };
}

/**
 * Get current streak
 */
export function getCurrentStreak() {
    return storage.getProgress().currentStreak;
}

/**
 * Get best streak
 */
export function getBestStreak() {
    return storage.getProgress().bestStreak;
}
