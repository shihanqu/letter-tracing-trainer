/**
 * Dashboard Module
 * Parent dashboard with progress stats and character mastery
 */

import * as storage from './storage.js';
import { getAllAchievements, ACHIEVEMENTS } from './achievements.js';

/**
 * Initialize dashboard
 */
export function initDashboard() {
    updateDashboardStats();
    renderBadges();
    renderCharacterMastery();
}

/**
 * Update dashboard statistics
 */
export function updateDashboardStats() {
    const progress = storage.getProgress();

    // Total attempts
    const totalAttemptsEl = document.getElementById('total-attempts');
    if (totalAttemptsEl) {
        totalAttemptsEl.textContent = progress.totalAttempts;
    }

    // Accuracy rate
    const accuracyEl = document.getElementById('accuracy-rate');
    if (accuracyEl) {
        const accuracy = progress.totalAttempts > 0
            ? Math.round((progress.correctAttempts / progress.totalAttempts) * 100)
            : 0;
        accuracyEl.textContent = accuracy + '%';
    }

    // Best streak
    const bestStreakEl = document.getElementById('best-streak');
    if (bestStreakEl) {
        bestStreakEl.textContent = progress.bestStreak;
    }
}

/**
 * Render achievement badges
 */
export function renderBadges() {
    const container = document.getElementById('badges-container');
    if (!container) return;

    const achievements = getAllAchievements();

    container.innerHTML = achievements.map((achievement, index) => `
    <div class="badge-item ${achievement.unlocked ? 'unlocked' : 'locked'}" 
         data-achievement-index="${index}">
      <div class="badge ${achievement.unlocked ? 'badge-unlocked' : 'badge-locked'}">
        ${achievement.icon}
      </div>
      <span class="badge-name">${achievement.name}</span>
    </div>
  `).join('');

    // Add click handlers
    container.querySelectorAll('.badge-item').forEach((item, index) => {
        item.addEventListener('click', () => {
            showAchievementDetails(achievements[index]);
        });
        item.style.cursor = 'pointer';
    });
}

/**
 * Show achievement details popup
 */
function showAchievementDetails(achievement) {
    // Create modal
    const backdrop = document.createElement('div');
    backdrop.className = 'modal-backdrop visible';
    backdrop.style.zIndex = '9999';

    const modal = document.createElement('div');
    modal.className = 'achievement-detail-modal';
    modal.innerHTML = `
        <div class="badge ${achievement.unlocked ? 'badge-unlocked' : 'badge-locked'}" 
             style="width: 80px; height: 80px; font-size: 40px;">
            ${achievement.icon}
        </div>
        <h3 class="achievement-detail-name">${achievement.name}</h3>
        <p class="achievement-detail-desc">${achievement.description}</p>
        <p class="achievement-detail-status">
            ${achievement.unlocked
            ? '✅ <strong>Unlocked!</strong>'
            : '🔒 <em>Not yet unlocked</em>'}
        </p>
        <button class="btn btn-primary">OK</button>
    `;

    backdrop.appendChild(modal);
    document.body.appendChild(backdrop);

    // Dismiss handlers
    const dismiss = () => backdrop.remove();
    modal.querySelector('button').addEventListener('click', dismiss);
    backdrop.addEventListener('click', (e) => {
        if (e.target === backdrop) dismiss();
    });
}

/**
 * Render character mastery grid
 */
export function renderCharacterMastery() {
    const container = document.getElementById('character-mastery');
    if (!container) return;

    const charStats = storage.getCharacterStats();

    // All characters in order
    const allChars = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z'
    ];

    container.innerHTML = allChars.map(char => {
        const stats = charStats[char];
        let masteryClass = 'not-started';
        let title = 'Not practiced yet';

        if (stats && stats.attempts > 0) {
            const accuracy = (stats.correct / stats.attempts) * 100;
            if (accuracy >= 80) {
                masteryClass = 'mastered';
                title = `Mastered! ${Math.round(accuracy)}% accuracy`;
            } else if (accuracy >= 50) {
                masteryClass = 'learning';
                title = `Learning: ${Math.round(accuracy)}% accuracy`;
            } else {
                masteryClass = 'struggling';
                title = `Needs practice: ${Math.round(accuracy)}% accuracy`;
            }
            title += ` (${stats.correct}/${stats.attempts})`;
        }

        return `
      <div class="character-stat ${masteryClass}" title="${title}">
        ${char}
      </div>
    `;
    }).join('');
}

/**
 * Show badge unlock animation
 * @param {object} achievement - The achievement object
 * @param {function} pauseTimer - Optional callback to pause game timer
 * @param {function} resumeTimer - Optional callback to resume game timer
 */
export function showBadgeUnlock(achievement, pauseTimer = null, resumeTimer = null) {
    // Pause the timer if callback provided
    if (pauseTimer) {
        pauseTimer();
    }

    // Create modal backdrop for centering
    const backdrop = document.createElement('div');
    backdrop.className = 'modal-backdrop visible';
    backdrop.style.zIndex = '10000';

    const overlay = document.createElement('div');
    overlay.className = 'badge-unlock-modal';
    overlay.innerHTML = `
    <div class="badge badge-unlocked" style="width: 100px; height: 100px; font-size: 50px;">
      ${achievement.icon}
    </div>
    <div class="badge-unlock-text">
      🎉 Achievement Unlocked! 🎉<br>
      <strong>${achievement.name}</strong><br>
      <small>${achievement.description}</small>
    </div>
    <button class="btn btn-primary" id="badge-dismiss-btn">Continue</button>
  `;

    backdrop.appendChild(overlay);
    document.body.appendChild(backdrop);

    // Handle dismiss
    const dismissBtn = overlay.querySelector('#badge-dismiss-btn');
    const dismiss = () => {
        backdrop.remove();
        renderBadges(); // Refresh badges display
        // Resume the timer if callback provided
        if (resumeTimer) {
            resumeTimer();
        }
    };

    dismissBtn?.addEventListener('click', dismiss);

    // Also dismiss on backdrop click
    backdrop.addEventListener('click', (e) => {
        if (e.target === backdrop) {
            dismiss();
        }
    });

    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (document.body.contains(backdrop)) {
            dismiss();
        }
    }, 5000);
}

/**
 * Get summary statistics
 */
export function getSummaryStats() {
    const progress = storage.getProgress();
    const charStats = storage.getCharacterStats();
    const achievements = getAllAchievements();

    const masteredCount = Object.values(charStats).filter(s =>
        s.attempts > 0 && (s.correct / s.attempts) >= 0.8
    ).length;

    const unlockedCount = achievements.filter(a => a.unlocked).length;

    return {
        totalAttempts: progress.totalAttempts,
        correctAttempts: progress.correctAttempts,
        accuracy: progress.totalAttempts > 0
            ? Math.round((progress.correctAttempts / progress.totalAttempts) * 100)
            : 0,
        currentStreak: progress.currentStreak,
        bestStreak: progress.bestStreak,
        score: progress.score,
        masteredCharacters: masteredCount,
        totalCharacters: 36, // 0-9 + A-Z
        unlockedAchievements: unlockedCount,
        totalAchievements: Object.keys(ACHIEVEMENTS).length
    };
}

/**
 * Export progress report as text
 */
export function exportProgressReport() {
    const stats = getSummaryStats();
    const charStats = storage.getCharacterStats();

    let report = '📊 Letter Tracing Progress Report\n';
    report += '================================\n\n';
    report += `📝 Total Practice Sessions: ${stats.totalAttempts}\n`;
    report += `✅ Correct Answers: ${stats.correctAttempts}\n`;
    report += `📈 Overall Accuracy: ${stats.accuracy}%\n`;
    report += `🔥 Best Streak: ${stats.bestStreak}\n`;
    report += `⭐ Total Score: ${stats.score}\n\n`;
    report += `🏆 Achievements: ${stats.unlockedAchievements}/${stats.totalAchievements}\n`;
    report += `📚 Characters Mastered: ${stats.masteredCharacters}/${stats.totalCharacters}\n\n`;

    report += 'Character Breakdown:\n';
    report += '-------------------\n';

    const allChars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
    for (const char of allChars) {
        const s = charStats[char];
        if (s && s.attempts > 0) {
            const acc = Math.round((s.correct / s.attempts) * 100);
            const status = acc >= 80 ? '✅' : acc >= 50 ? '📝' : '❌';
            report += `${status} ${char}: ${s.correct}/${s.attempts} (${acc}%)\n`;
        }
    }

    return report;
}
