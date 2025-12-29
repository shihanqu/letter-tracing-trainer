/**
 * Speech Module
 * Web Speech API wrapper for text-to-speech
 */

let synth = null;
let selectedVoice = null;
let initialized = false;

// Feedback phrases
const correctPhrases = [
    "Great job!",
    "Excellent!",
    "You got it!",
    "Perfect!",
    "Awesome!",
    "Wonderful!",
    "You're amazing!",
    "That's right!",
    "Super!",
    "Fantastic!"
];

const incorrectPhrases = [
    "That looked more like a {char}. Try again!",
    "Hmm, I saw a {char}. Let's try once more!",
    "Almost! That looked like {char}. Give it another go!",
    "I think that was a {char}. You can do it!",
    "That seemed like {char}. Let's practice again!"
];

/**
 * Initialize speech synthesis
 */
export function initSpeech() {
    if (initialized) return Promise.resolve(true);

    return new Promise((resolve) => {
        if (!('speechSynthesis' in window)) {
            console.warn('Speech synthesis not supported');
            resolve(false);
            return;
        }

        synth = window.speechSynthesis;

        // Load voices
        const loadVoices = () => {
            const voices = synth.getVoices();

            // Prefer local English voices for offline use
            selectedVoice = voices.find(v =>
                v.lang.startsWith('en') && v.localService
            ) || voices.find(v =>
                v.lang.startsWith('en')
            ) || voices[0];

            if (selectedVoice) {
                console.log('Selected voice:', selectedVoice.name);
                initialized = true;
                resolve(true);
            } else {
                console.warn('No suitable voice found');
                resolve(false);
            }
        };

        // Chrome loads voices async
        if (synth.getVoices().length > 0) {
            loadVoices();
        } else {
            synth.addEventListener('voiceschanged', loadVoices, { once: true });
            // Timeout in case voiceschanged never fires
            setTimeout(() => {
                if (!initialized) {
                    loadVoices();
                }
            }, 1000);
        }
    });
}

/**
 * Speak text
 */
export function speak(text, options = {}) {
    if (!synth || !initialized) {
        console.warn('Speech not initialized');
        return Promise.resolve();
    }

    return new Promise((resolve) => {
        // Cancel any ongoing speech
        synth.cancel();

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.voice = selectedVoice;
        utterance.rate = options.rate ?? 0.9; // Slightly slower for kids
        utterance.pitch = options.pitch ?? 1.1; // Slightly higher, friendlier
        utterance.volume = options.volume ?? 1.0;

        utterance.onend = () => resolve();
        utterance.onerror = (e) => {
            console.error('Speech error:', e);
            resolve();
        };

        synth.speak(utterance);
    });
}

/**
 * Speak a character name
 */
export function speakCharacter(char) {
    const charName = getCharacterName(char);
    return speak(charName);
}

/**
 * Get spoken name for a character
 */
function getCharacterName(char) {
    // Numbers
    if (/^\d$/.test(char)) {
        const numbers = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'];
        return numbers[parseInt(char)];
    }

    // Letters - just speak the letter
    return char.toUpperCase();
}

/**
 * Speak success feedback
 */
export function speakCorrect() {
    const phrase = correctPhrases[Math.floor(Math.random() * correctPhrases.length)];
    return speak(phrase);
}

/**
 * Speak incorrect feedback with predicted character
 */
export function speakIncorrect(predictedChar) {
    const template = incorrectPhrases[Math.floor(Math.random() * incorrectPhrases.length)];
    const phrase = template.replace('{char}', getCharacterName(predictedChar));
    return speak(phrase);
}

/**
 * Speak "Draw [character]"
 */
export function speakPrompt(char) {
    const charName = getCharacterName(char);
    return speak(`Draw ${charName}`);
}

/**
 * Stop speaking
 */
export function stopSpeaking() {
    if (synth) {
        synth.cancel();
    }
}

/**
 * Check if speech is supported
 */
export function isSpeechSupported() {
    return 'speechSynthesis' in window;
}
