/**
 * Templates Module
 * SVG trace templates for characters
 */

// SVG path data for each character
// These are simplified, kid-friendly strokes
const TEMPLATES = {
    // Numbers
    '0': { path: 'M50,15 C75,15 85,35 85,50 C85,65 75,85 50,85 C25,85 15,65 15,50 C15,35 25,15 50,15 Z', strokes: 1 },
    '1': { path: 'M35,25 L50,15 L50,85', strokes: 1 },
    '2': { path: 'M20,30 C20,15 40,10 55,15 C75,22 75,40 50,55 L15,85 L85,85', strokes: 1 },
    '3': { path: 'M20,20 C35,10 65,10 75,25 C85,38 70,48 50,50 C70,52 85,62 75,78 C65,92 30,92 15,78', strokes: 1 },
    '4': { path: 'M60,85 L60,15 L15,60 L85,60', strokes: 2 },
    '5': { path: 'M75,15 L25,15 L20,45 C35,38 60,38 75,50 C90,65 75,88 45,88 C25,88 15,78 15,70', strokes: 1 },
    '6': { path: 'M62,10 C50,10 40,25 32,45 C25,62 28,80 45,88 C62,95 80,85 82,68 C84,52 72,42 55,42 C40,42 32,55 32,55', strokes: 1 },
    '7': { path: 'M15,15 L85,15 L45,85', strokes: 1 },
    '8': { path: 'M50,50 C30,50 20,38 25,25 C30,12 45,8 50,8 C55,8 70,12 75,25 C80,38 70,50 50,50 C25,50 15,65 20,78 C28,92 72,92 80,78 C85,65 75,50 50,50', strokes: 1 },
    '9': { path: 'M70,35 C70,20 55,12 40,15 C25,18 18,30 20,45 C22,58 35,65 50,62 C65,60 72,48 70,35 L70,35 L65,85', strokes: 1 },

    // Uppercase Letters
    'A': { path: 'M10,85 L50,15 L90,85 M25,60 L75,60', strokes: 2 },
    'B': { path: 'M20,85 L20,15 L55,15 C75,15 80,22 80,32 C80,42 70,48 55,48 L20,48 L55,48 C75,48 85,55 85,68 C85,82 70,85 55,85 L20,85', strokes: 1 },
    'C': { path: 'M80,30 C65,10 30,10 18,30 C5,55 15,80 35,88 C55,95 75,85 85,70', strokes: 1 },
    'D': { path: 'M20,15 L20,85 L50,85 C80,85 92,65 92,50 C92,35 80,15 50,15 L20,15', strokes: 1 },
    'E': { path: 'M75,15 L20,15 L20,85 L75,85 M20,50 L60,50', strokes: 2 },
    'F': { path: 'M75,15 L20,15 L20,85 M20,50 L60,50', strokes: 2 },
    'G': { path: 'M80,30 C65,10 30,10 18,30 C5,55 15,80 35,88 C55,95 80,85 82,65 L55,65', strokes: 1 },
    'H': { path: 'M20,15 L20,85 M80,15 L80,85 M20,50 L80,50', strokes: 3 },
    'I': { path: 'M30,15 L70,15 M50,15 L50,85 M30,85 L70,85', strokes: 3 },
    'J': { path: 'M30,15 L70,15 M50,15 L50,70 C50,85 35,88 22,78', strokes: 2 },
    'K': { path: 'M20,15 L20,85 M75,15 L20,50 L75,85', strokes: 2 },
    'L': { path: 'M20,15 L20,85 L75,85', strokes: 1 },
    'M': { path: 'M15,85 L15,15 L50,55 L85,15 L85,85', strokes: 1 },
    'N': { path: 'M20,85 L20,15 L80,85 L80,15', strokes: 1 },
    'O': { path: 'M50,15 C80,15 90,35 90,50 C90,65 80,85 50,85 C20,85 10,65 10,50 C10,35 20,15 50,15', strokes: 1 },
    'P': { path: 'M20,85 L20,15 L60,15 C85,15 88,35 75,48 C65,58 40,55 20,50', strokes: 1 },
    'Q': { path: 'M50,15 C80,15 90,35 90,50 C90,65 80,85 50,85 C20,85 10,65 10,50 C10,35 20,15 50,15 M65,70 L85,90', strokes: 2 },
    'R': { path: 'M20,85 L20,15 L60,15 C85,15 88,35 75,48 C65,58 40,55 20,50 M50,50 L80,85', strokes: 2 },
    'S': { path: 'M75,25 C70,12 50,8 35,12 C15,18 12,35 30,48 C55,62 80,65 80,80 C80,95 50,95 30,88 C18,82 12,72 15,62', strokes: 1 },
    'T': { path: 'M15,15 L85,15 M50,15 L50,85', strokes: 2 },
    'U': { path: 'M20,15 L20,65 C20,85 35,90 50,90 C65,90 80,85 80,65 L80,15', strokes: 1 },
    'V': { path: 'M15,15 L50,85 L85,15', strokes: 1 },
    'W': { path: 'M10,15 L30,85 L50,35 L70,85 L90,15', strokes: 1 },
    'X': { path: 'M15,15 L85,85 M85,15 L15,85', strokes: 2 },
    'Y': { path: 'M15,15 L50,50 L85,15 M50,50 L50,85', strokes: 2 },
    'Z': { path: 'M15,15 L85,15 L15,85 L85,85', strokes: 1 },

    // Lowercase letters (for those in EMNIST Balanced that differ from uppercase)
    'a': { path: 'M65,35 C55,25 35,25 28,40 C20,55 25,72 40,78 C55,84 70,75 72,60 L72,35 L72,85', strokes: 1 },
    'b': { path: 'M25,15 L25,85 M25,50 C35,38 55,35 68,45 C82,58 80,75 65,82 C50,88 30,82 25,70', strokes: 1 },
    'd': { path: 'M75,15 L75,85 M75,50 C65,38 45,35 32,45 C18,58 20,75 35,82 C50,88 70,82 75,70', strokes: 1 },
    'e': { path: 'M20,50 C20,35 35,25 50,25 C68,25 78,38 78,50 L20,50 C20,68 32,82 52,82 C68,82 78,75 80,65', strokes: 1 },
    'f': { path: 'M70,20 C60,10 45,12 40,25 L40,85 M25,50 L60,50', strokes: 2 },
    'g': { path: 'M68,48 C68,35 56,27 45,27 C32,27 22,38 22,50 C22,62 32,72 46,72 C60,72 68,62 68,50 L68,27 L68,82 C68,92 55,97 40,92', strokes: 1 },
    'h': { path: 'M25,15 L25,85 M25,50 C40,38 55,38 65,50 C75,62 72,85 72,85', strokes: 1 },
    'n': { path: 'M25,35 L25,85 M25,55 C30,40 45,35 55,38 C68,42 72,55 72,65 L72,85', strokes: 1 },
    'q': { path: 'M70,35 C60,25 40,25 30,40 C22,55 28,70 45,75 C60,78 72,70 72,55 L72,35 L72,95', strokes: 1 },
    'r': { path: 'M28,35 L28,85 M28,55 C38,40 55,38 72,45', strokes: 1 },
    't': { path: 'M50,15 L50,75 C50,88 65,88 75,82 M30,35 L70,35', strokes: 2 }
};

/**
 * Get SVG path for a character
 */
export function getTemplatePath(char) {
    const template = TEMPLATES[char] || TEMPLATES[char.toUpperCase()];
    return template ? template.path : null;
}

/**
 * Get stroke count for a character
 */
export function getStrokeCount(char) {
    const template = TEMPLATES[char] || TEMPLATES[char.toUpperCase()];
    return template ? template.strokes : 1;
}

/**
 * Create SVG element for trace overlay
 */
export function createTraceSVG(char, width, height) {
    const path = getTemplatePath(char);
    if (!path) return null;

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', width);
    svg.setAttribute('height', height);
    svg.setAttribute('viewBox', '0 0 100 100');
    svg.style.position = 'absolute';
    svg.style.top = '0';
    svg.style.left = '0';
    svg.style.pointerEvents = 'none';

    const pathEl = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    pathEl.setAttribute('d', path);
    pathEl.setAttribute('fill', 'none');
    pathEl.setAttribute('stroke', 'rgba(255, 255, 255, 0.25)');
    pathEl.setAttribute('stroke-width', '8');
    pathEl.setAttribute('stroke-linecap', 'round');
    pathEl.setAttribute('stroke-linejoin', 'round');
    pathEl.setAttribute('stroke-dasharray', '12 8');

    svg.appendChild(pathEl);

    return svg;
}

/**
 * Draw trace on canvas context
 */
export function drawTraceOnCanvas(ctx, char, width, height) {
    const path = getTemplatePath(char);
    if (!path) return;

    const scale = width / 100;

    ctx.save();
    ctx.scale(scale, scale);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
    ctx.lineWidth = 8 / scale;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.setLineDash([12, 8]);

    const path2D = new Path2D(path);
    ctx.stroke(path2D);

    ctx.restore();
}

/**
 * Check if a character has a template
 */
export function hasTemplate(char) {
    return !!(TEMPLATES[char] || TEMPLATES[char.toUpperCase()]);
}

/**
 * Get all available characters with templates
 */
export function getAvailableCharacters() {
    return Object.keys(TEMPLATES);
}
