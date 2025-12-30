# Letter Tracing Fun! ✏️

> **🌐 [Try the Live App](https://shihanqu.github.io/letter-tracing-trainer/)** 

A delightful, AI-powered tracing app to help kids learn to write letters and numbers. Features real-time handwriting recognition using machine learning, right in the browser!

## ✨ Features

### 🎨 Interactive Drawing
- Smooth, responsive canvas for drawing
- Touch support for tablets and phones
- Trace guides to help learn letter shapes
- Mobile-first responsive design

### 🤖 AI-Powered Recognition
- Real-time handwriting recognition using ONNX Runtime
- Trained on EMNIST Balanced dataset (47 characters)
- Works completely offline - no server required!
- WebGPU acceleration with WASM fallback

### 🎮 Gamification
- Streak tracking with fire animations
- Achievement badges to unlock
- Score system to encourage practice
- Encouraging audio feedback

### 👪 Parent Dashboard
- Track progress across all characters
- View accuracy statistics
- See mastery levels per character
- Reset all progress option

### ⚙️ Customizable Settings
- **Character Ranges**: 1-10, 0-99, A-Z, a-z, or All
- **Display Modes**: See & Hear, See Only, Hear Only
- **Timer Options**: 10s, 20s, 30s, or unlimited
- Visual timer that depletes around the drawing border

## 🚀 Quick Start

### Option 1: GitHub Pages (Recommended)

1. Fork this repository
2. Go to Settings → Pages
3. Set source to "Deploy from a branch"
4. Select `main` branch and `/ (root)` folder
5. Click Save
6. Your app will be live at `https://YOUR_USERNAME.github.io/letter-tracing-fun/`

### Option 2: Local Development

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/letter-tracing-fun.git
cd letter-tracing-fun

# Start a local server (Python 3)
python3 -m http.server 8000

# Or with Node.js
npx serve .
```

Then open http://localhost:8000 in your browser.

## 📱 Browser Support

| Browser | Support | Notes |
|---------|---------|-------|
| Chrome 113+ | ✅ Full | WebGPU acceleration |
| Edge 113+ | ✅ Full | WebGPU acceleration |
| Firefox 120+ | ✅ Full | WASM backend |
| Safari 17+ | ✅ Full | WASM backend |
| Mobile Chrome | ✅ Full | Touch optimized |
| Mobile Safari | ✅ Full | Touch optimized |

## 🧠 Model Training

The included model was trained on EMNIST Balanced (47 classes) to ~89% accuracy. To retrain:

```bash
# Install dependencies
pip install torch torchvision onnx

# Train the model
python scripts/train_model.py

# Export to ONNX
python scripts/export_onnx.py
```

## 🔧 Developer Tools

### Debug Mode
Click the 🔧 button to access debug mode:
- View the preprocessed 28×28 image sent to the neural network
- See the top 10 prediction probabilities
- Helpful for understanding recognition issues

### Path Viewer
Access `tools/view-paths.html` to:
- View all 47 character SVG trace templates
- Edit paths with live preview
- Export updated template code

## 📁 Project Structure

```
letter-tracing-fun/
├── index.html          # Main app entry
├── css/
│   ├── variables.css   # CSS custom properties
│   ├── layout.css      # Responsive layout
│   ├── components.css  # UI components
│   └── animations.css  # Animations & effects
├── js/
│   ├── main.js         # App coordinator
│   ├── canvas.js       # Drawing & preprocessing
│   ├── inference.js    # ONNX Runtime wrapper
│   ├── game.js         # Game logic
│   ├── templates.js    # SVG trace paths
│   ├── speech.js       # Text-to-speech
│   ├── storage.js      # LocalStorage persistence
│   ├── achievements.js # Badge system
│   └── dashboard.js    # Parent dashboard
├── models/
│   ├── emnist_balanced.onnx  # Trained ML model
│   └── labels.json           # Character mappings
├── lib/
│   ├── ort.min.js      # ONNX Runtime
│   └── ort-wasm-*.wasm # WASM backends
├── tools/
│   └── view-paths.html # Path editor tool
└── scripts/
    ├── train_model.py  # Model training
    └── export_onnx.py  # ONNX export
```

## 🎯 Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Enter | Check drawing |
| Escape | Clear canvas |
| Space | Skip to next |

## 📄 License

MIT License - feel free to use, modify, and share!

---

Made with ❤️ for little learners
