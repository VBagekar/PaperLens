# PaperLens — Offline AI Research Assistant

> Upload a research paper PDF → get an instant summary. **No internet. No API key. No data leaves your device.**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [How to Run](#how-to-run)
4. [How It Works](#how-it-works)
5. [Output Modes](#output-modes)
6. [Technology Stack](#technology-stack)
7. [Customization Guide](#customization-guide)
8. [FAQ](#faq)

---

## Project Overview

PaperLens is a **100% browser-based** research paper simplifier. It uses:

- **PDF.js** to read PDF files entirely in-browser
- **Custom local NLP** (TF-IDF sentence scoring + keyword extraction) — no external AI model
- **Zero backend** — open the HTML file and it works

This means:
- ✅ Your papers are **never uploaded** anywhere
- ✅ Works **offline** after first load (fonts & PDF.js are the only CDN resources)
- ✅ No API keys, accounts, or subscriptions needed

---

## Project Structure

```
paperlens/
│
├── index.html                  ← Main HTML entry point (open this!)
│
├── src/
│   ├── css/
│   │   ├── reset.css           ← Browser normalization
│   │   ├── variables.css       ← Design tokens (colors, fonts, spacing)
│   │   ├── layout.css          ← Page structure & grid
│   │   ├── components.css      ← UI component styles
│   │   └── animations.css      ← Keyframes & motion utilities
│   │
│   └── js/
│       ├── app.js              ← Entry point; wires all modules together
│       └── modules/
│           ├── pdfExtractor.js     ← PDF text extraction via PDF.js
│           ├── nlpAnalyzer.js      ← TF-IDF scoring, keywords, sections
│           ├── outputGenerator.js  ← Renders results per output mode
│           └── uiController.js     ← All DOM interactions & UI state
│
├── config/
│   └── app.config.js           ← Centralized constants (informational)
│
└── docs/
    └── README.md               ← This file
```

---

## How to Run

### Option A — Just open the file (simplest)

1. Unzip `paperlens.zip`
2. Open `index.html` in any modern browser (Chrome, Firefox, Edge, Safari)
3. Done! No installation needed.

> ⚠️ **If you see a blank page or errors:**  
> Some browsers block local file access when using `file://`. Use Option B below.

---

### Option B — Local dev server (recommended for development)

**Using Python (built into macOS/Linux):**
```bash
cd paperlens
python3 -m http.server 8080
```
Then open: [http://localhost:8080](http://localhost:8080)

**Using Node.js:**
```bash
cd paperlens
npx serve .
```
Then open the URL shown in your terminal.

**Using VS Code:**
- Install the **Live Server** extension
- Right-click `index.html` → **Open with Live Server**

---

## How It Works

```
┌─────────────────┐
│  User uploads   │
│     PDF file    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  pdfExtractor.js            │
│  PDF.js reads file locally  │
│  → extracts raw text        │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  nlpAnalyzer.js             │
│  - Tokenizes text           │
│  - Computes TF-IDF scores   │
│  - Ranks sentences          │
│  - Extracts top keywords    │
│  - Detects section headers  │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  outputGenerator.js         │
│  - Picks top-N sentences    │
│  - Formats per chosen mode  │
│  - Returns HTML string      │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  uiController.js            │
│  - Renders results to DOM   │
│  - Updates stats, keywords  │
│  - Manages progress/steps   │
└─────────────────────────────┘
```

---

## Output Modes

| Mode | Description |
|---|---|
| **Plain-language summary** | 8 top sentences re-ordered naturally for readability |
| **Key bullet points** | Same sentences formatted as a bulleted list |
| **Explain like I'm 5** | Simplified vocabulary substitution on top sentences |
| **Technical abstract rewrite** | 6 top sentences formatted as a compact academic abstract |

---

## Technology Stack

| Technology | Purpose | External? |
|---|---|---|
| **PDF.js 3.11** | In-browser PDF text extraction | CDN (no data sent) |
| **Google Fonts** | Playfair Display + JetBrains Mono | CDN (no data sent) |
| **Vanilla JS** | All NLP + UI logic | Local only |
| **CSS Custom Properties** | Design token system | Local only |

No frameworks. No build tools. No bundlers. Just files.

---

## Customization Guide

### Change colors / fonts
Edit `src/css/variables.css`. All design tokens are in one place:
```css
:root {
  --color-accent: #c1440e;      /* Change to your brand color */
  --font-display: 'Playfair Display', serif;
}
```

### Add more output modes
1. Add an `<option>` to the `<select>` in `index.html`
2. Add a case to the `switch` in `outputGenerator.js`
3. Add a config entry to `MODE_CONFIG` in `outputGenerator.js`

### Adjust keyword count or sentence count
Edit `config/app.config.js` and the matching constants in `nlpAnalyzer.js` and `outputGenerator.js`.

### Add more stopwords
Edit the `STOPWORDS` Set in `nlpAnalyzer.js`.

---

## FAQ

**Q: Does this send my PDF to any server?**  
A: No. All file reading and text processing happens in your browser. Check the Network tab in DevTools — you'll see no POST requests.

**Q: Why does it need an internet connection at all?**  
A: Only for loading Google Fonts and PDF.js from a CDN on first visit. You can make it fully offline by downloading those files and updating the `<link>` / `<script>` tags to point to local copies.

**Q: Can I use this for very large PDFs (100+ pages)?**  
A: Yes, but extraction will take a few seconds. The progress bar keeps you updated.

**Q: The summary doesn't seem accurate — why?**  
A: PaperLens uses extractive summarization (picks real sentences from the paper). It does not use a neural network. For maximum accuracy, consider integrating [Transformers.js](https://huggingface.co/docs/transformers.js) with a local ONNX model.

**Q: Can I integrate a real AI model?**  
A: Yes. Replace the `NlpAnalyzer` module with a call to `Transformers.js` pipeline. The model runs in the browser via WebAssembly — still 100% local.

---

*PaperLens v1.0.0 — Built with ♥ using vanilla HTML, CSS, and JavaScript.*
