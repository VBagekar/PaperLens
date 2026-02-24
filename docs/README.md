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
