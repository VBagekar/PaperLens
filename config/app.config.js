/* =========================================================
   config/app.config.js — Application-level constants
   PaperLens v1.0.0

   NOTE: This file is informational. Values are embedded
   directly into each module for zero-dependency simplicity
   (no bundler required). Update both if you change a value.
   ========================================================= */

const APP_CONFIG = {

  meta: {
    name:    'PaperLens',
    version: '1.0.0',
    author:  'Your Team',
    license: 'MIT',
  },

  pdfjs: {
    // CDN URL for the PDF.js worker script.
    // This is the ONLY external resource loaded at runtime.
    // No user data is sent; this is just the JS engine file.
    workerSrc: 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js',
    version:   '3.11.174',
  },

  nlp: {
    // Number of top-ranked sentences to include per mode
    topSentences: {
      summary:  8,
      bullets:  8,
      eli5:     5,
      abstract: 6,
    },

    // Number of keywords to extract
    maxKeywords: 18,

    // Average words per minute for reading time estimate
    wordsPerMinute: 200,

    // Min/max character length for a valid sentence
    sentenceMinLength: 60,
    sentenceMaxLength: 450,
  },

  ui: {
    // Delay (ms) between pipeline steps for UX feel
    stepDelayMs: 300,

    // Auto-hide status message after copy (ms)
    copyStatusDurationMs: 2000,
  },

};
