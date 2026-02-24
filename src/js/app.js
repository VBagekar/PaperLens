/* =========================================================
   app.js — Application entry point & event orchestration
   Wires together: PdfExtractor, NlpAnalyzer,
                   OutputGenerator, UiController
   PaperLens v1.0.0
   ========================================================= */

'use strict';

/**
 * @module App
 * Top-level application controller. Binds DOM events and
 * orchestrates the analysis pipeline.
 */
const App = (() => {

  // ── App state ─────────────────────────────────────────────────────────────
  let currentFile     = null;   // File object from file picker / drop
  let extractedText   = '';     // Raw text extracted from PDF
  let extractedPages  = 0;      // Number of PDF pages

  // ── Utility ───────────────────────────────────────────────────────────────
  const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

  // ── File handling ─────────────────────────────────────────────────────────

  /**
   * Validate and start processing a dropped or selected PDF file.
   * @param {File} file
   */
  async function handleFile(file) {
    if (!file || file.type !== 'application/pdf') {
      UiController.setStatus('Please upload a valid PDF file.', 'error');
      return;
    }

    currentFile = file;
    UiController.setAnalyzeBtnEnabled(false);
    UiController.hideResults();
    UiController.showFilename(file.name);
    UiController.setStatus(
      `Loading: ${file.name} (${(file.size / 1024).toFixed(0)} KB)…`
    );

    try {
      const { text, pageCount } = await PdfExtractor.extract(file);
      extractedText  = text;
      extractedPages = pageCount;

      const wordCount = text.split(/\s+/).filter(Boolean).length;

      UiController.setStatus(
        `✓ PDF loaded — ${pageCount} pages, ~${wordCount.toLocaleString()} words. Ready to analyze.`,
        'ready'
      );
      UiController.setAnalyzeBtnEnabled(true);

    } catch (err) {
      console.error('[PaperLens] PDF extraction error:', err);
      UiController.setStatus(`Error reading PDF: ${err.message}`, 'error');
    }
  }

  // ── Analysis pipeline ─────────────────────────────────────────────────────

  /** Run the full analysis pipeline when the user clicks Analyze. */
  async function runAnalysis() {
    if (!extractedText) return;

    const mode = UiController.getSelectedMode();

    UiController.setAnalyzeBtnEnabled(false);
    UiController.hideResults();
    UiController.resetSteps();
    UiController.setProgressVisible(true);

    // ── Step 1: Extract ───────────────────────────────────────────────────
    UiController.setStep('step-extract', 'active');
    UiController.setProgress(10, 'Reviewing extracted content…');
    await delay(300);

    UiController.setStep('step-extract', 'done');

    // ── Step 2: Chunk & analyze ───────────────────────────────────────────
    UiController.setStep('step-chunk', 'active');
    UiController.setProgress(35, 'Chunking and scoring sentences…');

    // Run NLP analysis (synchronous but potentially heavy — small delay gives
    // the browser a chance to paint the progress update)
    await delay(50);
    const analysis = NlpAnalyzer.analyze(extractedText);
    await delay(300);

    UiController.setStep('step-chunk', 'done');

    // ── Step 3: Generate output ───────────────────────────────────────────
    UiController.setStep('step-analyze', 'active');
    UiController.setProgress(65, 'Generating output…');
    await delay(400);

    const { title: outputTitle, html: outputHtml } = OutputGenerator.generate(analysis, mode);

    UiController.setStep('step-analyze', 'done');

    // ── Step 4: Format ────────────────────────────────────────────────────
    UiController.setStep('step-format', 'active');
    UiController.setProgress(90, 'Formatting results…');
    await delay(300);

    UiController.setStep('step-format', 'done');
    UiController.setProgress(100, 'Done!');
    await delay(300);

    // ── Render results ────────────────────────────────────────────────────
    UiController.setProgressVisible(false);
    UiController.showResults({
      pageCount:   extractedPages,
      wordCount:   analysis.wordCount,
      readTime:    analysis.readTime,
      outputTitle,
      outputHtml,
      keywords:    analysis.keywords,
      sections:    analysis.sections,
    });

    UiController.setStatus(
      'Analysis complete — all processing was performed locally on your device.',
      'ready'
    );
    UiController.setAnalyzeBtnEnabled(true);
  }

  // ── Event binding ─────────────────────────────────────────────────────────

  function bindEvents() {
    const dropZone   = UiController.getDropZone();
    const fileInput  = UiController.getFileInput();
    const analyzeBtn = UiController.getAnalyzeBtn();

    // Drop zone — click
    dropZone.addEventListener('click', () => fileInput.click());

    // Drop zone — keyboard (accessibility)
    dropZone.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') fileInput.click();
    });

    // Drop zone — drag & drop
    dropZone.addEventListener('dragover', e => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
      dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      const file = e.dataTransfer.files[0];
      handleFile(file);
    });

    // File input change
    fileInput.addEventListener('change', () => {
      if (fileInput.files[0]) handleFile(fileInput.files[0]);
    });

    // Analyze button
    analyzeBtn.addEventListener('click', runAnalysis);

    // Copy buttons (delegated)
    document.addEventListener('click', e => {
      const btn = e.target.closest('.copy-btn');
      if (btn) {
        const target = btn.dataset.target;
        if (target) UiController.copyElementText(target);
      }
    });
  }

  // ── Init ──────────────────────────────────────────────────────────────────

  function init() {
    bindEvents();
    console.info('[PaperLens] Initialized — v3.0.0 — all processing is local.');
  }

  // Auto-init when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  return { init };

})();
