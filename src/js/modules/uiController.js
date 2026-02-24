/* =========================================================
   uiController.js — DOM manipulation & UI state management
   Module: All direct DOM interactions are handled here.
           No business logic — only UI rendering & events.
   PaperLens v1.0.0
   ========================================================= */

'use strict';

/**
 * @module UiController
 * Manages all UI state changes: status messages, progress bar,
 * step indicators, file display, and results rendering.
 */
const UiController = (() => {

  // ── Element references ────────────────────────────────────────────────────
  const els = {
    statusBar:       () => document.getElementById('status-bar'),
    progressContainer: () => document.getElementById('progress-container'),
    progressFill:    () => document.getElementById('progress-fill'),
    progressText:    () => document.getElementById('progress-text'),
    progressPct:     () => document.getElementById('progress-pct'),
    analyzeBtn:      () => document.getElementById('analyze-btn'),
    fileInput:       () => document.getElementById('file-input'),
    dropZone:        () => document.getElementById('drop-zone'),
    filenameDisplay: () => document.getElementById('filename-display'),
    modeSelect:      () => document.getElementById('mode-select'),
    results:         () => document.getElementById('results'),
    outputTitle:     () => document.getElementById('output-title'),
    outputContent:   () => document.getElementById('output-content'),
    keywordsContainer: () => document.getElementById('keywords-container'),
    sectionsList:    () => document.getElementById('sections-list'),
    statPages:       () => document.getElementById('stat-pages'),
    statWords:       () => document.getElementById('stat-words'),
    statRead:        () => document.getElementById('stat-read'),
  };

  // ── Step IDs ──────────────────────────────────────────────────────────────
  const STEPS = ['step-extract', 'step-chunk', 'step-analyze', 'step-format'];

  // ── Status Bar ────────────────────────────────────────────────────────────

  /**
   * Display a status message. Pass empty string to hide.
   * @param {string} message
   * @param {'ready'|'error'|''} [type='']
   */
  function setStatus(message, type = '') {
    const bar = els.statusBar();
    if (!message) {
      bar.className = '';
      bar.textContent = '';
      return;
    }
    bar.textContent = message;
    bar.className = `visible ${type}`.trim();
  }

  // ── Progress ──────────────────────────────────────────────────────────────

  /**
   * Show or hide the progress container.
   * @param {boolean} visible
   */
  function setProgressVisible(visible) {
    const container = els.progressContainer();
    container.classList.toggle('visible', visible);
    if (!visible) setProgress(0, '');
  }

  /**
   * Update the progress bar and label.
   * @param {number} percent  - 0 to 100
   * @param {string} label
   */
  function setProgress(percent, label) {
    els.progressFill().style.width = `${percent}%`;
    els.progressPct().textContent = `${percent}%`;
    if (label) els.progressText().textContent = label;
    els.progressContainer().setAttribute('aria-valuenow', percent);
  }

  /**
   * Set the state of a step indicator.
   * @param {string} stepId  - e.g. 'step-extract'
   * @param {'active'|'done'|''} state
   */
  function setStep(stepId, state) {
    const el = document.getElementById(stepId);
    if (el) el.className = `step ${state}`.trim();
  }

  /** Reset all step indicators to inactive. */
  function resetSteps() {
    STEPS.forEach(id => setStep(id, ''));
  }

  // ── File Display ──────────────────────────────────────────────────────────

  /**
   * Show the loaded file name below the drop zone.
   * @param {string} filename
   */
  function showFilename(filename) {
    const el = els.filenameDisplay();
    el.textContent = `✓ ${filename}`;
    el.style.display = 'block';
  }

  // ── Analyze Button ────────────────────────────────────────────────────────

  /**
   * Enable or disable the analyze button.
   * @param {boolean} enabled
   */
  function setAnalyzeBtnEnabled(enabled) {
    const btn = els.analyzeBtn();
    btn.disabled = !enabled;
    btn.setAttribute('aria-disabled', String(!enabled));
  }

  // ── Results Rendering ─────────────────────────────────────────────────────

  /**
   * Populate and show the results panel.
   *
   * @param {object} params
   * @param {number} params.pageCount
   * @param {number} params.wordCount
   * @param {number} params.readTime
   * @param {string} params.outputTitle
   * @param {string} params.outputHtml
   * @param {string[]} params.keywords
   * @param {string[]} params.sections
   */
  function showResults({ pageCount, wordCount, readTime, outputTitle, outputHtml, keywords, sections }) {
    // Stats
    els.statPages().textContent = pageCount;
    els.statWords().textContent = wordCount > 1000
      ? `${(wordCount / 1000).toFixed(1)}k`
      : wordCount;
    els.statRead().textContent = `~${readTime}`;

    // Main output
    els.outputTitle().textContent = outputTitle;
    els.outputContent().innerHTML = outputHtml;

    // Keywords
    els.keywordsContainer().innerHTML = keywords
      .map(k => `<span class="tag">${k}</span>`)
      .join('');

    // Sections
    const secList = els.sectionsList();
    if (sections.length > 0) {
      secList.innerHTML = sections.map(s => `<li>${s}</li>`).join('');
    } else {
      secList.innerHTML = '<li>No standard sections detected</li>';
    }

    // Reveal
    const resultsEl = els.results();
    resultsEl.classList.add('visible');
    resultsEl.scrollIntoView({ behavior: 'smooth' });
  }

  /** Hide results panel. */
  function hideResults() {
    els.results().classList.remove('visible');
  }

  // ── Clipboard ─────────────────────────────────────────────────────────────

  /**
   * Copy the inner text of an element to the clipboard.
   * @param {string} elementId
   */
  function copyElementText(elementId) {
    const el = document.getElementById(elementId);
    if (!el) return;
    navigator.clipboard.writeText(el.innerText).then(() => {
      setStatus('Copied to clipboard!', 'ready');
      setTimeout(() => setStatus(''), 2000);
    }).catch(() => {
      setStatus('Could not access clipboard.', 'error');
    });
  }

  // ── Getters ───────────────────────────────────────────────────────────────

  /** @returns {string} Current value of the mode select */
  function getSelectedMode() {
    return els.modeSelect().value;
  }

  /** @returns {HTMLElement} The drop zone element */
  function getDropZone() {
    return els.dropZone();
  }

  /** @returns {HTMLInputElement} The file input element */
  function getFileInput() {
    return els.fileInput();
  }

  /** @returns {HTMLButtonElement} The analyze button element */
  function getAnalyzeBtn() {
    return els.analyzeBtn();
  }

  return {
    setStatus,
    setProgressVisible,
    setProgress,
    setStep,
    resetSteps,
    showFilename,
    setAnalyzeBtnEnabled,
    showResults,
    hideResults,
    copyElementText,
    getSelectedMode,
    getDropZone,
    getFileInput,
    getAnalyzeBtn,
  };

})();
