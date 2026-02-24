/* =========================================================
   outputGenerator.js — Render analysis results as HTML
   Module: Transforms NLP analysis into formatted output
           for each supported mode.
   PaperLens v1.0.0
   ========================================================= */

'use strict';

/**
 * @module OutputGenerator
 * Takes ranked sentences from NlpAnalyzer and formats them
 * according to the selected output mode.
 */
const OutputGenerator = (() => {

  // ── ELI5 simplification map ──────────────────────────────────────────────
  const ELI5_MAP = {
    methodology:    'method',
    implementation: 'how it works',
    algorithm:      'step-by-step process',
    parameter:      'setting',
    optimization:   'improvement',
    architecture:   'structure',
    computational:  'computer-based',
    utilization:    'use',
    hypothesis:     'idea being tested',
    empirical:      'based on real data',
    theoretical:    'based on ideas',
    evaluation:     'test',
    performance:    'how well it works',
    initialization: 'starting point',
    configuration:  'setup',
    leveraging:     'using',
    datasets:       'collections of data',
    benchmark:      'standard test',
    accuracy:       'correctness',
    precision:      'exactness',
  };

  // ── Mode config ───────────────────────────────────────────────────────────
  const MODE_CONFIG = {
    summary:  { label: 'Plain-Language Summary',  topN: 8 },
    bullets:  { label: 'Key Bullet Points',        topN: 8 },
    eli5:     { label: 'Simple Explanation (ELI5)', topN: 5 },
    abstract: { label: 'Rewritten Abstract',        topN: 6 },
  };

  // ── Helpers ───────────────────────────────────────────────────────────────

  /**
   * Get top-N ranked sentences, then re-order them by original position.
   *
   * @param {Array<{sentence: string, score: number}>} ranked
   * @param {string[]} originalOrder
   * @param {number} n
   * @returns {string[]}
   */
  function getTopSentences(ranked, originalOrder, n) {
    const top = ranked.slice(0, n).map(x => x.sentence);
    return top.sort((a, b) => originalOrder.indexOf(a) - originalOrder.indexOf(b));
  }

  /**
   * Apply ELI5 vocabulary simplification to a sentence.
   *
   * @param {string} sentence
   * @returns {string}
   */
  function simplify(sentence) {
    return sentence.replace(
      new RegExp(`\\b(${Object.keys(ELI5_MAP).join('|')})\\b`, 'gi'),
      match => ELI5_MAP[match.toLowerCase()] || match
    );
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * Generate HTML output for the given mode.
   *
   * @param {object} analysis - Result from NlpAnalyzer.analyze()
   * @param {string} mode     - One of: 'summary', 'bullets', 'eli5', 'abstract'
   * @returns {{ title: string, html: string }}
   */
  function generate(analysis, mode) {
    const { rankedSentences, sentences } = analysis;
    const config = MODE_CONFIG[mode] || MODE_CONFIG.summary;
    const topSentences = getTopSentences(rankedSentences, sentences, config.topN);

    let html = '';

    switch (mode) {

      case 'summary': {
        html = `<p>${topSentences.join(' ')}</p>`;
        break;
      }

      case 'bullets': {
        const items = topSentences.map(s => `<li>${s}</li>`).join('');
        html = `<ul>${items}</ul>`;
        break;
      }

      case 'eli5': {
        const simplified = topSentences.map(simplify);
        html = `
          <p><strong>The short version:</strong> Researchers explored a topic and here's
          what they found — explained simply:</p>
          <br />
          <p>${simplified.join(' ')}</p>
        `;
        break;
      }

      case 'abstract': {
        html = `<p>${topSentences.join(' ')}</p>`;
        break;
      }

      default:
        html = `<p>Unknown output mode: ${mode}</p>`;
    }

    return { title: config.label, html };
  }

  /**
   * Get the human-readable label for a mode.
   *
   * @param {string} mode
   * @returns {string}
   */
  function getModeLabel(mode) {
    return MODE_CONFIG[mode]?.label ?? mode;
  }

  return { generate, getModeLabel };

})();
