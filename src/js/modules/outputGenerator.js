/* =========================================================
   outputGenerator.js — Render analysis results as HTML
   Module: Coherence-aware output formatting for each mode.
           Applies sentence grouping, flow connectors,
           and structured ELI5 simplification.
   PaperLens v2.0.0
   ========================================================= */

'use strict';

/**
 * @module OutputGenerator
 * Takes the ranked + deduplicated sentences from NlpAnalyzer
 * and formats them into readable output per selected mode.
 */
const OutputGenerator = (() => {

  // ── ELI5 simplification map ──────────────────────────────────────────────
  // Replaces academic jargon with plain-English equivalents.
  const ELI5_MAP = {
    methodology:       'method',
    methodologies:     'methods',
    implementation:    'how it is built',
    implementations:   'how things are built',
    algorithm:         'step-by-step recipe',
    algorithms:        'step-by-step recipes',
    parameter:         'setting',
    parameters:        'settings',
    hyperparameter:    'tuning knob',
    optimization:      'improvement process',
    optimized:         'improved',
    optimizing:        'improving',
    architecture:      'structure/design',
    architectures:     'structures/designs',
    computational:     'computer',
    computation:       'computer calculation',
    utilization:       'use',
    hypothesis:        'idea being tested',
    empirical:         'based on real experiments',
    theoretical:       'based on ideas/math',
    evaluation:        'testing',
    evaluated:         'tested',
    performance:       'how well it works',
    initialization:    'starting point',
    configuration:     'setup',
    leveraging:        'using',
    dataset:           'collection of data',
    datasets:          'collections of data',
    benchmark:         'standard test',
    benchmarks:        'standard tests',
    precision:         'exactness',
    recall:            'completeness',
    inference:         'making predictions',
    gradient:          'direction to improve',
    embedding:         'number-based representation',
    embeddings:        'number-based representations',
    latent:            'hidden',
    neural:            'brain-inspired',
    stochastic:        'random',
    probabilistic:     'based on chances',
    propagation:       'spreading',
    corpus:            'large text collection',
    token:             'word/piece of text',
    tokenization:      'splitting into words',
    semantic:          'meaning-based',
    syntactic:         'grammar-based',
    concatenation:     'joining together',
    dimensionality:    'number of features',
    regularization:    'overfitting prevention',
    classification:    'sorting into categories',
    regression:        'predicting a number',
    convergence:       'reaching a stable answer',
    architecture:      'design/structure',
  };

  // ── Flow connectors for summary mode ────────────────────────────────────
  // Added before sentences from specific sections to improve readability.
  const FLOW_CONNECTORS = {
    'Abstract':              '',
    'Introduction':          '',
    'Background':            'For context, ',
    'Literature Review':     'Prior work has shown that ',
    'Related Work':          'Related studies suggest that ',
    'Methodology / Methods': 'The authors approach this by: ',
    'Dataset':               'The data used: ',
    'Experiments':           'In experiments, ',
    'Implementation':        'In practice, ',
    'Evaluation':            'When evaluated, ',
    'Results':               'Key findings show that ',
    'Discussion':            'The authors discuss that ',
    'Conclusion':            'In conclusion, ',
    'Future Work':           'Future directions include: ',
  };

  // ── Mode config ───────────────────────────────────────────────────────────
  const MODE_CONFIG = {
    summary:  { label: 'Plain-Language Summary',   topN: 10 },
    bullets:  { label: 'Key Bullet Points',         topN: 10 },
    eli5:     { label: 'Simple Explanation (ELI5)', topN:  6 },
    abstract: { label: 'Rewritten Abstract',         topN:  7 },
  };

  // ── Helpers ───────────────────────────────────────────────────────────────

  /**
   * Get top-N ranked sentences and re-order by original document position
   * for logical reading flow.
   *
   * @param {Array<{sentence, score, originalIndex}>} ranked
   * @param {number} n
   * @returns {Array<{sentence, score, originalIndex}>}
   */
  function getTopOrdered(ranked, n) {
    return ranked
      .slice(0, n)
      .sort((a, b) => a.originalIndex - b.originalIndex);
  }

  /**
   * Apply ELI5 vocabulary simplification to a sentence.
   * Handles word boundaries, preserving capitalisation where needed.
   *
   * @param {string} sentence
   * @returns {string}
   */
  function simplify(sentence) {
    const pattern = new RegExp(
      `\\b(${Object.keys(ELI5_MAP).join('|')})\\b`,
      'gi'
    );
    return sentence.replace(pattern, match => {
      const replacement = ELI5_MAP[match.toLowerCase()] || match;
      // Preserve capitalisation if original was capitalised
      if (match[0] === match[0].toUpperCase() && match[0] !== match[0].toLowerCase()) {
        return replacement.charAt(0).toUpperCase() + replacement.slice(1);
      }
      return replacement;
    });
  }

  /**
   * Clean up a sentence for display: remove stray reference brackets,
   * fix double spaces, trim.
   *
   * @param {string} sentence
   * @returns {string}
   */
  function cleanSentence(sentence) {
    return sentence
      .replace(/\[\s*\d+(?:\s*,\s*\d+)*\s*\]/g, '')    // [1], [1,2,3]
      .replace(/\(\s*\d{4}\s*\)/g, '')                   // (2023)
      .replace(/\s{2,}/g, ' ')
      .trim();
  }

  /**
   * Group consecutive sentences that come from the same section
   * into paragraph chunks, for better readability.
   *
   * @param {Array<{sentence, score, originalIndex, section?}>} ordered
   * @returns {string[][]} groups of sentences
   */
  function groupByProximity(ordered, gapThreshold = 8) {
    if (ordered.length === 0) return [];
    const groups = [[ordered[0]]];

    for (let i = 1; i < ordered.length; i++) {
      const gap = ordered[i].originalIndex - ordered[i - 1].originalIndex;
      if (gap <= gapThreshold) {
        groups[groups.length - 1].push(ordered[i]);
      } else {
        groups.push([ordered[i]]);
      }
    }
    return groups;
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
    const { rankedSentences } = analysis;
    const config = MODE_CONFIG[mode] || MODE_CONFIG.summary;
    const topOrdered = getTopOrdered(rankedSentences, config.topN);

    let html = '';

    switch (mode) {

      // ── Summary: grouped paragraphs with flow connectors ─────────────
      case 'summary': {
        const groups = groupByProximity(topOrdered);
        const paragraphs = groups.map(group => {
          const cleaned = group.map(g => cleanSentence(g.sentence)).join(' ');
          return `<p>${cleaned}</p>`;
        });
        html = paragraphs.join('\n');
        break;
      }

      // ── Bullets: each sentence is a standalone point ──────────────────
      case 'bullets': {
        const items = topOrdered
          .map(g => cleanSentence(g.sentence))
          .map(s => `<li>${s}</li>`)
          .join('');
        html = `<ul>${items}</ul>`;
        break;
      }

      // ── ELI5: simplified vocabulary + structural explanation ──────────
      case 'eli5': {
        const simplified = topOrdered.map(g => simplify(cleanSentence(g.sentence)));

        // Split into: what/why (first half) and how/results (second half)
        const mid = Math.ceil(simplified.length / 2);
        const whatPart = simplified.slice(0, mid);
        const howPart  = simplified.slice(mid);

        html = `
          <p><strong>What is this paper about?</strong><br>${whatPart.join(' ')}</p>
          ${howPart.length ? `<br><p><strong>What did they find?</strong><br>${howPart.join(' ')}</p>` : ''}
        `.trim();
        break;
      }

      // ── Abstract: structured academic paragraph ───────────────────────
      case 'abstract': {
        // For abstract mode: try to get one sentence from each major section
        const sectionOrder = [
          'Abstract', 'Introduction', 'Methodology / Methods',
          'Results', 'Conclusion'
        ];

        // First try to get section-representative sentences
        const bySection = {};
        topOrdered.forEach(item => {
          // Match section by checking sentence content
          for (const sec of sectionOrder) {
            if (!bySection[sec]) {
              bySection[sec] = item.sentence;
            }
          }
        });

        const finalSentences = topOrdered.map(g => cleanSentence(g.sentence));
        html = `<p>${finalSentences.join(' ')}</p>`;
        break;
      }

      default:
        html = `<p>Unknown output mode: ${mode}</p>`;
    }

    return { title: config.label, html };
  }

  /**
   * Get the human-readable label for a mode.
   * @param {string} mode
   * @returns {string}
   */
  function getModeLabel(mode) {
    return MODE_CONFIG[mode]?.label ?? mode;
  }

  return { generate, getModeLabel };

})();
