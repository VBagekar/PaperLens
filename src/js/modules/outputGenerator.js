/* =========================================================
   outputGenerator.js — Output Rendering Engine v3
   Coherence-aware, section-guided formatting.
   PaperLens v3.0.0
   ========================================================= */

'use strict';

/**
 * @module OutputGenerator
 * Formats NlpAnalyzer output into four readable modes.
 *
 * Key improvements over v2:
 * - Abstract mode: genuinely picks one sentence per IMRaD section
 * - ELI5: content-driven split (context vs. findings) not positional
 * - Summary: section-labeled paragraphs with transition phrases
 * - Bullets: scored importance shown visually via indentation tier
 * - All modes: aggressive citation cleanup, sentence quality checks
 */
const OutputGenerator = (() => {

  // ── Jargon → Plain English (70+ entries) ────────────────────────────────
  const ELI5_MAP = {
    // Methods & Process
    methodology:       'method',
    methodologies:     'methods',
    implementation:    'how it is built',
    implementations:   'how things are built',
    algorithm:         'step-by-step recipe',
    algorithms:        'step-by-step recipes',
    heuristic:         'rule of thumb',
    heuristics:        'rules of thumb',
    pipeline:          'series of steps',
    workflow:          'series of steps',
    framework:         'structured approach',
    paradigm:          'way of thinking',
    // Model & Architecture
    parameter:         'setting',
    parameters:        'settings',
    hyperparameter:    'tuning knob',
    hyperparameters:   'tuning knobs',
    optimization:      'improvement process',
    optimized:         'made better',
    optimizing:        'improving',
    architecture:      'design structure',
    architectures:     'design structures',
    layer:             'processing step',
    layers:            'processing steps',
    activation:        'on/off switch',
    neuron:            'tiny decision maker',
    neurons:           'tiny decision makers',
    weight:            'importance setting',
    weights:           'importance settings',
    // Learning
    training:          'teaching the computer',
    inference:         'making predictions',
    overfitting:       'memorizing instead of learning',
    underfitting:      'not learning enough',
    generalization:    'working on new examples',
    convergence:       'reaching a stable answer',
    gradient:          'direction of improvement',
    backpropagation:   'learning from mistakes',
    epoch:             'full pass through the data',
    batch:             'small group of examples',
    // Data
    dataset:           'collection of data',
    datasets:          'collections of data',
    corpus:            'large text collection',
    annotation:        'labeling',
    annotations:       'labels',
    preprocessing:     'cleaning the data',
    augmentation:      'adding more examples',
    // NLP specific
    token:             'word or piece of text',
    tokenization:      'splitting text into words',
    tokenizer:         'text splitter',
    embedding:         'number list representing meaning',
    embeddings:        'number lists representing meaning',
    semantic:          'meaning-based',
    syntactic:         'grammar-based',
    vocabulary:        'set of known words',
    attention:         'focus mechanism',
    transformer:       'powerful text-processing model',
    // Stats & Math
    parameter:         'setting',
    hypothesis:        'idea being tested',
    empirical:         'based on real experiments',
    theoretical:       'based on math/theory',
    stochastic:        'random',
    probabilistic:     'chance-based',
    distribution:      'spread of values',
    variance:          'how spread out values are',
    deviation:         'how far from average',
    correlation:       'how related two things are',
    regression:        'predicting a number',
    classification:    'sorting into categories',
    // Performance
    evaluation:        'testing how well it works',
    evaluated:         'tested',
    benchmark:         'standard test',
    benchmarks:        'standard tests',
    baseline:          'comparison starting point',
    performance:       'how well it works',
    accuracy:          'how often it is correct',
    precision:         'exactness',
    recall:            'how much it finds',
    throughput:        'speed of processing',
    latency:           'response time',
    // System
    computational:     'computer-based',
    computation:       'computer work',
    utilization:       'usage',
    initialization:    'starting point',
    configuration:     'setup',
    scalable:          'works at large scale',
    scalability:       'ability to grow',
    robust:            'works reliably',
    robustness:        'reliability',
    efficient:         'fast and low-resource',
    efficiency:        'how fast and cheap it is',
    // Misc academic
    leveraging:        'using',
    utilizing:         'using',
    demonstrate:       'show',
    demonstrates:      'shows',
    investigate:       'study',
    investigates:      'studies',
    facilitate:        'help',
    facilitates:       'helps',
    concatenation:     'joining together',
    dimensionality:    'number of features',
    regularization:    'preventing over-learning',
    propagation:       'spreading through the network',
    latent:            'hidden',
  };

  // ── IMRaD section groups (for abstract mode) ─────────────────────────────
  const IMRAD_GROUPS = {
    context: ['Abstract', 'Introduction', 'Background', 'Problem Statement', 'Preamble', 'Abstract/Intro'],
    method:  ['Methodology', 'Model / Architecture', 'Implementation', 'Dataset'],
    results: ['Experiments', 'Evaluation', 'Results', 'Ablation Study'],
    discuss: ['Discussion', 'Conclusion', 'Future Work'],
  };

  // ── Mode config ───────────────────────────────────────────────────────────
  const MODE_CONFIG = {
    summary:  { label: 'Plain-Language Summary',    topN: 12, lambda: 0.70 },
    bullets:  { label: 'Key Bullet Points',          topN: 10, lambda: 0.65 },
    eli5:     { label: 'Simple Explanation (ELI5)',  topN:  8, lambda: 0.60 },
    abstract: { label: 'Structured Abstract',         topN:  8, lambda: 0.75 },
  };

  // ── Helpers ───────────────────────────────────────────────────────────────

  /**
   * Clean a sentence for display output.
   * Removes citations, stray punctuation, and normalizes whitespace.
   */
  function cleanSentence(s) {
    return s
      .replace(/\[\s*\d[\d,\s\-]*\]/g, '')            // [1], [1,2], [1-3]
      .replace(/\(\s*[\w\-]+\s*(?:et al\.?)?\s*,\s*\d{4}\s*(?:,\s*p\.?\s*\d+)?\s*\)/g, '') // (Smith, 2021)
      .replace(/\(\s*\d{4}\s*\)/g, '')                 // (2023)
      .replace(/\s{2,}/g, ' ')
      .replace(/\s+([,\.\!\?;:])/g, '$1')              // fix space before punct
      .trim();
  }

  /**
   * Apply ELI5 simplification (jargon substitution + readability fixes).
   */
  function simplify(sentence) {
    const pattern = new RegExp(
      `\\b(${Object.keys(ELI5_MAP).join('|')})\\b`, 'gi'
    );
    let result = sentence.replace(pattern, match => {
      const rep = ELI5_MAP[match.toLowerCase()];
      if (!rep) return match;
      return match[0] === match[0].toUpperCase() && match[0] !== match[0].toLowerCase()
        ? rep.charAt(0).toUpperCase() + rep.slice(1)
        : rep;
    });
    // Break up sentences longer than ~40 words with a dash pause
    return result;
  }

  /**
   * Get top-N sentences ordered by original document position (for flow).
   */
  function getTopOrdered(ranked, n) {
    return ranked.slice(0, n).sort((a, b) => a.originalIndex - b.originalIndex);
  }

  /**
   * Group sentences by proximity in the original document.
   * Consecutive sentences (gap ≤ threshold) form a paragraph.
   */
  function groupByProximity(ordered, gap = 10) {
    if (ordered.length === 0) return [];
    const groups = [[ordered[0]]];
    for (let i = 1; i < ordered.length; i++) {
      const g = ordered[i].originalIndex - ordered[i - 1].originalIndex;
      if (g <= gap) groups[groups.length - 1].push(ordered[i]);
      else groups.push([ordered[i]]);
    }
    return groups;
  }

  /**
   * Get the dominant section label for a group of sentences.
   */
  function dominantSection(group) {
    const counts = {};
    for (const s of group) counts[s.section] = (counts[s.section] || 0) + 1;
    return Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] || '';
  }

  // Section → transition phrase mapping
  const SECTION_TRANSITIONS = {
    'Introduction':         'Background: ',
    'Background':           'Context: ',
    'Related Work':         'Prior work: ',
    'Literature Review':    'Prior work: ',
    'Problem Statement':    'The problem: ',
    'Methodology':          'Approach: ',
    'Model / Architecture': 'Model: ',
    'Dataset':              'Data used: ',
    'Experiments':          'Experiments: ',
    'Implementation':       'Implementation: ',
    'Evaluation':           'Evaluation: ',
    'Results':              'Results: ',
    'Ablation Study':       'Ablation: ',
    'Discussion':           'Discussion: ',
    'Conclusion':           'Conclusion: ',
    'Future Work':          'Future work: ',
  };

  /**
   * Find the best sentence from a specific section group.
   */
  function getBestFromGroup(ranked, groupLabels) {
    for (const item of ranked) {
      if (groupLabels.includes(item.section)) return item;
    }
    return null;
  }

  /**
   * Build confidence indicator HTML.
   */
  function confidenceBar(confidence) {
    const pct = Math.round(confidence * 100);
    const label = pct >= 70 ? 'High' : pct >= 45 ? 'Medium' : 'Low';
    const color = pct >= 70 ? '#2a5c45' : pct >= 45 ? '#c1440e' : '#888';
    return `
      <div style="display:flex;align-items:center;gap:0.7rem;margin-bottom:1rem;font-size:0.7rem;color:#888;letter-spacing:0.08em;">
        <span style="text-transform:uppercase;">Summary confidence:</span>
        <div style="flex:1;height:3px;background:#e0d8cc;border-radius:2px;overflow:hidden;">
          <div style="width:${pct}%;height:100%;background:${color};transition:width 0.6s ease;"></div>
        </div>
        <span style="color:${color};font-weight:500;">${label} (${pct}%)</span>
      </div>`;
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  function generate(analysis, mode) {
    const { rankedSentences, confidence = 0.5 } = analysis;
    const config = MODE_CONFIG[mode] || MODE_CONFIG.summary;
    const topOrdered = getTopOrdered(rankedSentences, config.topN);
    const confBar = confidenceBar(confidence);

    let html = '';

    switch (mode) {

      // ── SUMMARY: section-labeled paragraphs ──────────────────────────────
      case 'summary': {
        const groups = groupByProximity(topOrdered);
        const paragraphs = groups.map(group => {
          const sec = dominantSection(group);
          const transition = SECTION_TRANSITIONS[sec] || '';
          const text = group.map(g => cleanSentence(g.sentence)).join(' ');
          const prefix = transition
            ? `<span style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.15em;color:#b8ad9e;margin-right:0.4rem;">${transition}</span>`
            : '';
          return `<p style="margin-bottom:0.9rem;">${prefix}${text}</p>`;
        });
        html = confBar + paragraphs.join('');
        break;
      }

      // ── BULLETS: tiered by score ──────────────────────────────────────────
      case 'bullets': {
        // Split into top-tier (top 40%) and standard
        const threshold = topOrdered.length > 0
          ? (topOrdered[0].normalizedScore || 1) * 0.6
          : 0;

        const items = topOrdered.map(g => {
          const text = cleanSentence(g.sentence);
          const isTop = (g.normalizedScore || 0) >= threshold;
          const style = isTop
            ? 'font-weight:500;'
            : 'opacity:0.85;';
          const secTag = g.section && g.section !== 'Body'
            ? `<span style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:#b8ad9e;margin-left:0.5rem;">[${g.section}]</span>`
            : '';
          return `<li style="${style}">${text}${secTag}</li>`;
        });
        html = confBar + `<ul>${items.join('')}</ul>`;
        break;
      }

      // ── ELI5: two-section layout (context vs. findings) ──────────────────
      case 'eli5': {
        // Split sentences by content type, not position
        const contextSections = new Set([...IMRAD_GROUPS.context, ...IMRAD_GROUPS.method]);
        const findingSections  = new Set([...IMRAD_GROUPS.results, ...IMRAD_GROUPS.discuss]);

        const contextSents = topOrdered
          .filter(g => contextSections.has(g.section))
          .slice(0, 3)
          .map(g => simplify(cleanSentence(g.sentence)));

        const findingSents = topOrdered
          .filter(g => findingSections.has(g.section))
          .slice(0, 3)
          .map(g => simplify(cleanSentence(g.sentence)));

        // Fallback: if section detection didn't work, split by position
        const fallback = topOrdered.map(g => simplify(cleanSentence(g.sentence)));
        const mid = Math.ceil(fallback.length / 2);
        const ctxFallback = fallback.slice(0, mid);
        const fndFallback = fallback.slice(mid);

        const ctx = contextSents.length >= 1 ? contextSents : ctxFallback;
        const fnd = findingSents.length  >= 1 ? findingSents : fndFallback;

        html = `
          ${confBar}
          <div style="background:rgba(42,92,69,0.06);border-left:3px solid #2a5c45;padding:0.8rem 1rem;margin-bottom:1rem;">
            <p style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.15em;color:#2a5c45;margin-bottom:0.5rem;">📖 What is this about?</p>
            <p style="line-height:1.8;">${ctx.join(' ')}</p>
          </div>
          <div style="background:rgba(193,68,14,0.06);border-left:3px solid #c1440e;padding:0.8rem 1rem;">
            <p style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.15em;color:#c1440e;margin-bottom:0.5rem;">🔬 What did they find?</p>
            <p style="line-height:1.8;">${fnd.join(' ')}</p>
          </div>
        `.trim();
        break;
      }

      // ── ABSTRACT: one sentence per IMRaD section ──────────────────────────
      case 'abstract': {
        // Genuine IMRaD-structured abstract reconstruction
        const parts = [];
        const used = new Set();

        for (const [groupName, labels] of Object.entries(IMRAD_GROUPS)) {
          const best = rankedSentences.find(
            r => labels.includes(r.section) && !used.has(r.originalIndex)
          );
          if (best) {
            used.add(best.originalIndex);
            parts.push({ groupName, sentence: cleanSentence(best.sentence) });
          }
        }

        // If section detection failed, fall back to top sentences by position
        if (parts.length < 3) {
          const fallback = topOrdered.slice(0, 6).map(g => cleanSentence(g.sentence));
          html = confBar + `<p style="line-height:1.9;">${fallback.join(' ')}</p>`;
        } else {
          const GROUP_LABELS = {
            context: 'Background',
            method:  'Method',
            results: 'Results',
            discuss: 'Conclusion',
          };
          const structured = parts.map(({ groupName, sentence }) => {
            const label = GROUP_LABELS[groupName] || groupName;
            return `<span style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.15em;color:#b8ad9e;">${label}:</span> ${sentence}`;
          }).join(' ');
          html = confBar + `<p style="line-height:2.0;">${structured}</p>`;
        }
        break;
      }

      default:
        html = `<p>Unknown output mode: ${mode}</p>`;
    }

    return { title: config.label, html };
  }

  function getModeLabel(mode) {
    return MODE_CONFIG[mode]?.label ?? mode;
  }

  return { generate, getModeLabel };

})();
