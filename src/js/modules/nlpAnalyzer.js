/* =========================================================
   nlpAnalyzer.js — Local NLP analysis (no external model)
   Module: TF-IDF keyword extraction, sentence scoring,
           section detection, readability stats.
   PaperLens v1.0.0
   ========================================================= */

'use strict';

/**
 * @module NlpAnalyzer
 * Performs all natural language processing locally in the browser.
 * Uses a TF-IDF-inspired approach to rank sentences and extract keywords.
 */
const NlpAnalyzer = (() => {

  // ── Stopwords list ───────────────────────────────────────────────────────
  const STOPWORDS = new Set([
    'the','a','an','and','or','but','in','on','at','to','for','of','with',
    'by','from','as','is','was','are','were','be','been','being','have',
    'has','had','do','does','did','will','would','could','should','may',
    'might','shall','this','that','these','those','it','its','we','our',
    'they','their','which','who','what','when','where','how','all','also',
    'more','can','not','paper','show','using','used','use','based','two',
    'one','each','than','such','into','through','between','after','during',
    'figure','table','al','et','ie','eg','however','proposed','present',
    'study','research','results','data','new','model','approach','method',
    'system','work','section','first','second','third','both','since',
    'while','given','well','thus','hence','note','also','other','several'
  ]);

  // ── Section heading patterns ─────────────────────────────────────────────
  const SECTION_PATTERNS = [
    { pattern: /abstract/i,              label: 'Abstract' },
    { pattern: /introduction/i,          label: 'Introduction' },
    { pattern: /background/i,            label: 'Background' },
    { pattern: /literature\s+review/i,   label: 'Literature Review' },
    { pattern: /related\s+work/i,        label: 'Related Work' },
    { pattern: /methodology|methods/i,   label: 'Methodology / Methods' },
    { pattern: /dataset/i,               label: 'Dataset' },
    { pattern: /experiment/i,            label: 'Experiments' },
    { pattern: /implementation/i,        label: 'Implementation' },
    { pattern: /evaluation/i,            label: 'Evaluation' },
    { pattern: /results?/i,              label: 'Results' },
    { pattern: /discussion/i,            label: 'Discussion' },
    { pattern: /conclusion/i,            label: 'Conclusion' },
    { pattern: /future\s+work/i,         label: 'Future Work' },
    { pattern: /references|bibliography/i, label: 'References' },
  ];

  // ── Helpers ──────────────────────────────────────────────────────────────

  /**
   * Tokenize text into cleaned lowercase words.
   * @param {string} text
   * @returns {string[]}
   */
  function tokenize(text) {
    return text
      .toLowerCase()
      .split(/\s+/)
      .map(w => w.replace(/[^a-z]/g, ''))
      .filter(w => w.length > 3 && !STOPWORDS.has(w));
  }

  /**
   * Compute term frequencies for a token array.
   * @param {string[]} tokens
   * @returns {Map<string, number>}
   */
  function termFrequency(tokens) {
    const freq = new Map();
    for (const t of tokens) {
      freq.set(t, (freq.get(t) || 0) + 1);
    }
    return freq;
  }

  /**
   * Split text into individual sentences.
   * @param {string} text
   * @returns {string[]}
   */
  function splitSentences(text) {
    return text
      .replace(/\n+/g, ' ')
      .split(/(?<=[.!?])\s+/)
      .map(s => s.trim())
      .filter(s => s.length > 60 && s.length < 450 && /[a-zA-Z]/.test(s));
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /**
   * Analyze a document and return structured insights.
   *
   * @param {string} text - Full extracted text from the PDF.
   * @returns {{
   *   sentences: string[],
   *   rankedSentences: Array<{sentence: string, score: number}>,
   *   keywords: string[],
   *   sections: string[],
   *   wordCount: number,
   *   readTime: number
   * }}
   */
  function analyze(text) {
    const words = text.split(/\s+/).filter(w => w.length > 0);
    const wordCount = words.length;
    const readTime = Math.ceil(wordCount / 200); // avg adult reading speed

    // Keyword extraction via term frequency
    const tokens = tokenize(text);
    const freq = termFrequency(tokens);
    const keywords = [...freq.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 18)
      .map(([word]) => word);

    const keySet = new Set(keywords);

    // Section detection
    const sections = SECTION_PATTERNS
      .filter(({ pattern }) => pattern.test(text))
      .map(({ label }) => label);

    // Sentence scoring
    const sentences = splitSentences(text);
    const rankedSentences = sentences.map((sentence, index) => {
      let score = 0;

      // Position bonus: early and late sentences tend to be more informative
      const relativePos = index / sentences.length;
      if (relativePos < 0.1 || relativePos > 0.85) score += 2;

      // Keyword density bonus
      const sentTokens = sentence.toLowerCase().split(/\s+/);
      sentTokens.forEach(w => {
        if (keySet.has(w.replace(/[^a-z]/g, ''))) score += 1;
      });

      // Prefer medium-length sentences
      if (sentence.length > 80 && sentence.length < 260) score += 1;

      // Penalty for citation-heavy sentences
      if (/\[\d+\]|\(\d{4}\)|\bet al\b/i.test(sentence)) score -= 2;

      // Penalty for sentences that look like figure/table captions
      if (/^(figure|table|fig\.?)\s+\d/i.test(sentence)) score -= 3;

      return { sentence, score };
    });

    // Sort by descending score
    rankedSentences.sort((a, b) => b.score - a.score);

    return { sentences, rankedSentences, keywords, sections, wordCount, readTime };
  }

  return { analyze };

})();
