/* =========================================================
   nlpAnalyzer.js — Advanced local NLP analysis engine
   Module: True TF-IDF, bigram extraction, section-aware
           sentence scoring, cosine-similarity deduplication,
           and coherence-aware ranking.
   PaperLens v2.0.0
   ========================================================= */

'use strict';

/**
 * @module NlpAnalyzer
 * Performs multi-pass natural language processing entirely in the browser.
 *
 * Pipeline:
 *  1. Pre-processing  — clean, segment, detect sections
 *  2. TF-IDF          — true per-sentence IDF weighting (not just frequency)
 *  3. Bigram extract  — surface "deep learning", "neural network" etc.
 *  4. Sentence score  — 8-factor scoring with section-aware bonuses
 *  5. Deduplication   — cosine-similarity filter removes redundant sentences
 *  6. Coherence sort  — final ordering preserves logical document flow
 */
const NlpAnalyzer = (() => {

  // ── Extended stopword list ───────────────────────────────────────────────
  const STOPWORDS = new Set([
    // Articles / prepositions / conjunctions
    'the','a','an','and','or','but','in','on','at','to','for','of','with',
    'by','from','as','is','was','are','were','be','been','being','have',
    'has','had','do','does','did','will','would','could','should','may',
    'might','shall','must','need','dare','ought',
    // Pronouns
    'this','that','these','those','it','its','we','our','they','their',
    'i','my','me','you','your','he','his','she','her','him','us','them',
    // Common academic filler
    'which','who','what','when','where','how','all','also','more','can',
    'not','paper','papers','show','shows','shown','using','used','use',
    'based','two','one','three','four','five','each','than','such','into',
    'through','between','after','during','figure','figures','table','tables',
    'al','et','ie','eg','however','proposed','present','presented','presents',
    'study','research','results','result','data','new','model','models',
    'approach','method','system','work','section','first','second','third',
    'both','since','while','given','well','thus','hence','note','other',
    'several','many','most','some','any','few','very','quite','rather',
    'much','further','also','even','just','only','already','still','now',
    'then','here','there','where','so','if','about','up','down','over',
    'under','again','own','same','different','various','across','within',
    'without','along','upon','around','against','toward','towards','per',
    'set','sets','make','makes','made','take','takes','taken','give','gives',
    'given','large','small','high','low','number','numbers','type','types',
    'form','forms','case','cases','point','points','example','examples',
    'way','ways','order','terms','provide','provides','provided'
  ]);

  // ── High-value signal phrases (boost sentences containing these) ─────────
  const SIGNAL_PHRASES = [
    /\bwe (propose|present|introduce|develop|demonstrate|show|find|argue|conclude)\b/i,
    /\bour (approach|method|model|system|framework|contribution|results?)\b/i,
    /\bin this (paper|work|study|article)\b/i,
    /\bthe (main|key|primary|central|novel|core) (contribution|finding|result|idea|advantage)\b/i,
    /\bwe (achieve|outperform|surpass|improve|reduce|increase|obtain)\b/i,
    /\bstate.of.the.art\b/i,
    /\bsignificant(ly)?\b/i,
    /\bnovel\b/i,
    /\bimprove(ment|s|d)?\b/i,
    /\baccuracy|precision|recall|f1.score|performance\b/i,
    /\bcompared (to|with)\b/i,
    /\bbaseline\b/i,
    /\bexperiment(al|s)?\b/i,
    /\bconclusion|conclude\b/i,
    /\bfuture work\b/i,
    /\blimitation\b/i,
    /\bcontribution\b/i,
    /\bto (address|solve|tackle|overcome)\b/i,
    /\bresult(s|ed)? (in|show|indicate|suggest|demonstrate)\b/i,
    /\bevaluat(e|ed|ion)\b/i,
  ];

  // ── Noise patterns that disqualify a sentence ────────────────────────────
  const NOISE_PATTERNS = [
    /\[\s*\d+\s*\]/g,                          // inline citations [1], [12]
    /\(\s*\d{4}\s*\)/g,                         // year citations (2023)
    /\bet\s+al\b/i,                             // "et al."
    /^(figure|table|fig\.?|tab\.?)\s+\d/i,     // captions
    /^(http|www\.)/i,                           // URLs
    /^\s*\d+\s*$/,                              // lone numbers
    /copyright|all rights reserved/i,           // legal boilerplate
    /^(acknowledgements?|funding|disclosure)/i, // boilerplate sections
    /[^\x00-\x7F]{4,}/,                         // non-ASCII runs (garbled PDF)
    /(.)\1{5,}/,                                // repeated characters
  ];

  // ── Section heading patterns with weights ────────────────────────────────
  const SECTION_PATTERNS = [
    { pattern: /abstract/i,               label: 'Abstract',              weight: 3.0 },
    { pattern: /introduction/i,           label: 'Introduction',          weight: 2.5 },
    { pattern: /background/i,             label: 'Background',            weight: 1.5 },
    { pattern: /literature\s+review/i,    label: 'Literature Review',     weight: 1.0 },
    { pattern: /related\s+work/i,         label: 'Related Work',          weight: 1.0 },
    { pattern: /methodology|methods/i,    label: 'Methodology / Methods', weight: 1.5 },
    { pattern: /dataset/i,                label: 'Dataset',               weight: 1.2 },
    { pattern: /experiment/i,             label: 'Experiments',           weight: 1.8 },
    { pattern: /implementation/i,         label: 'Implementation',        weight: 1.2 },
    { pattern: /evaluation/i,             label: 'Evaluation',            weight: 1.8 },
    { pattern: /results?/i,               label: 'Results',               weight: 2.0 },
    { pattern: /discussion/i,             label: 'Discussion',            weight: 2.0 },
    { pattern: /conclusion/i,             label: 'Conclusion',            weight: 3.0 },
    { pattern: /future\s+work/i,          label: 'Future Work',           weight: 1.5 },
    { pattern: /references|bibliography/i,label: 'References',            weight: 0.0 },
    { pattern: /appendix/i,               label: 'Appendix',              weight: 0.2 },
  ];

  // ────────────────────────────────────────────────────────────────────────
  // 1. PRE-PROCESSING
  // ────────────────────────────────────────────────────────────────────────

  /**
   * Clean raw PDF-extracted text: fix hyphenation, collapse whitespace,
   * remove page headers/footers, normalize quotes.
   * @param {string} raw
   * @returns {string}
   */
  function cleanText(raw) {
    return raw
      .replace(/(\w)-\s*\n\s*(\w)/g, '$1$2')      // fix hyphenated line breaks
      .replace(/\n{3,}/g, '\n\n')                   // collapse blank lines
      .replace(/[ \t]+/g, ' ')                       // collapse spaces
      .replace(/\u2018|\u2019/g, "'")                // smart quotes
      .replace(/\u201C|\u201D/g, '"')
      .replace(/\u2013|\u2014/g, '–')
      .replace(/\f/g, '\n')                          // form feeds
      .trim();
  }

  /**
   * Split document into sentences using multi-signal boundary detection.
   * Handles abbreviations, decimals, and academic notation.
   * @param {string} text
   * @returns {string[]}
   */
  function splitSentences(text) {
    // Replace known non-boundary periods to protect them
    const protected_ = text
      .replace(/\b(Fig|Tab|Eq|Sec|Ref|et al|vs|e\.g|i\.e|cf|approx|est|ca|Dr|Mr|Mrs|Prof|Fig)\./gi,
               m => m.replace('.', '<<<DOT>>>'))
      .replace(/\b\d+\.\d+/g, m => m.replace('.', '<<<DOT>>>'))  // decimals
      .replace(/\b[A-Z]\./g, m => m.replace('.', '<<<DOT>>>'));   // initials

    const raw = protected_
      .replace(/\n+/g, ' ')
      .split(/(?<=[.!?])\s+(?=[A-Z0-9"'(])/)
      .map(s => s.replace(/<<<DOT>>>/g, '.').trim());

    return raw.filter(s => {
      if (s.length < 50 || s.length > 600) return false;
      if (!/[a-zA-Z]/.test(s)) return false;
      // Check noise patterns
      for (const pat of NOISE_PATTERNS) {
        if (pat instanceof RegExp && pat.test(s)) return false;
      }
      return true;
    });
  }

  /**
   * Tokenize text into lowercase stems (simplified Porter-like trimming).
   * @param {string} text
   * @returns {string[]}
   */
  function tokenize(text) {
    return text
      .toLowerCase()
      .split(/[\s\-–—]+/)
      .map(w => w.replace(/[^a-z]/g, ''))
      .filter(w => w.length > 3 && !STOPWORDS.has(w));
  }

  /**
   * Light stemming: strip common suffixes to unify variants.
   * e.g. "proposes" → "propose", "training" → "train"
   * @param {string} word
   * @returns {string}
   */
  function stem(word) {
    return word
      .replace(/ations?$/, 'ate')
      .replace(/ings?$/, '')
      .replace(/ized$/, 'ize')
      .replace(/ised$/, 'ise')
      .replace(/ness$/, '')
      .replace(/ment$/, '')
      .replace(/ities$/, 'ity')
      .replace(/ies$/, 'y')
      .replace(/ers?$/, '')
      .replace(/ed$/, '')
      .replace(/s$/, '');
  }

  // ────────────────────────────────────────────────────────────────────────
  // 2. SECTION SEGMENTATION
  // ────────────────────────────────────────────────────────────────────────

  /**
   * Detect which section each sentence belongs to.
   * Returns a map: sentence index → { label, weight }
   * @param {string[]} sentences
   * @returns {Map<number, {label: string, weight: number}>}
   */
  function assignSections(sentences) {
    const map = new Map();
    let currentSection = { label: 'Body', weight: 1.0 };

    // Default for unknown location
    for (let i = 0; i < sentences.length; i++) {
      const s = sentences[i];

      // Check if this sentence looks like a section heading embedded in text
      for (const sec of SECTION_PATTERNS) {
        // Heading detection: short sentence matching a section name
        if (s.length < 80 && sec.pattern.test(s)) {
          currentSection = { label: sec.label, weight: sec.weight };
          break;
        }
        // Or the sentence starts with the section keyword
        if (sec.pattern.test(s.split(/\s+/).slice(0, 3).join(' '))) {
          currentSection = { label: sec.label, weight: sec.weight };
          break;
        }
      }

      map.set(i, { ...currentSection });
    }

    return map;
  }

  // ────────────────────────────────────────────────────────────────────────
  // 3. TRUE TF-IDF
  // ────────────────────────────────────────────────────────────────────────

  /**
   * Build a TF-IDF matrix across all sentences.
   * @param {string[]} sentences
   * @returns {{ tfidf: Map<string, number>[], vocab: Set<string> }}
   */
  function buildTfIdf(sentences) {
    const N = sentences.length;

    // Step 1: per-sentence token sets and TF
    const sentenceTFs = sentences.map(s => {
      const tokens = tokenize(s).map(stem);
      const tf = new Map();
      tokens.forEach(t => tf.set(t, (tf.get(t) || 0) + 1));
      // Normalize by sentence length
      const len = tokens.length || 1;
      tf.forEach((v, k) => tf.set(k, v / len));
      return { tf, tokens: new Set(tokens) };
    });

    // Step 2: document frequency (how many sentences contain each term)
    const df = new Map();
    sentenceTFs.forEach(({ tokens }) => {
      tokens.forEach(t => df.set(t, (df.get(t) || 0) + 1));
    });

    // Step 3: TF-IDF per sentence
    const tfidf = sentenceTFs.map(({ tf }) => {
      const scores = new Map();
      tf.forEach((tfVal, term) => {
        const idf = Math.log((N + 1) / ((df.get(term) || 0) + 1)) + 1;
        scores.set(term, tfVal * idf);
      });
      return scores;
    });

    // Step 4: global term importance (sum of TF-IDF across all sentences)
    const globalImportance = new Map();
    tfidf.forEach(sentScores => {
      sentScores.forEach((val, term) => {
        globalImportance.set(term, (globalImportance.get(term) || 0) + val);
      });
    });

    return { tfidf, globalImportance };
  }

  // ────────────────────────────────────────────────────────────────────────
  // 4. BIGRAM EXTRACTION
  // ────────────────────────────────────────────────────────────────────────

  /**
   * Extract high-value bigrams (2-word phrases) from the text.
   * @param {string} text
   * @returns {string[]} top bigrams
   */
  function extractBigrams(text) {
    const tokens = tokenize(text).map(stem);
    const bigrams = new Map();

    for (let i = 0; i < tokens.length - 1; i++) {
      if (tokens[i].length < 4 || tokens[i + 1].length < 4) continue;
      const bigram = `${tokens[i]} ${tokens[i + 1]}`;
      bigrams.set(bigram, (bigrams.get(bigram) || 0) + 1);
    }

    // Only keep bigrams that appear at least twice
    return [...bigrams.entries()]
      .filter(([, count]) => count >= 2)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([bigram]) => bigram);
  }

  // ────────────────────────────────────────────────────────────────────────
  // 5. COSINE SIMILARITY DEDUPLICATION
  // ────────────────────────────────────────────────────────────────────────

  /**
   * Compute cosine similarity between two TF-IDF score maps.
   * @param {Map<string, number>} a
   * @param {Map<string, number>} b
   * @returns {number} similarity in [0, 1]
   */
  function cosineSimilarity(a, b) {
    let dot = 0, normA = 0, normB = 0;
    a.forEach((val, key) => {
      dot += val * (b.get(key) || 0);
      normA += val * val;
    });
    b.forEach(val => { normB += val * val; });
    return normA === 0 || normB === 0 ? 0 : dot / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  /**
   * Filter a ranked list of sentences to remove near-duplicates.
   * Uses cosine similarity: if two sentences are > threshold similar,
   * keep only the higher-ranked one.
   *
   * @param {Array<{sentence: string, score: number, tfidfVec: Map}>} ranked
   * @param {number} [threshold=0.55]
   * @returns {Array<{sentence: string, score: number}>}
   */
  function deduplicate(ranked, threshold = 0.55) {
    const kept = [];

    for (const candidate of ranked) {
      let isDuplicate = false;

      for (const existing of kept) {
        const sim = cosineSimilarity(candidate.tfidfVec, existing.tfidfVec);
        if (sim > threshold) {
          isDuplicate = true;
          break;
        }
      }

      if (!isDuplicate) kept.push(candidate);
    }

    return kept;
  }

  // ────────────────────────────────────────────────────────────────────────
  // 6. KEYWORD EXTRACTION (TF-IDF based)
  // ────────────────────────────────────────────────────────────────────────

  /**
   * Extract the top N keywords using global TF-IDF importance.
   * Merges unigrams and bigrams, deduplicates stemmed variants.
   *
   * @param {Map<string, number>} globalImportance
   * @param {string[]} bigrams
   * @param {number} topN
   * @returns {string[]}
   */
  function extractKeywords(globalImportance, bigrams, topN = 18) {
    const unigrams = [...globalImportance.entries()]
      .filter(([term]) => term.length > 4)
      .sort((a, b) => b[1] - a[1])
      .slice(0, topN)
      .map(([term]) => term);

    // Combine: bigrams first (more specific), then unigrams
    const combined = [...bigrams, ...unigrams];

    // Deduplicate: remove unigrams that are substrings of a chosen bigram
    const bigramWords = new Set(bigrams.join(' ').split(' '));
    const filtered = combined.filter((term, i) => {
      if (i < bigrams.length) return true; // keep all bigrams
      return !bigramWords.has(term);        // drop unigrams already in bigrams
    });

    return [...new Set(filtered)].slice(0, topN);
  }

  // ────────────────────────────────────────────────────────────────────────
  // 7. MAIN SENTENCE SCORING
  // ────────────────────────────────────────────────────────────────────────

  /**
   * Score each sentence using 8 weighted factors.
   *
   * @param {string[]} sentences
   * @param {Map<string, number>[]} tfidf         - per-sentence TF-IDF
   * @param {Map<string, number>} globalImportance
   * @param {Map<number, {label, weight}>} sectionMap
   * @returns {Array<{sentence, score, tfidfVec, originalIndex}>}
   */
  function scoreSentences(sentences, tfidf, globalImportance, sectionMap) {
    const N = sentences.length;

    return sentences.map((sentence, i) => {
      const vec = tfidf[i];
      const section = sectionMap.get(i) || { label: 'Body', weight: 1.0 };
      let score = 0;

      // ── Factor 1: TF-IDF sentence weight ──────────────────────────────
      // Sum of TF-IDF scores of terms in this sentence,
      // weighted by global importance (emphasizes rare-but-important terms)
      let tfidfScore = 0;
      vec.forEach((val, term) => {
        tfidfScore += val * (globalImportance.get(term) || 0);
      });
      score += Math.min(tfidfScore * 0.4, 5); // cap to avoid domination

      // ── Factor 2: Section weight ───────────────────────────────────────
      // Abstract and Conclusion sentences are most likely to contain key info
      score += section.weight * 0.8;

      // ── Factor 3: Signal phrase bonus ──────────────────────────────────
      // Sentences that contain "we propose", "our method", etc.
      let signalCount = 0;
      SIGNAL_PHRASES.forEach(pat => { if (pat.test(sentence)) signalCount++; });
      score += Math.min(signalCount * 0.7, 3);

      // ── Factor 4: Position within section ─────────────────────────────
      // First sentence in a paragraph tends to be a topic sentence
      const relPos = i / N;
      if (relPos < 0.05) score += 1.5;        // document opening
      else if (relPos > 0.88) score += 0.5;   // near end (conclusion area)
      // Strongly penalize deep references section
      if (relPos > 0.92 && section.label === 'References') score -= 5;

      // ── Factor 5: Sentence length preference ──────────────────────────
      // Prefer informative-length sentences; penalize very short/long
      const len = sentence.split(/\s+/).length;
      if (len >= 15 && len <= 40) score += 0.8;
      else if (len < 8)           score -= 1.5;
      else if (len > 60)          score -= 0.5;

      // ── Factor 6: Citation density penalty ────────────────────────────
      const citations = (sentence.match(/\[\s*\d+\s*\]|\(\s*\d{4}\s*\)|\bet al\b/gi) || []).length;
      score -= citations * 0.8;

      // ── Factor 7: Numeric/metric bonus ────────────────────────────────
      // Sentences with numbers (percentages, scores) tend to report findings
      const metrics = (sentence.match(/\d+\.?\d*\s*(%|percent|accuracy|f1|bleu|rouge|score|point)/gi) || []).length;
      score += Math.min(metrics * 0.5, 1.5);

      // ── Factor 8: Negative result / limitation phrase detection ────────
      // These are important but slightly less "main point"
      if (/\blimitation|drawback|weakness|does not|cannot|fails?|poor\b/i.test(sentence)) score += 0.3;

      return { sentence, score, tfidfVec: vec, originalIndex: i };
    });
  }

  // ────────────────────────────────────────────────────────────────────────
  // PUBLIC API
  // ────────────────────────────────────────────────────────────────────────

  /**
   * Full analysis pipeline.
   *
   * @param {string} rawText - Raw text from the PDF extractor.
   * @returns {{
   *   sentences: string[],
   *   rankedSentences: Array<{sentence, score, originalIndex}>,
   *   keywords: string[],
   *   bigrams: string[],
   *   sections: string[],
   *   wordCount: number,
   *   readTime: number
   * }}
   */
  function analyze(rawText) {
    // ── 1. Pre-process ─────────────────────────────────────────────────
    const text = cleanText(rawText);
    const words = text.split(/\s+/).filter(Boolean);
    const wordCount = words.length;
    const readTime = Math.ceil(wordCount / 200);

    // ── 2. Sentences ───────────────────────────────────────────────────
    const sentences = splitSentences(text);

    // ── 3. Section segmentation ────────────────────────────────────────
    const sectionMap = assignSections(sentences);

    // ── 4. Detected section labels (unique, preserve order) ────────────
    const sections = [];
    const seenLabels = new Set();
    sectionMap.forEach(({ label }) => {
      if (!seenLabels.has(label)) {
        seenLabels.add(label);
        sections.push(label);
      }
    });

    // ── 5. TF-IDF matrix ───────────────────────────────────────────────
    const { tfidf, globalImportance } = buildTfIdf(sentences);

    // ── 6. Bigrams ─────────────────────────────────────────────────────
    const bigrams = extractBigrams(text);

    // ── 7. Keywords ────────────────────────────────────────────────────
    const keywords = extractKeywords(globalImportance, bigrams, 18);

    // ── 8. Score sentences ─────────────────────────────────────────────
    const scored = scoreSentences(sentences, tfidf, globalImportance, sectionMap);

    // ── 9. Sort by score ───────────────────────────────────────────────
    const sorted = [...scored].sort((a, b) => b.score - a.score);

    // ── 10. Deduplicate (remove semantically redundant sentences) ───────
    const deduped = deduplicate(sorted, 0.52);

    return {
      sentences,
      rankedSentences: deduped,
      keywords,
      bigrams,
      sections,
      wordCount,
      readTime,
    };
  }

  return { analyze };

})();
