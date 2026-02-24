/* =========================================================
   nlpAnalyzer.js — NLP Analysis Engine v3
   TF-IDF × TextRank hybrid, n-gram extraction,
   robust section detection, score normalization,
   confidence estimation, MMR diversity selection.
   PaperLens v3.0.0
   ========================================================= */

'use strict';

/**
 * @module NlpAnalyzer
 * Full pipeline:
 *  1.  Pre-processing   — clean text, fix PDF artifacts
 *  2.  Sentence split   — abbreviation-aware boundary detection
 *  3.  Section detect   — numbered headings + keyword patterns
 *  4.  TF-IDF matrix    — per-sentence, length-normalized, smoothed IDF
 *  5.  TextRank         — graph-based centrality (sentences vote for each other)
 *  6.  N-gram extract   — bigrams + trigrams, unstemmed for display
 *  7.  Sentence score   — 10-factor weighted combination
 *  8.  Score normalize  — min-max to [0,1], compute confidence
 *  9.  MMR selection    — Maximal Marginal Relevance for diversity
 *  10. Output           — ranked, diverse, position-ordered sentences
 */
const NlpAnalyzer = (() => {

  // ── Stopwords ─────────────────────────────────────────────────────────────
  const STOPWORDS = new Set([
    'the','a','an','and','or','but','in','on','at','to','for','of','with',
    'by','from','as','is','was','are','were','be','been','being','have',
    'has','had','do','does','did','will','would','could','should','may',
    'might','shall','must','need','dare','ought','let','get','got','gets',
    'this','that','these','those','it','its','we','our','they','their',
    'i','my','me','you','your','he','his','she','her','him','us','them',
    'which','who','what','when','where','how','all','also','more','can',
    'not','paper','papers','show','shows','shown','using','used','use',
    'based','two','one','three','four','five','six','seven','eight','nine',
    'ten','each','than','such','into','through','between','after','during',
    'figure','figures','table','tables','al','et','ie','eg','viz','however',
    'proposed','present','presented','presents','study','research','results',
    'result','data','new','model','models','approach','method','system',
    'work','section','first','second','third','fourth','fifth','both',
    'since','while','given','well','thus','hence','note','other','others',
    'several','many','most','some','any','few','very','quite','rather',
    'much','further','also','even','just','only','already','still','now',
    'then','here','there','where','so','if','about','up','down','over',
    'under','again','own','same','different','various','across','within',
    'without','along','upon','around','against','toward','towards','per',
    'set','sets','make','makes','made','take','takes','taken','give','gives',
    'given','large','small','high','low','number','numbers','type','types',
    'form','forms','case','cases','point','points','example','examples',
    'way','ways','order','terms','provide','provides','provided','consider',
    'considered','show','shown','note','noted','recently','often','typically',
    'generally','usually','commonly','recently','previously','subsequently',
    'therefore','moreover','furthermore','additionally','consequently',
    'nevertheless','nonetheless','whereas','whereby','thereby','therein',
    'herein','wherein','thereof','hence','thus','whether','either','neither',
    'both','each','every','throughout','according','due','related','able',
    'unable','likely','unlikely','possible','impossible','important','used',
    'known','called','named','defined','described','referred'
  ]);

  // ── High-value signal phrases ─────────────────────────────────────────────
  // Each entry: [regex, weight] — weight added to sentence score
  const SIGNAL_PHRASES = [
    [/\bwe\s+(propose|present|introduce|develop|demonstrate)\b/i,          1.8],
    [/\bwe\s+(find|found|show|showed|report|observe|observed)\b/i,         1.5],
    [/\bwe\s+(achieve|outperform|surpass|improve|reduce|increase)\b/i,     1.6],
    [/\bour\s+(approach|method|model|system|framework|architecture)\b/i,   1.4],
    [/\bour\s+(contribution|contributions|results?|findings?)\b/i,         1.4],
    [/\bin\s+this\s+(paper|work|study|article|thesis)\b/i,                 1.3],
    [/\b(main|key|primary|central|novel|core)\s+(contribution|finding)\b/i,1.5],
    [/\bstate.of.the.art\b/i,                                              1.6],
    [/\bsignificantly?\s+(better|worse|higher|lower|outperforms?)\b/i,     1.4],
    [/\bnovel\s+(approach|method|framework|technique|algorithm)\b/i,       1.3],
    [/\bto\s+(address|solve|tackle|overcome|handle)\b/i,                   1.0],
    [/\bresults?\s+(show|indicate|suggest|demonstrate|reveal)\b/i,         1.3],
    [/\bcompared\s+(to|with|against)\b/i,                                  0.8],
    [/\bbaseline\b/i,                                                       0.7],
    [/\b(superior|inferior|competitive)\s+(performance|results?)\b/i,      1.2],
    [/\bwe\s+conclude\b/i,                                                  1.5],
    [/\bin\s+conclusion\b/i,                                                1.5],
    [/\bfuture\s+work\b/i,                                                  0.9],
    [/\blimitation\b/i,                                                     0.8],
    [/\bablation\s+stud/i,                                                  1.0],
    [/\bstatistically\s+significant\b/i,                                    1.3],
    [/\bp\s*[<=>]\s*0\.\d+\b/,                                              1.2], // p-values
    [/\b\d+\.?\d*\s*(%|percent)\s+(improvement|reduction|increase|gain)\b/i,1.4],
    [/\berror\s+rate\b/i,                                                   0.9],
    [/\brobust(ness)?\b/i,                                                  0.8],
    [/\bscalable|scalability\b/i,                                           0.8],
    [/\befficient(ly)?\b/i,                                                 0.7],
  ];

  // ── Section heading patterns (ordered by priority) ────────────────────────
  // pattern: matches the heading text; numPattern: matches numbered headings
  const SECTION_DEFS = [
    { label: 'Abstract',              weight: 3.2, pattern: /^abstract$/i },
    { label: 'Introduction',          weight: 2.6, pattern: /^(\d[\.\d]*\s+)?introduction$/i },
    { label: 'Background',            weight: 1.4, pattern: /^(\d[\.\d]*\s+)?background$/i },
    { label: 'Related Work',          weight: 1.0, pattern: /^(\d[\.\d]*\s+)?related\s+work$/i },
    { label: 'Literature Review',     weight: 1.0, pattern: /^(\d[\.\d]*\s+)?literature\s+review$/i },
    { label: 'Problem Statement',     weight: 1.8, pattern: /^(\d[\.\d]*\s+)?problem\s+(statement|formulation|definition)$/i },
    { label: 'Methodology',           weight: 1.6, pattern: /^(\d[\.\d]*\s+)?(methodology|methods?|approach|proposed\s+method)$/i },
    { label: 'Model / Architecture',  weight: 1.5, pattern: /^(\d[\.\d]*\s+)?(model|architecture|framework|system\s+design)$/i },
    { label: 'Dataset',               weight: 1.2, pattern: /^(\d[\.\d]*\s+)?(dataset|data\s+collection|corpus)$/i },
    { label: 'Experiments',           weight: 2.0, pattern: /^(\d[\.\d]*\s+)?(experiments?|experimental\s+setup|experimental\s+results?)$/i },
    { label: 'Implementation',        weight: 1.1, pattern: /^(\d[\.\d]*\s+)?implementation(\s+details?)?$/i },
    { label: 'Evaluation',            weight: 2.0, pattern: /^(\d[\.\d]*\s+)?(evaluation|benchmark|comparison)$/i },
    { label: 'Results',               weight: 2.2, pattern: /^(\d[\.\d]*\s+)?(results?|findings?|analysis)$/i },
    { label: 'Discussion',            weight: 2.0, pattern: /^(\d[\.\d]*\s+)?discussion$/i },
    { label: 'Ablation Study',        weight: 1.4, pattern: /^(\d[\.\d]*\s+)?ablation\s+(study|analysis)$/i },
    { label: 'Conclusion',            weight: 3.2, pattern: /^(\d[\.\d]*\s+)?(conclusion|conclusions?|concluding\s+remarks?)$/i },
    { label: 'Future Work',           weight: 1.4, pattern: /^(\d[\.\d]*\s+)?future\s+work$/i },
    { label: 'References',            weight: -1.0, pattern: /^(\d[\.\d]*\s+)?(references?|bibliography|works?\s+cited)$/i },
    { label: 'Appendix',              weight: 0.1, pattern: /^(\d[\.\d]*\s+)?(appendix|appendices|supplementary)$/i },
    { label: 'Acknowledgements',      weight: -2.0, pattern: /^(\d[\.\d]*\s+)?(acknowledgements?|funding|conflict\s+of\s+interest)$/i },
  ];

  // ── Noise patterns (do not use /g flag — avoids lastIndex sticky bug) ─────
  const DISQUALIFIERS = [
    s => /^\s*\d+\s*$/.test(s),                            // lone page number
    s => /^(http|www\.)/i.test(s),                          // URL
    s => /copyright|all\s+rights\s+reserved/i.test(s),     // legal
    s => /^(acknowledgement|funding|conflict|disclosure)/i.test(s),
    s => /[^\x00-\x7F]{5,}/.test(s),                       // garbled non-ASCII
    s => /(.)\1{6,}/.test(s),                               // repeated chars
    s => (s.match(/\[\d+\]/g) || []).length > 4,            // citation spam >4
    s => /^\s*(figure|fig\.?|table|tab\.?)\s+\d/i.test(s), // caption
    s => s.split(/\s+/).length < 6,                          // too short
    s => /^[A-Z\s,\-\.]{20,}$/.test(s),                    // ALL CAPS line (header)
    s => /\b(click here|doi:|arxiv:|isbn:|issn:)\b/i.test(s),
  ];

  // ════════════════════════════════════════════════════════════════════════
  // 1. PRE-PROCESSING
  // ════════════════════════════════════════════════════════════════════════

  function cleanText(raw) {
    return raw
      // Fix hyphenated word-wraps at line breaks
      .replace(/(\w)-\s*\n\s*(\w)/g, '$1$2')
      // Remove running headers/footers: short all-digit or short all-cap lines
      .replace(/^[ \t]*\d{1,4}[ \t]*$/gm, '')
      // Collapse 3+ blank lines
      .replace(/\n{3,}/g, '\n\n')
      // Collapse horizontal whitespace
      .replace(/[ \t]+/g, ' ')
      // Normalize unicode punctuation
      .replace(/[\u2018\u2019]/g, "'")
      .replace(/[\u201C\u201D]/g, '"')
      .replace(/[\u2013\u2014]/g, ' – ')
      .replace(/\u2026/g, '...')
      // Remove form feeds
      .replace(/\f/g, '\n')
      // Fix spacing after periods that got merged (pdfjs artifact)
      .replace(/([a-z])\.([A-Z])/g, '$1. $2')
      .trim();
  }

  // ════════════════════════════════════════════════════════════════════════
  // 2. SENTENCE SPLITTING
  // ════════════════════════════════════════════════════════════════════════

  function splitSentences(text) {
    // Protect non-boundary periods
    let s = text
      .replace(/\b(Fig|Tab|Eq|Sec|Ref|vs|e\.g|i\.e|cf|approx|est|ca|al|Dr|Mr|Mrs|Prof|Jr|Sr|No|Vol|pp|dept|univ|corp|inc|ltd)\./gi,
               m => m.slice(0, -1) + '‹DOT›')
      .replace(/\b\d+\.\d+/g,  m => m.replace('.', '‹DOT›'))   // decimals
      .replace(/\b[A-Z]{1,2}\./g, m => m.replace('.', '‹DOT›')) // initials
      .replace(/\.\s*\d+\s*[,\)]/g, m => m.replace('.', '‹DOT›')); // e.g. 3.4)

    const raw = s
      .replace(/\n+/g, ' ')
      .split(/(?<=[.!?])\s+(?=[A-Z"'(\d])/)
      .map(seg => seg.replace(/‹DOT›/g, '.').trim())
      .filter(seg => {
        if (seg.length < 40 || seg.length > 700) return false;
        if (!/[a-zA-Z]{3,}/.test(seg)) return false;
        return !DISQUALIFIERS.some(fn => fn(seg));
      });

    return raw;
  }

  // ════════════════════════════════════════════════════════════════════════
  // 3. SECTION DETECTION (robust to numbered headings and inline headings)
  // ════════════════════════════════════════════════════════════════════════

  /**
   * Two-pass section detection:
   *  Pass 1: scan raw text lines for heading-like lines (short, possibly numbered)
   *          and record their character positions.
   *  Pass 2: for each sentence, binary-search the heading list to assign section.
   */
  function detectSections(rawText, sentences) {
    // Find heading positions in the raw text
    const lines = rawText.split('\n');
    const headings = []; // { pos, label, weight }
    let charPos = 0;

    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.length > 0 && trimmed.length < 80) {
        for (const def of SECTION_DEFS) {
          if (def.pattern.test(trimmed)) {
            headings.push({ pos: charPos, label: def.label, weight: def.weight });
            break;
          }
        }
      }
      charPos += line.length + 1;
    }

    // If no headings found, fall back to positional heuristics
    if (headings.length === 0) {
      return sentences.map((_, i) => {
        const rel = i / sentences.length;
        if (rel < 0.08) return { label: 'Abstract/Intro', weight: 2.8 };
        if (rel > 0.88) return { label: 'Conclusion',     weight: 3.0 };
        return { label: 'Body', weight: 1.2 };
      });
    }

    // Map each sentence to its section by finding its position in raw text
    const textLower = rawText.toLowerCase();
    return sentences.map(sent => {
      // Find approximate position of sentence in document
      const idx = textLower.indexOf(sent.slice(0, 40).toLowerCase());
      if (idx === -1) return { label: 'Body', weight: 1.2 };

      // Find the last heading before this position
      let section = { label: 'Preamble', weight: 2.0 };
      for (const h of headings) {
        if (h.pos <= idx) section = h;
        else break;
      }
      return { label: section.label, weight: section.weight };
    });
  }

  // ════════════════════════════════════════════════════════════════════════
  // 4. TOKENIZATION & STEMMING
  // ════════════════════════════════════════════════════════════════════════

  function tokenize(text) {
    return text
      .toLowerCase()
      .split(/[\s\-–—\/]+/)
      .map(w => w.replace(/[^a-z]/g, ''))
      .filter(w => w.length > 2 && !STOPWORDS.has(w));
  }

  /**
   * Lightweight rule-based stemmer (covers most English morphology).
   * Keeps words recognizable — not an aggressive Porter stemmer.
   */
  function stem(word) {
    if (word.length < 5) return word;
    return word
      .replace(/ational$/, 'ate')
      .replace(/tional$/, 'tion')
      .replace(/enci$/, 'ence')
      .replace(/anci$/, 'ance')
      .replace(/izing$/, 'ize')
      .replace(/ising$/, 'ise')
      .replace(/izing$/, 'ize')
      .replace(/ations?$/, 'ate')
      .replace(/ators?$/, 'ate')
      .replace(/alism$/, 'al')
      .replace(/iveness$/, 'ive')
      .replace(/fulness$/, 'ful')
      .replace(/ousness$/, 'ous')
      .replace(/aliti$/, 'al')
      .replace(/iviti$/, 'ive')
      .replace(/biliti$/, 'ble')
      .replace(/icate$/, 'ic')
      .replace(/alize$/, 'al')
      .replace(/iciti$/, 'ic')
      .replace(/ical$/, 'ic')
      .replace(/fulness$/, 'ful')
      .replace(/ness$/, '')
      .replace(/ment$/, '')
      .replace(/ings?$/, '')
      .replace(/ities$/, 'ity')
      .replace(/iers?$/, 'y')
      .replace(/ied$/, 'y')
      .replace(/ies$/, 'y')
      .replace(/ers?$/, '')
      .replace(/ed$/, '')
      .replace(/ly$/, '')
      .replace(/s$/, '');
  }

  // ════════════════════════════════════════════════════════════════════════
  // 5. TF-IDF MATRIX
  // ════════════════════════════════════════════════════════════════════════

  function buildTfIdf(sentences) {
    const N = sentences.length;

    const sentenceData = sentences.map(s => {
      const tokens = tokenize(s).map(stem);
      const tf = new Map();
      for (const t of tokens) tf.set(t, (tf.get(t) || 0) + 1);
      const len = tokens.length || 1;
      // Log-normalized TF: (1 + log(tf)) / docLen
      tf.forEach((v, k) => tf.set(k, (1 + Math.log(v)) / len));
      return { tf, tokenSet: new Set(tokens) };
    });

    // Document frequency
    const df = new Map();
    for (const { tokenSet } of sentenceData) {
      for (const t of tokenSet) df.set(t, (df.get(t) || 0) + 1);
    }

    // Smoothed IDF: log((N+1)/(df+1)) + 1
    const tfidfVecs = sentenceData.map(({ tf }) => {
      const vec = new Map();
      tf.forEach((tfVal, term) => {
        const idf = Math.log((N + 1) / ((df.get(term) || 0) + 1)) + 1;
        vec.set(term, tfVal * idf);
      });
      return vec;
    });

    // Global term importance = max TF-IDF score across all sentences
    // (max better than sum — prevents common terms in many sentences from dominating)
    const globalImportance = new Map();
    for (const vec of tfidfVecs) {
      vec.forEach((val, term) => {
        if (val > (globalImportance.get(term) || 0)) {
          globalImportance.set(term, val);
        }
      });
    }

    return { tfidfVecs, globalImportance };
  }

  // ════════════════════════════════════════════════════════════════════════
  // 6. TEXTRANK (graph-based sentence centrality)
  // ════════════════════════════════════════════════════════════════════════

  function cosine(a, b) {
    let dot = 0, na = 0, nb = 0;
    a.forEach((v, k) => { dot += v * (b.get(k) || 0); na += v * v; });
    b.forEach(v => { nb += v * v; });
    return (na === 0 || nb === 0) ? 0 : dot / (Math.sqrt(na) * Math.sqrt(nb));
  }

  /**
   * TextRank: build a similarity graph between sentences,
   * then run power iteration to find the most "central" sentences.
   * Sentences that are similar to many other sentences rank higher.
   *
   * @param {Map<string,number>[]} vecs  - TF-IDF vectors per sentence
   * @param {number} iterations
   * @param {number} damping
   * @returns {number[]} TextRank scores, one per sentence
   */
  function textRank(vecs, iterations = 30, damping = 0.85) {
    const N = vecs.length;
    if (N === 0) return [];

    // Build similarity matrix (only upper triangle, symmetric)
    // For performance: cap at 300 sentences
    const cap = Math.min(N, 300);
    const sim = Array.from({ length: cap }, () => new Float32Array(cap));

    for (let i = 0; i < cap; i++) {
      for (let j = i + 1; j < cap; j++) {
        const s = cosine(vecs[i], vecs[j]);
        sim[i][j] = s;
        sim[j][i] = s;
      }
    }

    // Row-normalize (each row sums to 1)
    const normSim = sim.map(row => {
      const total = row.reduce((a, b) => a + b, 0);
      return total === 0 ? row : row.map(v => v / total);
    });

    // Power iteration
    let scores = new Float32Array(cap).fill(1 / cap);
    for (let iter = 0; iter < iterations; iter++) {
      const next = new Float32Array(cap).fill((1 - damping) / cap);
      for (let i = 0; i < cap; i++) {
        for (let j = 0; j < cap; j++) {
          next[j] += damping * normSim[i][j] * scores[i];
        }
      }
      scores = next;
    }

    // Extend with 0 for any sentences beyond cap
    const result = Array.from(scores);
    for (let i = cap; i < N; i++) result.push(0);
    return result;
  }

  // ════════════════════════════════════════════════════════════════════════
  // 7. N-GRAM EXTRACTION (bigrams + trigrams, unstemmed for display)
  // ════════════════════════════════════════════════════════════════════════

  function extractNgrams(text, minCount = 2) {
    // Use unstemmed tokens for display quality
    const tokens = tokenize(text); // not stemmed
    const counts = new Map();

    // Bigrams
    for (let i = 0; i < tokens.length - 1; i++) {
      if (tokens[i].length < 4 || tokens[i+1].length < 4) continue;
      const bg = `${tokens[i]} ${tokens[i+1]}`;
      counts.set(bg, (counts.get(bg) || 0) + 1);
    }

    // Trigrams
    for (let i = 0; i < tokens.length - 2; i++) {
      if (tokens[i].length < 4 || tokens[i+1].length < 3 || tokens[i+2].length < 4) continue;
      const tg = `${tokens[i]} ${tokens[i+1]} ${tokens[i+2]}`;
      counts.set(tg, (counts.get(tg) || 0) + 1);
    }

    // Filter by min count and sort by frequency
    const ngrams = [...counts.entries()]
      .filter(([, c]) => c >= minCount)
      .sort((a, b) => {
        // Prefer trigrams (more specific) over bigrams of same count
        const aWords = a[0].split(' ').length;
        const bWords = b[0].split(' ').length;
        if (b[1] !== a[1]) return b[1] - a[1];
        return bWords - aWords;
      });

    // Deduplicate: remove bigrams already covered by a trigram
    const selected = [];
    const covered = new Set();

    for (const [phrase] of ngrams) {
      if (covered.has(phrase)) continue;
      selected.push(phrase);
      // Mark sub-phrases as covered
      const parts = phrase.split(' ');
      if (parts.length === 3) {
        covered.add(`${parts[0]} ${parts[1]}`);
        covered.add(`${parts[1]} ${parts[2]}`);
      }
      if (selected.length >= 12) break;
    }

    return selected;
  }

  // ════════════════════════════════════════════════════════════════════════
  // 8. KEYWORD EXTRACTION (TF-IDF global importance, unstemmed display)
  // ════════════════════════════════════════════════════════════════════════

  function extractKeywords(text, globalImportance, ngrams, topN = 15) {
    // Compute global importance for UNSTEMMED tokens (for display)
    const rawTokens = tokenize(text);
    const rawFreq = new Map();
    for (const t of rawTokens) rawFreq.set(t, (rawFreq.get(t) || 0) + 1);

    // Map stemmed importance back to the most frequent unstemmed form
    const stemToRaw = new Map();
    for (const [raw, freq] of rawFreq) {
      const s = stem(raw);
      if (!stemToRaw.has(s) || freq > rawFreq.get(stemToRaw.get(s))) {
        stemToRaw.set(s, raw);
      }
    }

    const unigrams = [...globalImportance.entries()]
      .filter(([term]) => term.length > 3)
      .sort((a, b) => b[1] - a[1])
      .slice(0, topN * 2)
      .map(([stem_]) => stemToRaw.get(stem_) || stem_)
      .filter(Boolean);

    // Combine: n-grams first (more specific), then unigrams
    const ngramWords = new Set(ngrams.join(' ').split(' '));
    const filteredUni = unigrams.filter(u => !ngramWords.has(u));

    return [...new Set([...ngrams, ...filteredUni])].slice(0, topN);
  }

  // ════════════════════════════════════════════════════════════════════════
  // 9. SENTENCE SCORING (10 factors)
  // ════════════════════════════════════════════════════════════════════════

  function scoreSentences(sentences, tfidfVecs, globalImportance, sectionInfo, textRankScores) {
    const N = sentences.length;

    return sentences.map((sent, i) => {
      const vec     = tfidfVecs[i];
      const section = sectionInfo[i] || { label: 'Body', weight: 1.2 };
      let score     = 0;

      // ── F1: TF-IDF term weight ─────────────────────────────────────────
      // Weighted sum: each term's TF-IDF × its global max importance
      let tfidfScore = 0;
      vec.forEach((v, term) => { tfidfScore += v * (globalImportance.get(term) || 0); });
      // Normalize by sentence vocabulary size to avoid length bias
      const vocabSize = vec.size || 1;
      score += Math.min((tfidfScore / vocabSize) * 3, 4.0);

      // ── F2: TextRank centrality ────────────────────────────────────────
      // Measures how "central" the sentence is to the whole document
      score += (textRankScores[i] || 0) * 8.0;  // scale to meaningful range

      // ── F3: Section weight ─────────────────────────────────────────────
      score += Math.max(section.weight * 0.7, 0);

      // ── F4: Signal phrase bonus ────────────────────────────────────────
      let signalBonus = 0;
      for (const [pat, w] of SIGNAL_PHRASES) {
        if (pat.test(sent)) signalBonus += w;
      }
      score += Math.min(signalBonus, 4.0);

      // ── F5: Position bonus ─────────────────────────────────────────────
      const relPos = i / N;
      if (relPos < 0.04)       score += 1.8;   // very first sentences
      else if (relPos < 0.10)  score += 1.0;   // early intro
      else if (relPos > 0.90)  score += 0.8;   // near conclusion
      // Hard penalty for deep references territory
      if (section.label === 'References' || section.label === 'Acknowledgements') {
        score -= 10;
      }

      // ── F6: Sentence length preference ────────────────────────────────
      const wordCount = sent.split(/\s+/).length;
      if      (wordCount >= 15 && wordCount <= 35) score += 1.0;
      else if (wordCount >= 10 && wordCount < 15)  score += 0.4;
      else if (wordCount > 35  && wordCount <= 55) score += 0.2;
      else if (wordCount < 8)                      score -= 1.5;
      else if (wordCount > 70)                     score -= 0.8;

      // ── F7: Citation penalty ───────────────────────────────────────────
      const citationCount = (sent.match(/\[\s*\d[\d,\s]*\]|\(\s*[\w]+\s*,\s*\d{4}\s*\)/g) || []).length;
      score -= citationCount * 0.6;

      // ── F8: Quantitative result bonus ──────────────────────────────────
      // Sentences reporting measurements are high-value
      const metrics = (sent.match(/\d+\.?\d*\s*(%|percent|pp\b|points?|accuracy|f1|bleu|rouge|recall|precision|auc|score)/gi) || []).length;
      score += Math.min(metrics * 0.7, 2.0);

      // ── F9: Comparison/contrast bonus ─────────────────────────────────
      if (/\b(outperform|surpass|exceed|superior|better|worse|versus|vs\.?|compared)\b/i.test(sent)) {
        score += 0.6;
      }

      // ── F10: Structural coherence — penalize sentence fragments ────────
      // Real sentences start with a capital and end with punctuation
      if (!/^[A-Z"]/.test(sent))     score -= 0.5;
      if (!/[.!?]$/.test(sent.trim())) score -= 0.3;

      return { sentence: sent, score, tfidfVec: vec, originalIndex: i, section: section.label };
    });
  }

  // ════════════════════════════════════════════════════════════════════════
  // 10. SCORE NORMALIZATION & CONFIDENCE
  // ════════════════════════════════════════════════════════════════════════

  function normalizeScores(scored) {
    const scores = scored.map(s => s.score);
    const min = Math.min(...scores);
    const max = Math.max(...scores);
    const range = max - min || 1;

    return scored.map(s => ({
      ...s,
      normalizedScore: (s.score - min) / range,
    }));
  }

  /**
   * Estimate confidence: how well-distributed the top scores are.
   * High confidence = big gap between top sentence and average.
   * @returns {number} 0–1
   */
  function estimateConfidence(normalized, topN = 10) {
    if (normalized.length === 0) return 0;
    const top = normalized.slice(0, topN).map(s => s.normalizedScore);
    const rest = normalized.slice(topN).map(s => s.normalizedScore);
    const topAvg = top.reduce((a, b) => a + b, 0) / (top.length || 1);
    const restAvg = rest.length ? rest.reduce((a, b) => a + b, 0) / rest.length : 0;
    // Confidence = gap between top and rest, capped to [0,1]
    return Math.min((topAvg - restAvg) * 2, 1.0);
  }

  // ════════════════════════════════════════════════════════════════════════
  // 11. MMR SELECTION (Maximal Marginal Relevance)
  // ════════════════════════════════════════════════════════════════════════

  /**
   * MMR balances relevance and diversity:
   *   score = λ · relevance - (1-λ) · max_similarity_to_selected
   *
   * Pure top-N by score picks redundant sentences (same idea, different words).
   * MMR ensures each selected sentence adds new information.
   *
   * @param {Array} candidates  - scored sentences with tfidfVec
   * @param {number} topN       - how many to select
   * @param {number} lambda     - 0=max diversity, 1=max relevance, 0.65=balanced
   */
  function mmrSelect(candidates, topN, lambda = 0.65) {
    if (candidates.length <= topN) return candidates;

    const selected = [];
    const remaining = [...candidates];

    while (selected.length < topN && remaining.length > 0) {
      let bestIdx = 0;
      let bestMMR = -Infinity;

      for (let i = 0; i < remaining.length; i++) {
        const relevance = remaining[i].normalizedScore;

        // Max similarity to any already-selected sentence
        let maxSim = 0;
        for (const sel of selected) {
          const sim = cosine(remaining[i].tfidfVec, sel.tfidfVec);
          if (sim > maxSim) maxSim = sim;
        }

        const mmr = lambda * relevance - (1 - lambda) * maxSim;
        if (mmr > bestMMR) {
          bestMMR = mmr;
          bestIdx = i;
        }
      }

      selected.push(remaining[bestIdx]);
      remaining.splice(bestIdx, 1);
    }

    return selected;
  }

  // ════════════════════════════════════════════════════════════════════════
  // PUBLIC API
  // ════════════════════════════════════════════════════════════════════════

  function analyze(rawText) {
    // 1. Clean
    const text = cleanText(rawText);
    const wordCount = text.split(/\s+/).filter(Boolean).length;
    const readTime  = Math.ceil(wordCount / 200);

    // 2. Sentences
    const sentences = splitSentences(text);
    if (sentences.length === 0) {
      return { sentences: [], rankedSentences: [], keywords: [], ngrams: [],
               sections: [], wordCount, readTime, confidence: 0 };
    }

    // 3. Sections
    const sectionInfo = detectSections(text, sentences);
    const sections = [...new Set(sectionInfo.map(s => s.label))];

    // 4. TF-IDF
    const { tfidfVecs, globalImportance } = buildTfIdf(sentences);

    // 5. TextRank
    const trScores = textRank(tfidfVecs);

    // 6. N-grams
    const ngrams = extractNgrams(text, 2);

    // 7. Keywords
    const keywords = extractKeywords(text, globalImportance, ngrams, 18);

    // 8. Score
    const scored = scoreSentences(sentences, tfidfVecs, globalImportance, sectionInfo, trScores);

    // 9. Sort
    const sorted = [...scored].sort((a, b) => b.score - a.score);

    // 10. Normalize + confidence
    const normalized = normalizeScores(sorted);
    const confidence = estimateConfidence(normalized);

    // 11. MMR selection (diverse top sentences)
    const diverse = mmrSelect(normalized, Math.min(20, normalized.length), 0.65);

    return {
      sentences,
      rankedSentences: diverse,
      keywords,
      ngrams,
      sections,
      wordCount,
      readTime,
      confidence,        // 0–1: how "clean" the ranking is
    };
  }

  return { analyze };

})();
