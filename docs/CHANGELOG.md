# Changelog

All notable changes to PaperLens are documented here.

---

## [2.0.0] — Accuracy Overhaul

### nlpAnalyzer.js — Complete Rewrite

**Problem with v1:** Simple term-frequency counting rewarded common words,
not important ones. Position heuristics were too coarse. No deduplication
meant the top-N often contained near-identical sentences.

**What changed:**

#### True TF-IDF (replaces raw frequency)
- Each term is now scored by TF × IDF across all sentences as "documents"
- Rare-but-important terms (e.g. a paper's unique method name) now score
  much higher than common academic words that appear everywhere
- Sentence TF is normalized by sentence length so long sentences don't
  dominate purely by having more words

#### Bigram extraction
- Extracts 2-word phrases ("deep learning", "attention mechanism", etc.)
- Bigrams that appear ≥ 2 times are promoted above single-word keywords
- Deduplication removes unigrams already captured by a bigram

#### Light stemming
- Unifies word variants: "proposes"/"proposed"/"proposal" → single stem
- Prevents TF-IDF from fragmenting signal across morphological variants

#### Section-aware sentence scoring (8 factors)
1. **TF-IDF weight** — core relevance signal
2. **Section weight** — Abstract (3.0×) and Conclusion (3.0×) boosted;
   References (0.0×) effectively excluded
3. **Signal phrase detection** — 20 patterns like "we propose", "we achieve",
   "state-of-the-art" add targeted boosts
4. **Position within document** — opening sentences get a boost; deep
   references section gets a strong penalty
5. **Sentence length** — 15–40 word sentences preferred; very short/long penalized
6. **Citation density** — each citation marker reduces score by 0.8
7. **Metric detection** — sentences reporting numbers, percentages, scores get a boost
8. **Limitation detection** — weakness statements preserved (important context)

#### Cosine-similarity deduplication (new)
- After scoring, sentences are compared pairwise using cosine similarity
  on their TF-IDF vectors
- If two sentences are > 52% similar, only the higher-scored one is kept
- Eliminates the problem of the summary repeating the same idea twice

#### Improved sentence segmentation
- Protects abbreviations (Fig., et al., e.g., Dr.) from false splits
- Protects decimal numbers from false splits
- Additional noise filters: URLs, page headers, lone numbers, non-ASCII runs

#### Text pre-processing
- Fixes hyphenated line-break artifacts from PDF extraction
- Normalises smart quotes, em-dashes, form feeds

---

### outputGenerator.js — Coherence & Structure

#### Summary mode
- Sentences are grouped into paragraphs by proximity (consecutive sentences
  from nearby document positions stay together)
- Stray citation brackets cleaned from output

#### ELI5 mode
- Expanded jargon dictionary from 20 → 40+ terms
- Output now split into two labelled sections: "What is this about?" and
  "What did they find?" for better structure

#### Bullets mode
- Citations cleaned from each bullet
- Ordering preserved by document position

#### All modes
- `originalIndex` tracked through the pipeline so re-ordering is stable
- `cleanSentence()` helper strips residual citations from all outputs

---

## [1.0.0] — Initial Release

- Basic TF-IDF keyword extraction (raw frequency)
- Simple position + keyword density sentence scoring
- 4 output modes: summary, bullets, ELI5, abstract
- PDF.js integration
- Corporate project structure (HTML / CSS modules / JS modules)
