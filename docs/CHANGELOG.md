# Changelog

All notable changes to PaperLens are documented here.

---

## [3.0.0] — Deep Accuracy Overhaul

### nlpAnalyzer.js

#### TextRank graph algorithm (new)
- Builds a full sentence similarity graph using cosine distance between TF-IDF vectors
- Runs power-iteration PageRank on the graph: sentences that are similar to *many other sentences* rise to the top
- This captures "centrality to the document's core topic" — something TF-IDF alone cannot do
- Combined with TF-IDF score for a hybrid relevance × centrality ranking

#### Maximal Marginal Relevance (MMR) selection (new, replaces cosine dedup)
- MMR explicitly balances relevance vs. diversity: `score = λ × relevance − (1−λ) × similarity_to_selected`
- Each sentence selected is the one that adds maximum *new information* relative to what's already chosen
- λ = 0.65 (65% relevance, 35% diversity) — tuned for academic summarization

#### Trigrams added to n-gram extraction
- Now extracts both bigrams and trigrams ("convolutional neural network", "transfer learning approach")
- Trigrams are preferred over bigrams when they cover the same words (more specific)
- Displayed unstemmed so keywords look natural ("learning" not "learn")

#### Robust section detection (complete rewrite)
- Two-pass algorithm: scan raw text line-by-line for heading patterns, then assign sentences by character position
- Handles numbered headings ("3.2 Methodology", "IV. Results")
- 20 section types defined with weights (Abstract: 3.2×, Conclusion: 3.2×, References: −1.0×)
- Falls back to positional heuristics when no headings found

#### Score normalization + confidence estimation (new)
- All scores normalized to [0,1] via min-max after full scoring
- Confidence = gap between top-N average and rest-of-document average
- Shown as a progress bar in the output UI

#### Improved TF-IDF
- Log-normalized TF: `(1 + log(tf)) / docLen` — reduces length bias further
- Global importance now uses **max** across sentences (not sum) — prevents terms appearing in 50 sentences from dominating over paper-specific terms appearing in 5

#### Scoring expanded to 10 factors
- New: F2 TextRank centrality (weighted ×8.0)
- New: F9 Comparison/contrast phrase detection
- New: F10 Structural coherence (capitalization, punctuation checks)
- Improved: F7 Citation regex now catches `(Smith, 2021)` author-year style
- Improved: F8 Metric detection covers more units (pp, AUC, BLEU, ROUGE)
- Improved: F5 References/Acknowledgements sections get −10 penalty (vs −5)

#### Noise filter bug fix
- `/g` flag removed from `DISQUALIFIERS` patterns — was causing the JS sticky `lastIndex` bug where alternating sentences would pass/fail the same test incorrectly

#### Stemmer improved
- Handles more suffix patterns: -ational, -tional, -enci, -anci, -alism, -iveness, -fulness, -ousness, -aliti, -iviti, -biliti

---

### outputGenerator.js

#### Abstract mode fixed
- v2 `bySection` logic built a map but never used it — fell through to `topOrdered`
- v3 genuinely selects one sentence per IMRaD group (Context / Method / Results / Discussion)
- Labels each part ("Background:", "Method:", "Results:", "Conclusion:") inline
- Falls back to top-N ordered sentences if section detection failed

#### ELI5 split changed from positional to content-driven
- v2: split first half = "what", second half = "findings" (wrong for most papers)
- v3: sentences categorized by detected section label into context vs. findings groups
- Two color-coded blocks with emoji labels; fallback to positional split if sections unclear

#### Bullets mode: tiered display
- Top 40% scoring sentences shown in bold
- Section label shown as a small tag after each bullet
- Helps user see which section each point comes from

#### Summary mode: section transitions
- Each paragraph group labeled with its section type ("Results:", "Conclusion:", etc.)
- Makes the summary scannable without reading every word

#### Confidence bar (new)
- Shown at the top of every output
- Green (High) / Amber (Medium) / Grey (Low) with percentage
- Tells user how cleanly the algorithm was able to distinguish key sentences

#### ELI5 dictionary: 20 → 70+ entries
- Added ML/NLP-specific terms: attention, transformer, embedding, backpropagation, epoch, overfitting, etc.
- Added system terms: scalable, robust, throughput, latency
- Added academic verbs: demonstrate, investigate, facilitate, leverage, utilize

---

## [2.0.0] — Accuracy Overhaul (see previous entry)

## [1.0.0] — Initial Release


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
