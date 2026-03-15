# PaperLens — Research Paper Simplifier

> Upload a research paper PDF → get an instant simplified summary.  
> **100% local. No internet. No API. Runs in your browser.**

---

## Project Info

| Field | Details |
|---|---|
| College | Shri Ramdeobaba College of Engineering & Management, Nagpur |
| Department | Data Science |
| Session | 2025–26, VI Sem C |
| Students | Mit Paliwal (72), Vaishnavi Bagekar (73) |

---

## What It Does

1. User uploads a research paper PDF
2. System extracts and cleans the text
3. NLP pipeline scores every sentence
4. Best sentences selected using MMR algorithm
5. Summary displayed instantly in the browser

---

## Results — Model Comparison

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Composite |
|---|---|---|---|---|
| **MMR (lambda=0.6)** ★ | **0.4216** | **0.1366** | **0.2130** | **0.2697** |
| Weighted Tuned (n=6) | 0.4112 | 0.1380 | 0.2140 | 0.2663 |
| Weighted Tuned (n=5) | 0.4056 | 0.1347 | 0.2145 | 0.2630 |
| Weighted (original) | 0.3997 | 0.1308 | 0.2129 | 0.2589 |
| TextRank | 0.3951 | 0.1316 | 0.2124 | 0.2572 |
| TF-IDF Extractive | 0.3904 | 0.1307 | 0.2112 | 0.2547 |
| LexRank | 0.3797 | 0.1241 | 0.2090 | 0.2476 |
| BART (Abstractive) | 0.3217 | 0.0944 | 0.1851 | 0.2080 |
| LSA | 0.2715 | 0.0717 | 0.1603 | 0.1738 |

**Best Model: MMR with lambda=0.6, n=6 sentences**  
**Improvement: 0.3997 → 0.4216 (+2.2 ROUGE-1 points)**

---

## Project Structure
```
PaperLens/
├── src/
│   ├── step1_dataset.py           # ArXiv dataset loading & exploration
│   ├── step2_preprocessing.py     # Text cleaning pipeline
│   ├── step3_features.py          # TF-IDF, position, length, keyword features
│   ├── step4_5_train_evaluate.py  # 5 summarization models + ROUGE evaluation
│   ├── step5b_tune_weights.py     # Weight tuning via grid search
│   ├── step5d_mmr.py              # MMR summarizer + lambda tuning
│   └── step6_best_model.py        # Best model selection
├── gui/
│   └── index.html                 # Web-based GUI (no install needed)
├── dataset/                       # Preprocessed data cache
├── models/                        # Saved model configs
├── outputs/                       # Charts, CSVs, ROUGE results
├── report/                        # Project report
└── requirements.txt
```

---

## NLP Pipeline Steps

| Step | Description | Output |
|---|---|---|
| 1 | Dataset loading — 10,000 ArXiv papers | Distribution charts |
| 2 | Text preprocessing — citations, LaTeX, stopwords | 19.8% token reduction |
| 3 | Feature extraction — TF-IDF, position, length, keyword | Feature distributions |
| 4&5 | Train 5 models + ROUGE evaluation | Model comparison table |
| 5b | Weight tuning — grid search over 68 combinations | Best: TF-IDF=0.5, Pos=0.3 |
| 5d | MMR summarizer + lambda tuning | ROUGE-1: 0.4216 |
| 6 | Best model selection — composite ROUGE score | MMR selected |
| 7 | GUI — web app, PDF upload, instant summary | index.html |

---

## How to Run

### Python Pipeline (generate all results)
```bash
pip install -r requirements.txt
cd src
python step1_dataset.py
python step2_preprocessing.py
python step3_features.py
python step4_5_train_evaluate.py
python step5b_tune_weights.py
python step5d_mmr.py
python step6_best_model.py
```

### GUI (web app)
```bash
cd gui
python -m http.server 3000
```
Open browser → `http://localhost:3000`

---

## Dataset

**ArXiv Summarization** (HuggingFace: ccdv/arxiv-summarization)
- 10,000 training papers
- 2,000 validation papers
- Mean article length: 5,998 words
- Mean abstract length: 275 words

---

## Tech Stack

| Technology | Purpose |
|---|---|
| scikit-learn | TF-IDF vectorization, cosine similarity |
| NLTK | Tokenization, stopwords, lemmatization |
| datasets (HuggingFace) | ArXiv dataset loading |
| rouge-score | ROUGE evaluation |
| PyMuPDF | PDF processing |
| PDF.js | Browser-side PDF extraction |
| HTML/CSS/JS | GUI frontend |

---

## GitHub

[https://github.com/VBagekar/PaperLens](https://github.com/VBagekar/PaperLens)
```
