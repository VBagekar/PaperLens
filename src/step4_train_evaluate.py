"""
PaperLens - Research Paper Simplifier
Steps 4 & 5 - Training Multiple Summarization Models + Evaluation
Models:
  1. TF-IDF Extractive Summarizer
  2. TextRank Summarizer
  3. LSA (Latent Semantic Analysis) Summarizer
  4. LexRank Summarizer
  5. Weighted Feature Summarizer (our custom model)
Evaluation: ROUGE-1, ROUGE-2, ROUGE-L scores
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

os.makedirs('../outputs', exist_ok=True)
os.makedirs('../models',  exist_ok=True)

print("=" * 55)
print("STEPS 4 & 5: TRAINING + EVALUATION")
print("=" * 55)

# ── Load Data ─────────────────────────────────────────────────
print("\nLoading features...")
with open('../dataset/val_features.pkl', 'rb') as f:
    val_features = pickle.load(f)
with open('../dataset/val_abstracts.pkl', 'rb') as f:
    val_abstracts = pickle.load(f)

print(f"  Val documents : {len(val_features):,}")

# ── Summarization Models ──────────────────────────────────────

def summarize_tfidf(features, n_sentences=5):
    """TF-IDF based extractive summarization."""
    sentences    = features['sentences']
    tfidf_scores = features['tfidf_scores']
    if len(sentences) <= n_sentences:
        return ' '.join(sentences)
    ranked = sorted(zip(tfidf_scores, range(len(sentences))), reverse=True)
    top_idx = sorted([idx for _, idx in ranked[:n_sentences]])
    return ' '.join([sentences[i] for i in top_idx])


def summarize_textrank(features, n_sentences=5):
    """TextRank: graph-based sentence ranking."""
    sentences = features['sentences']
    if len(sentences) <= n_sentences:
        return ' '.join(sentences)

    # Build TF-IDF matrix
    try:
        vec    = TfidfVectorizer(stop_words='english')
        matrix = vec.fit_transform(sentences)
    except:
        return summarize_tfidf(features, n_sentences)

    # Similarity matrix
    sim_matrix = cosine_similarity(matrix)
    np.fill_diagonal(sim_matrix, 0)

    # PageRank-style scoring
    scores = np.ones(len(sentences)) / len(sentences)
    for _ in range(30):
        row_sums = sim_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1
        norm_matrix = sim_matrix / row_sums[:, np.newaxis]
        scores = 0.85 * norm_matrix.T @ scores + 0.15 / len(sentences)

    ranked  = sorted(zip(scores, range(len(sentences))), reverse=True)
    top_idx = sorted([idx for _, idx in ranked[:n_sentences]])
    return ' '.join([sentences[i] for i in top_idx])


def summarize_lsa(features, n_sentences=5):
    """LSA: Latent Semantic Analysis summarization."""
    sentences = features['sentences']
    if len(sentences) <= n_sentences:
        return ' '.join(sentences)

    try:
        vec    = TfidfVectorizer(stop_words='english', max_features=5000)
        matrix = vec.fit_transform(sentences)

        n_components = min(10, len(sentences) - 1, matrix.shape[1] - 1)
        if n_components < 1:
            return summarize_tfidf(features, n_sentences)

        svd    = TruncatedSVD(n_components=n_components, random_state=42)
        lsa    = svd.fit_transform(matrix)
        scores = np.sqrt((lsa ** 2).sum(axis=1))

        ranked  = sorted(zip(scores, range(len(sentences))), reverse=True)
        top_idx = sorted([idx for _, idx in ranked[:n_sentences]])
        return ' '.join([sentences[i] for i in top_idx])
    except:
        return summarize_tfidf(features, n_sentences)


def summarize_lexrank(features, n_sentences=5, threshold=0.1):
    """LexRank: threshold-based graph summarization."""
    sentences = features['sentences']
    if len(sentences) <= n_sentences:
        return ' '.join(sentences)

    try:
        vec    = TfidfVectorizer(stop_words='english')
        matrix = vec.fit_transform(sentences)
        sim    = cosine_similarity(matrix)

        # Apply threshold
        sim[sim < threshold] = 0
        np.fill_diagonal(sim, 0)

        # Degree centrality
        degree = sim.sum(axis=1)
        degree[degree == 0] = 1
        norm_sim = sim / degree[:, np.newaxis]

        scores = np.ones(len(sentences)) / len(sentences)
        for _ in range(30):
            scores = 0.85 * norm_sim.T @ scores + 0.15 / len(sentences)

        ranked  = sorted(zip(scores, range(len(sentences))), reverse=True)
        top_idx = sorted([idx for _, idx in ranked[:n_sentences]])
        return ' '.join([sentences[i] for i in top_idx])
    except:
        return summarize_tfidf(features, n_sentences)


def summarize_weighted(features, n_sentences=5,
                        w_tfidf=0.4, w_pos=0.3,
                        w_len=0.2, w_kw=0.1):
    """Weighted feature combination summarizer."""
    sentences = features['sentences']
    if len(sentences) <= n_sentences:
        return ' '.join(sentences)

    tfidf_s = np.array(features['tfidf_scores'])
    pos_s   = np.array(features['position_scores'])
    len_s   = np.array(features['length_scores'])
    kw_s    = np.array(features['keyword_scores'])

    # Normalize each to [0,1]
    def norm(x):
        r = x.max() - x.min()
        return (x - x.min()) / r if r > 0 else x

    combined = (w_tfidf * norm(tfidf_s) +
                w_pos   * norm(pos_s)   +
                w_len   * norm(len_s)   +
                w_kw    * norm(kw_s))

    ranked  = sorted(zip(combined, range(len(sentences))), reverse=True)
    top_idx = sorted([idx for _, idx in ranked[:n_sentences]])
    return ' '.join([sentences[i] for i in top_idx])


# ── Evaluation ────────────────────────────────────────────────
MODELS = {
    'TF-IDF'   : summarize_tfidf,
    'TextRank' : summarize_textrank,
    'LSA'      : summarize_lsa,
    'LexRank'  : summarize_lexrank,
    'Weighted' : summarize_weighted,
}

scorer   = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
N_EVAL   = 500  # evaluate on first 500 validation docs

print(f"\nEvaluating {len(MODELS)} models on {N_EVAL} documents...")
print(f"  {'Model':<12} {'ROUGE-1':>10} {'ROUGE-2':>10} {'ROUGE-L':>10}")
print("  " + "-" * 44)

results = {}
for model_name, model_fn in MODELS.items():
    r1_scores, r2_scores, rl_scores = [], [], []

    for i in range(min(N_EVAL, len(val_features))):
        features = val_features[i]
        reference = val_abstracts[i] if i < len(val_abstracts) else ""
        if not reference:
            continue

        summary = model_fn(features, n_sentences=5)
        scores  = scorer.score(reference, summary)

        r1_scores.append(scores['rouge1'].fmeasure)
        r2_scores.append(scores['rouge2'].fmeasure)
        rl_scores.append(scores['rougeL'].fmeasure)

    results[model_name] = {
        'rouge1': np.mean(r1_scores),
        'rouge2': np.mean(r2_scores),
        'rougeL': np.mean(rl_scores),
    }

    print(f"  {model_name:<12} {results[model_name]['rouge1']:>9.4f}  "
          f"{results[model_name]['rouge2']:>9.4f}  "
          f"{results[model_name]['rougeL']:>9.4f}")

# ── Save Results ──────────────────────────────────────────────
print("\nSaving results...")
with open('../dataset/model_results.pkl', 'wb') as f:
    pickle.dump(results, f)

rows = [{'Model': n,
         'ROUGE-1': round(r['rouge1'], 4),
         'ROUGE-2': round(r['rouge2'], 4),
         'ROUGE-L': round(r['rougeL'], 4)}
        for n, r in results.items()]
pd.DataFrame(rows).to_csv('../outputs/step5_rouge_results.csv', index=False)
print("  Saved: outputs/step5_rouge_results.csv")

# ── Save model functions ──────────────────────────────────────
model_functions = {
    'TF-IDF'   : summarize_tfidf,
    'TextRank' : summarize_textrank,
    'LSA'      : summarize_lsa,
    'LexRank'  : summarize_lexrank,
    'Weighted' : summarize_weighted,
}
with open('../models/summarizers.pkl', 'wb') as f:
    pickle.dump(model_functions, f)
print("  Saved: models/summarizers.pkl")

# ── Plot: ROUGE Comparison ────────────────────────────────────
names  = list(results.keys())
r1     = [results[n]['rouge1'] for n in names]
r2     = [results[n]['rouge2'] for n in names]
rl     = [results[n]['rougeL'] for n in names]

x     = np.arange(len(names))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width, r1, width, label='ROUGE-1', color='steelblue', edgecolor='black')
ax.bar(x,         r2, width, label='ROUGE-2', color='green',     edgecolor='black')
ax.bar(x + width, rl, width, label='ROUGE-L', color='orange',    edgecolor='black')

ax.set_title('Summarization Models — ROUGE Score Comparison')
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel('ROUGE Score')
ax.set_ylim(0, 0.6)
ax.legend()

for bars in [ax.containers[0], ax.containers[1], ax.containers[2]]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f'{h:.3f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('../outputs/step5_rouge_comparison.png', dpi=150)
plt.close()
print("  Saved: outputs/step5_rouge_comparison.png")

# ── Sample summaries ──────────────────────────────────────────
print("\nSample summaries (document 0):")
print(f"\n  Reference abstract:\n  {val_abstracts[0][:300]}...")
for name, fn in MODELS.items():
    summary = fn(val_features[0], n_sentences=3)
    print(f"\n  {name}:\n  {summary[:300]}...")

print("\n" + "=" * 55)
print("STEPS 4 & 5 COMPLETE")
print("=" * 55)