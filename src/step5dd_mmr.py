"""
PaperLens - Research Paper Simplifier
Step 5d - MMR (Maximal Marginal Relevance) Summarizer
MMR balances relevance and diversity:
  score = lambda * relevance - (1-lambda) * max_similarity_to_selected
This reduces redundancy between selected sentences.
"""

import os
import pickle
import numpy as np
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

os.makedirs('../outputs', exist_ok=True)

print("=" * 55)
print("STEP 5d: MMR SUMMARIZATION")
print("=" * 55)

# ── Load Data ─────────────────────────────────────────────────
print("\nLoading data...")
with open('../dataset/val_features.pkl', 'rb') as f:
    val_features = pickle.load(f)
with open('../dataset/val_abstracts.pkl', 'rb') as f:
    val_abstracts = pickle.load(f)

print(f"  Val documents : {len(val_features):,}")

# ── MMR Summarizer ────────────────────────────────────────────
def summarize_mmr(features, n_sentences=6, lambda_=0.6,
                  w_tfidf=0.5, w_pos=0.3, w_len=0.1, w_kw=0.1):
    """
    MMR Summarizer:
    1. Score sentences using weighted features (relevance)
    2. Build TF-IDF vectors for similarity computation
    3. Iteratively select sentences maximizing:
       MMR = lambda * relevance - (1-lambda) * max_sim_to_selected
    """
    sentences = features['sentences']
    if len(sentences) <= n_sentences:
        return ' '.join(sentences)

    # ── Compute relevance scores ──
    tfidf_s = np.array(features['tfidf_scores'])
    pos_s   = np.array(features['position_scores'])
    len_s   = np.array(features['length_scores'])
    kw_s    = np.array(features['keyword_scores'])

    def norm(x):
        r = x.max() - x.min()
        return (x - x.min()) / r if r > 0 else x

    relevance = (w_tfidf * norm(tfidf_s) +
                 w_pos   * norm(pos_s)   +
                 w_len   * norm(len_s)   +
                 w_kw    * norm(kw_s))

    # ── Build TF-IDF vectors for similarity ──
    try:
        vec    = TfidfVectorizer(stop_words='english', max_features=5000)
        matrix = vec.fit_transform(sentences).toarray()
    except:
        # Fallback to weighted summarizer if vectorization fails
        ranked  = sorted(zip(relevance, range(len(sentences))), reverse=True)
        top_idx = sorted([idx for _, idx in ranked[:n_sentences]])
        return ' '.join([sentences[i] for i in top_idx])

    # ── MMR selection ──
    selected   = []
    remaining  = list(range(len(sentences)))

    while len(selected) < n_sentences and remaining:
        if not selected:
            # First sentence: pick highest relevance
            best_idx = max(remaining, key=lambda i: relevance[i])
        else:
            # MMR score for each remaining sentence
            mmr_scores = {}
            for i in remaining:
                # Max similarity to any already selected sentence
                sim_to_selected = max(
                    cosine_similarity([matrix[i]], [matrix[j]])[0][0]
                    for j in selected
                )
                mmr_scores[i] = (lambda_ * relevance[i] -
                                 (1 - lambda_) * sim_to_selected)
            best_idx = max(mmr_scores, key=mmr_scores.get)

        selected.append(best_idx)
        remaining.remove(best_idx)

    # Re-order by original position for coherent reading
    selected.sort()
    return ' '.join([sentences[i] for i in selected])


# ── Tune lambda ───────────────────────────────────────────────
print("\nTuning lambda parameter...")
scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

lambda_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
print(f"\n  {'Lambda':>8} {'ROUGE-1':>10}")
print("  " + "-" * 20)

best_lambda = 0.6
best_r1     = 0.0
lambda_results = {}

for lam in lambda_values:
    scores = []
    for i in range(200):
        ref = val_abstracts[i] if i < len(val_abstracts) else ''
        if not ref:
            continue
        summary = summarize_mmr(val_features[i], n_sentences=6, lambda_=lam)
        scores.append(scorer.score(ref, summary)['rouge1'].fmeasure)
    mean_r1 = np.mean(scores)
    lambda_results[lam] = mean_r1
    print(f"  {lam:>8.1f} {mean_r1:>9.4f}")
    if mean_r1 > best_r1:
        best_r1     = mean_r1
        best_lambda = lam

print(f"\n  Best lambda : {best_lambda}")
print(f"  ROUGE-1     : {best_r1:.4f}")

# ── Full ROUGE evaluation ─────────────────────────────────────
print(f"\nFull ROUGE evaluation (lambda={best_lambda}, n=6)...")
full_scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

r1, r2, rl = [], [], []
for i in range(500):
    ref = val_abstracts[i] if i < len(val_abstracts) else ''
    if not ref:
        continue
    summary = summarize_mmr(val_features[i], n_sentences=6,
                             lambda_=best_lambda)
    scores  = full_scorer.score(ref, summary)
    r1.append(scores['rouge1'].fmeasure)
    r2.append(scores['rouge2'].fmeasure)
    rl.append(scores['rougeL'].fmeasure)

print(f"\n  MMR Results (lambda={best_lambda}):")
print(f"    ROUGE-1 : {np.mean(r1):.4f}  (previous best: 0.4112)")
print(f"    ROUGE-2 : {np.mean(r2):.4f}  (previous best: 0.1380)")
print(f"    ROUGE-L : {np.mean(rl):.4f}  (previous best: 0.2140)")

# ── Update model results ──────────────────────────────────────
with open('../dataset/model_results.pkl', 'rb') as f:
    model_results = pickle.load(f)

model_results[f'MMR (lambda={best_lambda})'] = {
    'rouge1': np.mean(r1),
    'rouge2': np.mean(r2),
    'rougeL': np.mean(rl),
}
with open('../dataset/model_results.pkl', 'wb') as f:
    pickle.dump(model_results, f)
print("\n  Updated: dataset/model_results.pkl")

# ── Plot ──────────────────────────────────────────────────────
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(list(lambda_results.keys()),
        list(lambda_results.values()),
        marker='o', color='steelblue', linewidth=2)
ax.axvline(best_lambda, color='red', linestyle='--',
           label=f'Best lambda={best_lambda}')
ax.set_title('MMR — Lambda Tuning vs ROUGE-1')
ax.set_xlabel('Lambda (relevance weight)')
ax.set_ylabel('ROUGE-1 Score')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/step5d_mmr_lambda.png', dpi=150)
plt.close()
print("  Saved: outputs/step5d_mmr_lambda.png")

print("\n" + "=" * 55)
print("STEP 5d COMPLETE")
print("=" * 55)