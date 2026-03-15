"""
PaperLens - Research Paper Simplifier
Step 5b - Weight Tuning for Weighted Summarizer
Finds optimal combination of TF-IDF, Position,
Length, and Keyword weights using grid search.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
from itertools import product

os.makedirs('../outputs', exist_ok=True)

print("=" * 55)
print("STEP 5b: WEIGHT TUNING")
print("=" * 55)

# ── Load Data ─────────────────────────────────────────────────
print("\nLoading data...")
with open('../dataset/val_features.pkl', 'rb') as f:
    val_features = pickle.load(f)
with open('../dataset/val_abstracts.pkl', 'rb') as f:
    val_abstracts = pickle.load(f)

N_EVAL  = 200  # use 200 docs for speed
scorer  = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# ── Weighted Summarizer ───────────────────────────────────────
def summarize_weighted(features, n_sentences=5,
                        w_tfidf=0.4, w_pos=0.3,
                        w_len=0.2, w_kw=0.1):
    sentences = features['sentences']
    if len(sentences) <= n_sentences:
        return ' '.join(sentences)

    tfidf_s = np.array(features['tfidf_scores'])
    pos_s   = np.array(features['position_scores'])
    len_s   = np.array(features['length_scores'])
    kw_s    = np.array(features['keyword_scores'])

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

def evaluate_weights(w_tfidf, w_pos, w_len, w_kw):
    """Evaluate a weight combination on N_EVAL documents."""
    r1_scores = []
    for i in range(min(N_EVAL, len(val_features))):
        ref     = val_abstracts[i] if i < len(val_abstracts) else ""
        if not ref:
            continue
        summary = summarize_weighted(val_features[i], n_sentences=5,
                                     w_tfidf=w_tfidf, w_pos=w_pos,
                                     w_len=w_len, w_kw=w_kw)
        score   = scorer.score(ref, summary)
        r1_scores.append(score['rouge1'].fmeasure)
    return np.mean(r1_scores)

# ── Grid Search ───────────────────────────────────────────────
print("\nRunning grid search over weight combinations...")
print("(evaluating on 200 documents per combination)\n")

# Weight options that sum to 1.0
weight_options = [0.1, 0.2, 0.3, 0.4, 0.5]
best_score  = 0
best_weights = None
results_list = []

total = 0
for w1, w2, w3 in product(weight_options, repeat=3):
    w4 = round(1.0 - w1 - w2 - w3, 1)
    if w4 in weight_options:
        total += 1

print(f"  Total combinations to try: {total}")
count = 0

for w1, w2, w3 in product(weight_options, repeat=3):
    w4 = round(1.0 - w1 - w2 - w3, 1)
    if w4 not in weight_options:
        continue

    count += 1
    score = evaluate_weights(w1, w2, w3, w4)
    results_list.append((w1, w2, w3, w4, score))

    if score > best_score:
        best_score   = score
        best_weights = (w1, w2, w3, w4)
        print(f"  [{count:3d}/{total}] New best: "
              f"TF-IDF={w1} POS={w2} LEN={w3} KW={w4} "
              f"→ ROUGE-1={score:.4f}")

# ── Results ───────────────────────────────────────────────────
print(f"\n  {'='*50}")
print(f"  BEST WEIGHTS:")
print(f"    TF-IDF   : {best_weights[0]}")
print(f"    Position : {best_weights[1]}")
print(f"    Length   : {best_weights[2]}")
print(f"    Keyword  : {best_weights[3]}")
print(f"    ROUGE-1  : {best_score:.4f}")
print(f"  {'='*50}")

# ── Save best weights ─────────────────────────────────────────
import json
weight_meta = {
    'w_tfidf'  : best_weights[0],
    'w_pos'    : best_weights[1],
    'w_len'    : best_weights[2],
    'w_kw'     : best_weights[3],
    'rouge1'   : round(best_score, 4),
}
with open('../outputs/best_weights.json', 'w') as f:
    json.dump(weight_meta, f, indent=2)
print("\n  Saved: outputs/best_weights.json")

# ── Now re-evaluate tuned model with full ROUGE ───────────────
print("\nFull ROUGE evaluation with best weights...")
full_scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
r1, r2, rl = [], [], []

for i in range(min(500, len(val_features))):
    ref = val_abstracts[i] if i < len(val_abstracts) else ""
    if not ref:
        continue
    summary = summarize_weighted(
        val_features[i], n_sentences=5,
        w_tfidf=best_weights[0], w_pos=best_weights[1],
        w_len=best_weights[2],   w_kw=best_weights[3])
    scores = full_scorer.score(ref, summary)
    r1.append(scores['rouge1'].fmeasure)
    r2.append(scores['rouge2'].fmeasure)
    rl.append(scores['rougeL'].fmeasure)

print(f"\n  Tuned Weighted Summarizer:")
print(f"    ROUGE-1 : {np.mean(r1):.4f}  (was 0.3997)")
print(f"    ROUGE-2 : {np.mean(r2):.4f}  (was 0.1308)")
print(f"    ROUGE-L : {np.mean(rl):.4f}  (was 0.2129)")

# ── Update model results ──────────────────────────────────────
with open('../dataset/model_results.pkl', 'rb') as f:
    model_results = pickle.load(f)

model_results['Weighted (Tuned)'] = {
    'rouge1': np.mean(r1),
    'rouge2': np.mean(r2),
    'rougeL': np.mean(rl),
}
with open('../dataset/model_results.pkl', 'wb') as f:
    pickle.dump(model_results, f)
print("\n  Updated: dataset/model_results.pkl")

# ── Plot: weight sensitivity ──────────────────────────────────
tfidf_vals = sorted(set(r[0] for r in results_list))
scores_by_tfidf = []
for v in tfidf_vals:
    group = [r[4] for r in results_list if r[0] == v]
    scores_by_tfidf.append(np.mean(group))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(tfidf_vals, scores_by_tfidf, marker='o',
        color='steelblue', linewidth=2)
ax.set_title('Weight Sensitivity — TF-IDF Weight vs ROUGE-1')
ax.set_xlabel('TF-IDF Weight')
ax.set_ylabel('Mean ROUGE-1 Score')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/step5b_weight_sensitivity.png', dpi=150)
plt.close()
print("  Saved: outputs/step5b_weight_sensitivity.png")

print("\n" + "=" * 55)
print("STEP 5b COMPLETE")
print("=" * 55)