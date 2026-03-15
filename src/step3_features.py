"""
PaperLens - Research Paper Simplifier
Step 3 - Feature Extraction
Extracts features for extractive summarization:
  1. TF-IDF sentence scores
  2. Sentence position scores
  3. Sentence length scores
  4. Title similarity scores
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

os.makedirs('../outputs', exist_ok=True)

print("=" * 55)
print("STEP 3: FEATURE EXTRACTION")
print("=" * 55)

# ── Load Preprocessed Data ────────────────────────────────────
print("\nLoading preprocessed data...")
with open('../dataset/train_processed.pkl', 'rb') as f:
    train_processed = pickle.load(f)
with open('../dataset/val_processed.pkl', 'rb') as f:
    val_processed = pickle.load(f)

print(f"  Train : {len(train_processed):,}")
print(f"  Val   : {len(val_processed):,}")

# ── Feature Extraction Functions ─────────────────────────────

def extract_tfidf_scores(sentences, vectorizer=None, fit=False):
    """Score sentences using TF-IDF cosine similarity to document."""
    if len(sentences) == 0:
        return []
    if fit or vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
    else:
        tfidf_matrix = vectorizer.transform(sentences)

    # Document vector = mean of all sentence vectors
    doc_vector = np.asarray(tfidf_matrix.mean(axis=0))
    scores = cosine_similarity(tfidf_matrix, doc_vector).flatten()
    return scores.tolist()

def extract_position_scores(sentences):
    """Score sentences by position — first and last are more important."""
    n = len(sentences)
    if n == 0:
        return []
    scores = []
    for i in range(n):
        if i < n * 0.1:        # first 10%
            scores.append(1.0)
        elif i > n * 0.9:      # last 10%
            scores.append(0.8)
        elif i < n * 0.2:      # next 10%
            scores.append(0.6)
        else:
            scores.append(0.3)
    return scores

def extract_length_scores(sentences):
    """Score sentences by length — prefer medium length sentences."""
    scores = []
    lengths = [len(s.split()) for s in sentences]
    if not lengths:
        return []
    mean_len = np.mean(lengths)
    for length in lengths:
        if 10 <= length <= 40:
            scores.append(1.0)
        elif length < 10:
            scores.append(0.3)
        elif length > 80:
            scores.append(0.4)
        else:
            scores.append(0.7)
    return scores

def extract_keyword_scores(sentences, keywords):
    """Score sentences by keyword frequency."""
    scores = []
    keyword_set = set(keywords)
    for sent in sentences:
        words = sent.lower().split()
        count = sum(1 for w in words if w in keyword_set)
        scores.append(count / (len(words) + 1))
    # Normalize
    max_score = max(scores) if scores else 1
    return [s / max_score if max_score > 0 else 0 for s in scores]

def get_top_keywords(text, n=20):
    """Extract top N keywords using TF-IDF."""
    try:
        vec = TfidfVectorizer(max_features=n, stop_words='english')
        vec.fit_transform([text])
        return list(vec.vocabulary_.keys())
    except:
        return []

def extract_all_features(processed_item):
    """Extract all features for a single document."""
    sentences = processed_item['sentences']
    if len(sentences) < 2:
        return None

    tfidf_scores    = extract_tfidf_scores(sentences, fit=True)
    position_scores = extract_position_scores(sentences)
    length_scores   = extract_length_scores(sentences)
    keywords        = get_top_keywords(processed_item['article_clean'])
    keyword_scores  = extract_keyword_scores(sentences, keywords)

    return {
        'sentences'      : sentences,
        'tfidf_scores'   : tfidf_scores,
        'position_scores': position_scores,
        'length_scores'  : length_scores,
        'keyword_scores' : keyword_scores,
        'keywords'       : keywords,
    }

# ── Extract Features ──────────────────────────────────────────
print("\nExtracting features from training set...")
train_features = []
skipped = 0
for i, item in enumerate(train_processed):
    features = extract_all_features(item)
    if features:
        train_features.append(features)
    else:
        skipped += 1
    if (i + 1) % 1000 == 0:
        print(f"  {i+1}/10000 done...")

print(f"\n  Extracted : {len(train_features):,}")
print(f"  Skipped   : {skipped}")

print("\nExtracting features from validation set...")
val_features = []
for i, item in enumerate(val_processed):
    features = extract_all_features(item)
    if features:
        val_features.append(features)
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/2000 done...")

print(f"\n  Extracted : {len(val_features):,}")

# ── Feature Statistics ────────────────────────────────────────
print("\nFeature Statistics:")
sample = train_features[:100]

mean_tfidf    = np.mean([np.mean(f['tfidf_scores'])    for f in sample])
mean_position = np.mean([np.mean(f['position_scores']) for f in sample])
mean_length   = np.mean([np.mean(f['length_scores'])   for f in sample])
mean_keyword  = np.mean([np.mean(f['keyword_scores'])  for f in sample])

print(f"  Mean TF-IDF score    : {mean_tfidf:.4f}")
print(f"  Mean Position score  : {mean_position:.4f}")
print(f"  Mean Length score    : {mean_length:.4f}")
print(f"  Mean Keyword score   : {mean_keyword:.4f}")

# ── Save Features ─────────────────────────────────────────────
print("\nSaving features...")
with open('../dataset/train_features.pkl', 'wb') as f:
    pickle.dump(train_features, f)
with open('../dataset/val_features.pkl', 'wb') as f:
    pickle.dump(val_features, f)
print("  Saved: dataset/train_features.pkl")
print("  Saved: dataset/val_features.pkl")

# ── Plot: Feature score distributions ────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Feature Score Distributions', fontsize=14)

# Flatten all scores for plotting
all_tfidf    = [s for f in sample for s in f['tfidf_scores']]
all_position = [s for f in sample for s in f['position_scores']]
all_length   = [s for f in sample for s in f['length_scores']]
all_keyword  = [s for f in sample for s in f['keyword_scores']]

axes[0,0].hist(all_tfidf,    bins=30, color='steelblue', edgecolor='black')
axes[0,0].set_title('TF-IDF Scores')
axes[0,0].set_xlabel('Score')

axes[0,1].hist(all_position, bins=10, color='green', edgecolor='black')
axes[0,1].set_title('Position Scores')
axes[0,1].set_xlabel('Score')

axes[1,0].hist(all_length,   bins=10, color='orange', edgecolor='black')
axes[1,0].set_title('Length Scores')
axes[1,0].set_xlabel('Score')

axes[1,1].hist(all_keyword,  bins=30, color='purple', edgecolor='black')
axes[1,1].set_title('Keyword Scores')
axes[1,1].set_xlabel('Score')

for ax in axes.flat:
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('../outputs/step3_feature_distributions.png', dpi=150)
plt.close()
print("  Saved: outputs/step3_feature_distributions.png")

print("\n" + "=" * 55)
print("STEP 3 COMPLETE")
print("=" * 55)