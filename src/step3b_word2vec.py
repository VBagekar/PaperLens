"""
Step 3b - Word2Vec Feature Extraction
Trains Word2Vec embeddings and creates document vectors
by averaging word vectors for each document.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.datasets import fetch_20newsgroups

os.makedirs('../outputs', exist_ok=True)

print("=" * 50)
print("STEP 3b: WORD2VEC FEATURE EXTRACTION")
print("=" * 50)

# ── Load Preprocessed Data ────────────────────────────────────
print("\nLoading preprocessed data...")
with open('../dataset/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

train_texts    = data['train_clean']
test_texts     = data['test_clean']
train_labels   = data['train_labels']
test_labels    = data['test_labels']
category_names = data['category_names']

# Tokenize into lists of words
train_tokens = [text.split() for text in train_texts]
test_tokens  = [text.split() for text in test_texts]

print(f"  Train documents : {len(train_tokens)}")
print(f"  Test  documents : {len(test_tokens)}")

# ── Train Word2Vec ────────────────────────────────────────────
print("\nTraining Word2Vec model...")
print("  vector_size = 300")
print("  window      = 5")
print("  min_count   = 2")
print("  epochs      = 10")

w2v_model = Word2Vec(
    sentences   = train_tokens,
    vector_size = 300,
    window      = 5,
    min_count   = 2,
    workers     = 4,
    epochs      = 10,
    seed        = 42,
)

vocab_size = len(w2v_model.wv)
print(f"  Vocabulary size : {vocab_size:,} words")

# ── Document Vectors (average word vectors) ───────────────────
def document_vector(tokens, model, vector_size=300):
    """Average all word vectors in a document."""
    vectors = []
    for word in tokens:
        if word in model.wv:
            vectors.append(model.wv[word])
    if len(vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

print("\nCreating document vectors...")
X_train_w2v = np.array([document_vector(tokens, w2v_model) for tokens in train_tokens])
X_test_w2v  = np.array([document_vector(tokens, w2v_model) for tokens in test_tokens])

print(f"  X_train shape : {X_train_w2v.shape}")
print(f"  X_test  shape : {X_test_w2v.shape}")

# ── Save ──────────────────────────────────────────────────────
print("\nSaving Word2Vec features...")
w2v_features = {
    'X_train'      : X_train_w2v,
    'X_test'       : X_test_w2v,
    'train_labels' : train_labels,
    'test_labels'  : test_labels,
    'category_names': category_names,
    'vocab_size'   : vocab_size,
}
with open('../dataset/w2v_features.pkl', 'wb') as f:
    pickle.dump(w2v_features, f)
print("  Saved: dataset/w2v_features.pkl")

# ── Plot: Similar words for 3 sample words ────────────────────
print("\nSample word similarities:")
sample_words = ['computer', 'baseball', 'religion']
for word in sample_words:
    if word in w2v_model.wv:
        similar = w2v_model.wv.most_similar(word, topn=5)
        print(f"\n  Most similar to '{word}':")
        for w, score in similar:
            print(f"    {w:<20} {score:.4f}")

# ── Plot: Vocabulary coverage ─────────────────────────────────
coverage = []
for tokens in train_tokens:
    if len(tokens) == 0:
        coverage.append(0)
        continue
    known = sum(1 for t in tokens if t in w2v_model.wv)
    coverage.append(known / len(tokens))

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(coverage, bins=30, color='steelblue', edgecolor='black')
ax.set_title('Word2Vec Vocabulary Coverage per Document')
ax.set_xlabel('Proportion of Words Found in W2V Vocabulary')
ax.set_ylabel('Number of Documents')
ax.axvline(np.mean(coverage), color='red', linestyle='--',
           label=f'Mean = {np.mean(coverage):.2f}')
ax.legend()
plt.tight_layout()
plt.savefig('../outputs/step3b_w2v_coverage.png', dpi=150)
plt.close()
print(f"\n  Mean vocabulary coverage : {np.mean(coverage)*100:.1f}%")
print("  Saved: outputs/step3b_w2v_coverage.png")

print("\n" + "=" * 50)
print("STEP 3b COMPLETE")
print("=" * 50)