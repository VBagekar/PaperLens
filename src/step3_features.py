"""
Step 3 - Feature Extraction
Methods: Bag of Words, TF-IDF, TF-IDF + Bigrams
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

os.makedirs('../outputs', exist_ok=True)

# ── Load Preprocessed Data ────────────────────────────────────
print("=" * 50)
print("STEP 3: FEATURE EXTRACTION")
print("=" * 50)

print("\nLoading preprocessed data...")
with open('../dataset/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

train_texts    = data['train_clean']
test_texts     = data['test_clean']
train_labels   = data['train_labels']
category_names = data['category_names']

print(f"  Train documents : {len(train_texts)}")
print(f"  Test  documents : {len(test_texts)}")

# ── Vectorizers ───────────────────────────────────────────────
print("\nFitting vectorizers...")

# 1. Bag of Words
bow = CountVectorizer(max_features=20000)
X_train_bow = bow.fit_transform(train_texts)
X_test_bow  = bow.transform(test_texts)
print(f"  Bag of Words   - vocab: {len(bow.vocabulary_):,}  shape: {X_train_bow.shape}")

# 2. TF-IDF Unigrams
tfidf = TfidfVectorizer(max_features=50000, sublinear_tf=True, 
                        min_df=2, max_df=0.95, ngram_range=(1,2))


X_train_tfidf = tfidf.fit_transform(train_texts)
X_test_tfidf  = tfidf.transform(test_texts)
print(f"  TF-IDF         - vocab: {len(tfidf.vocabulary_):,}  shape: {X_train_tfidf.shape}")

# 3. TF-IDF + Bigrams
tfidf_ng = TfidfVectorizer(max_features=30000, ngram_range=(1,2), sublinear_tf=True)
X_train_ng = tfidf_ng.fit_transform(train_texts)
X_test_ng  = tfidf_ng.transform(test_texts)
print(f"  TF-IDF+Bigrams - vocab: {len(tfidf_ng.vocabulary_):,}  shape: {X_train_ng.shape}")

# ── Save Features ─────────────────────────────────────────────
print("\nSaving features...")
features = {
    'bow'        : {'X_train': X_train_bow,   'X_test': X_test_bow,   'vectorizer': bow},
    'tfidf'      : {'X_train': X_train_tfidf, 'X_test': X_test_tfidf, 'vectorizer': tfidf},
    'tfidf_ngram': {'X_train': X_train_ng,    'X_test': X_test_ng,    'vectorizer': tfidf_ng},
}
with open('../dataset/features.pkl', 'wb') as f:
    pickle.dump(features, f)
print("  Saved: dataset/features.pkl")

# ── Plot: Vocabulary Size Comparison ─────────────────────────
methods = ['Bag of Words', 'TF-IDF', 'TF-IDF + Bigrams']
vocabs  = [len(bow.vocabulary_), len(tfidf.vocabulary_), len(tfidf_ng.vocabulary_)]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(methods, vocabs, color=['steelblue', 'green', 'orange'], edgecolor='black', width=0.5)
ax.set_title('Feature Extraction - Vocabulary Size Comparison')
ax.set_ylabel('Vocabulary Size')
for bar, val in zip(bars, vocabs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
            f'{val:,}', ha='center', fontsize=11)
ax.set_ylim(0, max(vocabs) * 1.15)
plt.tight_layout()
plt.savefig('../outputs/step3_vocab_comparison.png', dpi=150)
plt.close()
print("  Saved: outputs/step3_vocab_comparison.png")

# ── Plot: Top TF-IDF Terms for 3 Categories ──────────────────
feature_names = np.array(tfidf.get_feature_names_out())
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Top 10 TF-IDF Terms per Category (Sample)', fontsize=13)

for idx, ax in enumerate(axes):
    mask       = np.array(train_labels) == idx
    mean_tfidf = np.asarray(X_train_tfidf[mask].mean(axis=0)).flatten()
    top_idx    = mean_tfidf.argsort()[-10:][::-1]
    ax.barh(feature_names[top_idx][::-1], mean_tfidf[top_idx][::-1], color='steelblue')
    ax.set_title(category_names[idx], fontsize=9)
    ax.tick_params(axis='y', labelsize=8)

plt.tight_layout()
plt.savefig('../outputs/step3_top_tfidf_terms.png', dpi=150)
plt.close()
print("  Saved: outputs/step3_top_tfidf_terms.png")

print("\n" + "=" * 50)
print("STEP 3 COMPLETE")
print("=" * 50)