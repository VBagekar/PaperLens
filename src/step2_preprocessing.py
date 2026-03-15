"""
Step 2 - Text Preprocessing
Pipeline: lowercase, remove noise, tokenize, stopwords, lemmatize
"""

import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import fetch_20newsgroups
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

os.makedirs('../outputs', exist_ok=True)
os.makedirs('../dataset', exist_ok=True)

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

# ── Preprocessing Function ────────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)     # remove URLs
    text = re.sub(r'\S+@\S+', '', text)               # remove emails
    text = re.sub(r'\d+', '', text)                   # remove numbers
    text = re.sub(r'[^a-z\s]', ' ', text)            # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()          # collapse spaces
    tokens = word_tokenize(text)                       # tokenize
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens # lemmatize
              if t not in STOP_WORDS and len(t) > 2]  # remove stopwords
    return ' '.join(tokens)

# ── Load Dataset ──────────────────────────────────────────────
print("=" * 50)
print("STEP 2: TEXT PREPROCESSING")
print("=" * 50)

print("\nLoading dataset...")
train_data = fetch_20newsgroups(subset='train')
test_data  = fetch_20newsgroups(subset='test')

# ── Preprocess ────────────────────────────────────────────────
print("\nPreprocessing training data...")
train_clean = []
for i, text in enumerate(train_data.data):
    train_clean.append(preprocess(text))
    if (i + 1) % 1000 == 0:
        print(f"  {i+1}/{len(train_data.data)} done...")

print("\nPreprocessing test data...")
test_clean = []
for i, text in enumerate(test_data.data):
    test_clean.append(preprocess(text))
    if (i + 1) % 1000 == 0:
        print(f"  {i+1}/{len(test_data.data)} done...")

# ── Show Examples ─────────────────────────────────────────────
print("\nBefore vs After examples:")
for i in range(3):
    print(f"\n  [{i+1}] BEFORE: {train_data.data[i][:150].strip()}")
    print(f"  [{i+1}] AFTER : {train_clean[i][:150]}")

# ── Stats ─────────────────────────────────────────────────────
raw_lengths   = [len(t.split()) for t in train_data.data]
clean_lengths = [len(t.split()) for t in train_clean]
print(f"\nToken count before preprocessing - Mean: {np.mean(raw_lengths):.1f}")
print(f"Token count after  preprocessing - Mean: {np.mean(clean_lengths):.1f}")
reduction = (1 - np.mean(clean_lengths) / np.mean(raw_lengths)) * 100
print(f"Token reduction: {reduction:.1f}%")

# ── Save Preprocessed Data ────────────────────────────────────
print("\nSaving preprocessed data...")
data = {
    'train_clean'   : train_clean,
    'test_clean'    : test_clean,
    'train_labels'  : train_data.target.tolist(),
    'test_labels'   : test_data.target.tolist(),
    'category_names': train_data.target_names,
}
with open('../dataset/preprocessed_data.pkl', 'wb') as f:
    pickle.dump(data, f)
print("  Saved: dataset/preprocessed_data.pkl")

# ── Plot: Before vs After ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Text Preprocessing - Token Count Comparison', fontsize=14)

axes[0].hist(raw_lengths,   bins=50, color='steelblue', edgecolor='black')
axes[0].axvline(np.mean(raw_lengths), color='red', linestyle='--',
                label=f'Mean = {np.mean(raw_lengths):.0f}')
axes[0].set_title('Before Preprocessing')
axes[0].set_xlabel('Token Count')
axes[0].set_ylabel('Frequency')
axes[0].set_xlim(0, 1500)
axes[0].legend()

axes[1].hist(clean_lengths, bins=50, color='green', edgecolor='black')
axes[1].axvline(np.mean(clean_lengths), color='red', linestyle='--',
                label=f'Mean = {np.mean(clean_lengths):.0f}')
axes[1].set_title('After Preprocessing')
axes[1].set_xlabel('Token Count')
axes[1].set_ylabel('Frequency')
axes[1].set_xlim(0, 1500)
axes[1].legend()

plt.tight_layout()
plt.savefig('../outputs/step2_preprocessing_comparison.png', dpi=150)
plt.close()
print("  Saved: outputs/step2_preprocessing_comparison.png")

print("\n" + "=" * 50)
print("STEP 2 COMPLETE")
print("=" * 50)