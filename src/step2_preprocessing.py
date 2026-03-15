"""
PaperLens - Research Paper Simplifier
Step 2 - Text Preprocessing
Pipeline: clean text, remove citations, tokenize,
          remove stopwords, lemmatize
"""

import os
import re
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

os.makedirs('../outputs', exist_ok=True)
os.makedirs('../dataset', exist_ok=True)

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

print("=" * 55)
print("STEP 2: TEXT PREPROCESSING")
print("=" * 55)

# ── Preprocessing Functions ───────────────────────────────────

def clean_text(text):
    """Full cleaning pipeline for academic text."""
    # Remove citations like [1], [2,3], [Smith et al., 2020]
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
    text = re.sub(r'\[[\w\s]+et al\.?,?\s*\d{4}\]', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove LaTeX commands
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    text = re.sub(r'\$[^$]+\$', '', text)
    # Remove special characters but keep sentence structure
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_for_features(text):
    """Preprocess for feature extraction (removes stopwords)."""
    text = clean_text(text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens
              if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)

def get_sentences(text):
    """Split text into clean sentences."""
    text = clean_text(text)
    sentences = sent_tokenize(text)
    sentences = [s.strip() for s in sentences if len(s.split()) > 5]
    return sentences

# ── Load Dataset ──────────────────────────────────────────────
print("\nLoading dataset...")
dataset = load_dataset("ccdv/arxiv-summarization", split="train[:10000]")
val_set = load_dataset("ccdv/arxiv-summarization", split="validation[:2000]")
print(f"  Train : {len(dataset):,}")
print(f"  Val   : {len(val_set):,}")

# ── Preprocess ────────────────────────────────────────────────
print("\nPreprocessing articles...")
train_processed = []
for i, item in enumerate(dataset):
    train_processed.append({
        'article_clean'    : clean_text(item['article']),
        'article_features' : preprocess_for_features(item['article']),
        'abstract_clean'   : clean_text(item['abstract']),
        'sentences'        : get_sentences(item['article']),
    })
    if (i + 1) % 1000 == 0:
        print(f"  {i+1}/10000 done...")

print("\nPreprocessing validation set...")
val_processed = []
for i, item in enumerate(val_set):
    val_processed.append({
        'article_clean'    : clean_text(item['article']),
        'article_features' : preprocess_for_features(item['article']),
        'abstract_clean'   : clean_text(item['abstract']),
        'sentences'        : get_sentences(item['article']),
    })
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/2000 done...")

# ── Show examples ─────────────────────────────────────────────
print("\nBefore vs After examples:")
for i in range(2):
    raw   = dataset[i]['article'][:300]
    clean = train_processed[i]['article_clean'][:300]
    print(f"\n  [{i+1}] RAW   : {raw}...")
    print(f"  [{i+1}] CLEAN : {clean}...")

# ── Stats ─────────────────────────────────────────────────────
raw_lengths   = [len(d['article'].split())               for d in dataset]
clean_lengths = [len(d['article_clean'].split())         for d in train_processed]
sent_counts   = [len(d['sentences'])                     for d in train_processed]

print(f"\n  Token count before : Mean = {np.mean(raw_lengths):.0f}")
print(f"  Token count after  : Mean = {np.mean(clean_lengths):.0f}")
print(f"  Reduction          : {(1 - np.mean(clean_lengths)/np.mean(raw_lengths))*100:.1f}%")
print(f"  Mean sentences     : {np.mean(sent_counts):.0f} per article")

# ── Save ──────────────────────────────────────────────────────
print("\nSaving preprocessed data...")
with open('../dataset/train_processed.pkl', 'wb') as f:
    pickle.dump(train_processed, f)
with open('../dataset/val_processed.pkl', 'wb') as f:
    pickle.dump(val_processed, f)

# Save abstracts separately for evaluation
train_abstracts = [clean_text(d['abstract']) for d in dataset]
val_abstracts   = [clean_text(d['abstract']) for d in val_set]
with open('../dataset/train_abstracts.pkl', 'wb') as f:
    pickle.dump(train_abstracts, f)
with open('../dataset/val_abstracts.pkl', 'wb') as f:
    pickle.dump(val_abstracts, f)

print("  Saved: dataset/train_processed.pkl")
print("  Saved: dataset/val_processed.pkl")

# ── Plot ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Text Preprocessing — Token Count Comparison', fontsize=13)

axes[0].hist(raw_lengths,   bins=50, color='steelblue', edgecolor='black')
axes[0].axvline(np.mean(raw_lengths), color='red', linestyle='--',
                label=f'Mean={np.mean(raw_lengths):.0f}')
axes[0].set_title('Before Preprocessing')
axes[0].set_xlabel('Token Count')
axes[0].set_ylabel('Frequency')
axes[0].set_xlim(0, 15000)
axes[0].legend()

axes[1].hist(clean_lengths, bins=50, color='green', edgecolor='black')
axes[1].axvline(np.mean(clean_lengths), color='red', linestyle='--',
                label=f'Mean={np.mean(clean_lengths):.0f}')
axes[1].set_title('After Preprocessing')
axes[1].set_xlabel('Token Count')
axes[1].set_ylabel('Frequency')
axes[1].set_xlim(0, 15000)
axes[1].legend()

plt.tight_layout()
plt.savefig('../outputs/step2_preprocessing_comparison.png', dpi=150)
plt.close()
print("  Saved: outputs/step2_preprocessing_comparison.png")

print("\n" + "=" * 55)
print("STEP 2 COMPLETE")
print("=" * 55)