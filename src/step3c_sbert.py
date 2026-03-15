"""
Step 3c - Sentence-BERT Feature Extraction
Uses pre-trained all-MiniLM-L6-v2 model to generate
384-dimensional semantic sentence embeddings.
"""

import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import torch

os.makedirs('../outputs', exist_ok=True)

print("=" * 50)
print("STEP 3c: SENTENCE-BERT FEATURE EXTRACTION")
print("=" * 50)

# ── Check GPU ─────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n  Device : {device}")
if device == 'cuda':
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")

# ── Load Preprocessed Data ────────────────────────────────────
print("\nLoading preprocessed data...")
with open('../dataset/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

train_texts    = data['train_clean']
test_texts     = data['test_clean']
train_labels   = data['train_labels']
test_labels    = data['test_labels']
category_names = data['category_names']

print(f"  Train documents : {len(train_texts)}")
print(f"  Test  documents : {len(test_texts)}")

# ── Load SBERT Model ──────────────────────────────────────────
print("\nLoading Sentence-BERT model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
print("  Model loaded successfully")

# ── Encode Documents ──────────────────────────────────────────
print("\nEncoding training documents...")
t0 = time.time()
X_train = model.encode(
    train_texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    device=device
)
train_time = time.time() - t0
print(f"  Done in {train_time:.1f}s  shape: {X_train.shape}")

print("\nEncoding test documents...")
t0 = time.time()
X_test = model.encode(
    test_texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    device=device
)
test_time = time.time() - t0
print(f"  Done in {test_time:.1f}s  shape: {X_test.shape}")

# ── Save ──────────────────────────────────────────────────────
print("\nSaving SBERT features...")
sbert_features = {
    'X_train'       : X_train,
    'X_test'        : X_test,
    'train_labels'  : train_labels,
    'test_labels'   : test_labels,
    'category_names': category_names,
}
with open('../dataset/sbert_features.pkl', 'wb') as f:
    pickle.dump(sbert_features, f)
print("  Saved: dataset/sbert_features.pkl")

# ── Plot: Embedding dimension distribution ────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
sample = X_train[:100].flatten()
ax.hist(sample, bins=50, color='steelblue', edgecolor='black')
ax.set_title('SBERT Embedding Value Distribution (sample)')
ax.set_xlabel('Embedding Value')
ax.set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('../outputs/step3c_sbert_distribution.png', dpi=150)
plt.close()
print("  Saved: outputs/step3c_sbert_distribution.png")

print(f"\n  Total encoding time : {train_time + test_time:.1f}s")
print(f"  Embedding dimension : {X_train.shape[1]}")

print("\n" + "=" * 50)
print("STEP 3c COMPLETE")
print("=" * 50)