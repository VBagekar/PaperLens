"""
PaperLens - Research Paper Simplifier
Step 1 - Dataset Loading and Exploration
Dataset: ArXiv dataset from Hugging Face
         215,000+ research papers with abstracts
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from datasets import load_dataset

os.makedirs('../outputs', exist_ok=True)

print("=" * 55)
print("STEP 1: DATASET LOADING AND EXPLORATION")
print("=" * 55)

# ── Load Dataset ──────────────────────────────────────────────
print("\nLoading ArXiv dataset from Hugging Face...")
print("(This may take a few minutes on first run...)")

dataset = load_dataset("ccdv/arxiv-summarization", split="train[:10000]")
val_set = load_dataset("ccdv/arxiv-summarization", split="validation[:2000]")

print(f"\n  Train samples      : {len(dataset):,}")
print(f"  Validation samples : {len(val_set):,}")
print(f"  Features           : {list(dataset.features.keys())}")

# ── Sample Documents ──────────────────────────────────────────
print("\nSample documents:")
for i in range(2):
    article = dataset[i]['article']
    abstract = dataset[i]['abstract']
    print(f"\n  [{i+1}] Article preview  : {article[:200]}...")
    print(f"  [{i+1}] Abstract preview : {abstract[:200]}...")

# ── Statistics ────────────────────────────────────────────────
print("\nComputing statistics...")

article_lengths  = [len(d['article'].split())  for d in dataset]
abstract_lengths = [len(d['abstract'].split()) for d in dataset]

print(f"\n  Article Length (words):")
print(f"    Mean   : {np.mean(article_lengths):.0f}")
print(f"    Median : {np.median(article_lengths):.0f}")
print(f"    Min    : {min(article_lengths)}")
print(f"    Max    : {max(article_lengths)}")

print(f"\n  Abstract Length (words):")
print(f"    Mean   : {np.mean(abstract_lengths):.0f}")
print(f"    Median : {np.median(abstract_lengths):.0f}")
print(f"    Min    : {min(abstract_lengths)}")
print(f"    Max    : {max(abstract_lengths)}")

compression = [a/b for a, b in zip(abstract_lengths, article_lengths) if b > 0]
print(f"\n  Compression Ratio (abstract/article):")
print(f"    Mean   : {np.mean(compression):.3f}")
print(f"    Meaning: abstracts are ~{1/np.mean(compression):.0f}x shorter than articles")

# ── Save sample data ──────────────────────────────────────────
print("\nSaving sample data...")
samples = []
for i in range(5):
    samples.append({
        'article' : dataset[i]['article'][:1000],
        'abstract': dataset[i]['abstract'],
    })
with open('../outputs/step1_samples.json', 'w') as f:
    json.dump(samples, f, indent=2)
print("  Saved: outputs/step1_samples.json")

# ── Plot 1: Article length distribution ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('ArXiv Dataset — Length Distribution', fontsize=14)

axes[0].hist(article_lengths,  bins=50, color='steelblue', edgecolor='black')
axes[0].axvline(np.mean(article_lengths), color='red', linestyle='--',
                label=f'Mean = {np.mean(article_lengths):.0f}')
axes[0].set_title('Article Lengths')
axes[0].set_xlabel('Word Count')
axes[0].set_ylabel('Frequency')
axes[0].set_xlim(0, 15000)
axes[0].legend()

axes[1].hist(abstract_lengths, bins=50, color='green', edgecolor='black')
axes[1].axvline(np.mean(abstract_lengths), color='red', linestyle='--',
                label=f'Mean = {np.mean(abstract_lengths):.0f}')
axes[1].set_title('Abstract Lengths')
axes[1].set_xlabel('Word Count')
axes[1].set_ylabel('Frequency')
axes[1].legend()

plt.tight_layout()
plt.savefig('../outputs/step1_length_distribution.png', dpi=150)
plt.close()
print("  Saved: outputs/step1_length_distribution.png")

# ── Plot 2: Compression ratio ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(compression, bins=50, color='orange', edgecolor='black')
ax.axvline(np.mean(compression), color='red', linestyle='--',
           label=f'Mean = {np.mean(compression):.3f}')
ax.set_title('Compression Ratio — Abstract vs Article Length')
ax.set_xlabel('Compression Ratio (abstract words / article words)')
ax.set_ylabel('Frequency')
ax.set_xlim(0, 0.5)
ax.legend()
plt.tight_layout()
plt.savefig('../outputs/step1_compression_ratio.png', dpi=150)
plt.close()
print("  Saved: outputs/step1_compression_ratio.png")

print("\n" + "=" * 55)
print("STEP 1 COMPLETE")
print("=" * 55)