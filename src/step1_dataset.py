"""
Step 1 - Dataset Loading and Exploration
Dataset: 20 Newsgroups (20 categories of news posts)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import fetch_20newsgroups

# Create outputs folder if it doesn't exist
os.makedirs('../outputs', exist_ok=True)

# ── Load Dataset ──────────────────────────────────────────────
print("=" * 50)
print("STEP 1: DATASET LOADING AND EXPLORATION")
print("=" * 50)

print("\nLoading 20 Newsgroups dataset...")
train_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
test_data  = fetch_20newsgroups(subset='test',  remove=('headers', 'footers', 'quotes'))

print(f"  Train samples : {len(train_data.data)}")
print(f"  Test  samples : {len(test_data.data)}")
print(f"  Categories    : {len(train_data.target_names)}")

# ── Category Distribution ─────────────────────────────────────
print("\nCategory Distribution:")
train_counts = Counter(train_data.target)
for i, name in enumerate(train_data.target_names):
    print(f"  {i:2d}. {name:<35} {train_counts[i]} samples")

# ── Text Length Stats ─────────────────────────────────────────
lengths = [len(t.split()) for t in train_data.data]
print(f"\nDocument Length Statistics:")
print(f"  Mean   : {np.mean(lengths):.1f} words")
print(f"  Median : {np.median(lengths):.1f} words")
print(f"  Min    : {min(lengths)} words")
print(f"  Max    : {max(lengths)} words")

# ── Plot 1: Category Distribution ────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
categories = train_data.target_names
counts     = [train_counts[i] for i in range(20)]
ax.barh(categories, counts, color='steelblue', edgecolor='black')
ax.set_xlabel('Number of Training Samples')
ax.set_title('20 Newsgroups - Samples per Category')
plt.tight_layout()
plt.savefig('../outputs/step1_category_distribution.png', dpi=150)
plt.close()
print("\nSaved: outputs/step1_category_distribution.png")

# ── Plot 2: Document Length Distribution ─────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(lengths, bins=50, color='steelblue', edgecolor='black')
ax.axvline(np.mean(lengths),   color='red',    linestyle='--', label=f'Mean = {np.mean(lengths):.0f}')
ax.axvline(np.median(lengths), color='orange', linestyle='--', label=f'Median = {np.median(lengths):.0f}')
ax.set_xlabel('Document Length (words)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Document Lengths')
ax.set_xlim(0, 1500)
ax.legend()
plt.tight_layout()
plt.savefig('../outputs/step1_length_distribution.png', dpi=150)
plt.close()
print("Saved: outputs/step1_length_distribution.png")

print("\n" + "=" * 50)
print("STEP 1 COMPLETE")
print("=" * 50)