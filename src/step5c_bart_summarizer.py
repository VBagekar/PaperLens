"""
PaperLens - Research Paper Simplifier
Step 5c - Abstractive Summarization using BART
Uses facebook/bart-large-cnn pre-trained on CNN/DailyMail
for abstractive summarization of research papers.
Compared against all extractive models using ROUGE scores.
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

os.makedirs('../outputs', exist_ok=True)

print("=" * 55)
print("STEP 5c: BART ABSTRACTIVE SUMMARIZATION")
print("=" * 55)

# ── Device ────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n  Device : {device}")
if device.type == 'cuda':
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")

# ── Load BART ─────────────────────────────────────────────────
print("\nLoading BART model (sshleifer/distilbart-cnn-12-6)...")
print("  (First run downloads ~900MB — please wait...)")

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
tokenizer  = BartTokenizer.from_pretrained(MODEL_NAME)
model      = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
model      = model.to(device)
model.eval()
print("  Model loaded successfully")

# ── Summarization Function ────────────────────────────────────
def summarize_bart(text, max_input=1024, max_output=150, min_output=50):
    """Generate abstractive summary using BART."""
    # Truncate input to max tokens
    inputs = tokenizer(
        text,
        max_length     = max_input,
        truncation     = True,
        return_tensors = 'pt'
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            inputs['input_ids'],
            attention_mask  = inputs['attention_mask'],
            max_length      = max_output,
            min_length      = min_output,
            length_penalty  = 2.0,
            num_beams       = 4,
            early_stopping  = True,
        )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# ── Load Data ─────────────────────────────────────────────────
print("\nLoading preprocessed data...")
with open('../dataset/val_processed.pkl', 'rb') as f:
    val_processed = pickle.load(f)
with open('../dataset/val_abstracts.pkl', 'rb') as f:
    val_abstracts = pickle.load(f)

print(f"  Val documents : {len(val_processed):,}")

# ── Evaluate BART ─────────────────────────────────────────────
N_EVAL  = 100  # 100 docs — enough for reliable ROUGE estimate
scorer  = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

print(f"\nEvaluating BART on {N_EVAL} documents...")
print("  (This will take 10-20 minutes on GPU...)")

r1_scores, r2_scores, rl_scores = [], [], []
sample_summaries = []

for i in range(min(N_EVAL, len(val_processed))):
    text      = val_processed[i]['article_clean']
    reference = val_abstracts[i] if i < len(val_abstracts) else ""
    if not reference or not text:
        continue

    summary = summarize_bart(text)
    scores  = scorer.score(reference, summary)

    r1_scores.append(scores['rouge1'].fmeasure)
    r2_scores.append(scores['rouge2'].fmeasure)
    rl_scores.append(scores['rougeL'].fmeasure)

    if i < 3:
        sample_summaries.append({
            'reference': reference[:300],
            'bart'     : summary,
        })

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{N_EVAL} — "
              f"ROUGE-1: {np.mean(r1_scores):.4f}  "
              f"ROUGE-2: {np.mean(r2_scores):.4f}  "
              f"ROUGE-L: {np.mean(rl_scores):.4f}")

# ── Results ───────────────────────────────────────────────────
bart_r1 = np.mean(r1_scores)
bart_r2 = np.mean(r2_scores)
bart_rl = np.mean(rl_scores)

print(f"\n  {'='*50}")
print(f"  BART Results:")
print(f"    ROUGE-1 : {bart_r1:.4f}")
print(f"    ROUGE-2 : {bart_r2:.4f}")
print(f"    ROUGE-L : {bart_rl:.4f}")
print(f"  {'='*50}")

# ── Sample summaries ──────────────────────────────────────────
print("\nSample BART summaries:")
for i, s in enumerate(sample_summaries):
    print(f"\n  [{i+1}] Reference : {s['reference'][:200]}...")
    print(f"  [{i+1}] BART      : {s['bart']}")

# ── Update model results ──────────────────────────────────────
print("\nUpdating model results...")
with open('../dataset/model_results.pkl', 'rb') as f:
    model_results = pickle.load(f)

model_results['BART (Abstractive)'] = {
    'rouge1': bart_r1,
    'rouge2': bart_r2,
    'rougeL': bart_rl,
}
with open('../dataset/model_results.pkl', 'wb') as f:
    pickle.dump(model_results, f)

# ── Save BART metrics ─────────────────────────────────────────
with open('../outputs/bart_metrics.json', 'w') as f:
    json.dump({
        'model'  : MODEL_NAME,
        'rouge1' : round(bart_r1, 4),
        'rouge2' : round(bart_r2, 4),
        'rougeL' : round(bart_rl, 4),
        'n_eval' : N_EVAL,
    }, f, indent=2)
print("  Saved: outputs/bart_metrics.json")

# ── Full comparison plot ───────────────────────────────────────
print("\nGenerating comparison chart...")
names  = list(model_results.keys())
r1     = [model_results[n]['rouge1'] for n in names]
r2     = [model_results[n]['rouge2'] for n in names]
rl     = [model_results[n]['rougeL'] for n in names]
best   = names[r1.index(max(r1))]
colors = ['green' if n == best else
          ('darkorange' if 'BART' in n else 'steelblue')
          for n in names]

x     = np.arange(len(names))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 6))
b1 = ax.bar(x - width, r1, width, label='ROUGE-1',
            color=[c if i == 0 else 'steelblue'
                   for i, c in enumerate(colors)],
            edgecolor='black')
b2 = ax.bar(x,         r2, width, label='ROUGE-2',
            color='lightgreen', edgecolor='black')
b3 = ax.bar(x + width, rl, width, label='ROUGE-L',
            color='lightsalmon', edgecolor='black')

ax.set_title('All Models — ROUGE Score Comparison\n(Extractive vs Abstractive)')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15)
ax.set_ylabel('ROUGE Score')
ax.set_ylim(0, max(r1) * 1.25)
ax.legend()

for bar in b1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
            f'{h:.3f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('../outputs/step5c_full_comparison.png', dpi=150)
plt.close()
print("  Saved: outputs/step5c_full_comparison.png")

print("\n" + "=" * 55)
print("STEP 5c COMPLETE")
print("=" * 55)