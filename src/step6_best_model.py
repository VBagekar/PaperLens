"""
PaperLens - Research Paper Simplifier
Step 6 - Best Model Selection
Selects best summarizer based on composite ROUGE score.
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('../outputs', exist_ok=True)

print("=" * 55)
print("STEP 6: BEST MODEL SELECTION")
print("=" * 55)

# ── Load Results ──────────────────────────────────────────────
print("\nLoading all model results...")
with open('../dataset/model_results.pkl', 'rb') as f:
    results = pickle.load(f)

print(f"  Total models : {len(results)}")

# ── Composite Score ───────────────────────────────────────────
# ROUGE-1: unigram overlap (content coverage)
# ROUGE-2: bigram overlap  (fluency)
# ROUGE-L: longest common subsequence (structure)
# Weights: R1=0.40, R2=0.35, RL=0.25
print("\nComputing composite ROUGE scores...")
print(f"\n  {'Model':<25} {'ROUGE-1':>9} {'ROUGE-2':>9} {'ROUGE-L':>9} {'Composite':>11}")
print("  " + "-" * 66)

scored = {}
for name, r in results.items():
    composite = (0.40 * r['rouge1'] +
                 0.35 * r['rouge2'] +
                 0.25 * r['rougeL'])
    scored[name] = {**r, 'composite': composite}

sorted_models = sorted(scored.items(), key=lambda x: -x[1]['composite'])
for name, r in sorted_models:
    marker = ' <-- BEST' if name == sorted_models[0][0] else ''
    print(f"  {name:<25} {r['rouge1']:>8.4f}  {r['rouge2']:>8.4f}  "
          f"{r['rougeL']:>8.4f}  {r['composite']:>10.4f}{marker}")

# ── Best Model ────────────────────────────────────────────────
best_name = sorted_models[0][0]
best      = sorted_models[0][1]

print(f"\n  {'='*52}")
print(f"  BEST MODEL  : {best_name}")
print(f"  ROUGE-1     : {best['rouge1']:.4f}")
print(f"  ROUGE-2     : {best['rouge2']:.4f}")
print(f"  ROUGE-L     : {best['rougeL']:.4f}")
print(f"  Composite   : {best['composite']:.4f}")
print(f"  {'='*52}")

# ── Save best model config ────────────────────────────────────
best_config = {
    'model'      : best_name,
    'rouge1'     : round(best['rouge1'], 4),
    'rouge2'     : round(best['rouge2'], 4),
    'rougeL'     : round(best['rougeL'], 4),
    'composite'  : round(best['composite'], 4),
    'n_sentences': 6,
    'lambda'     : 0.6,
    'w_tfidf'    : 0.5,
    'w_pos'      : 0.3,
    'w_len'      : 0.1,
    'w_kw'       : 0.1,
    'all_models' : {n: {
                     'rouge1'   : round(r['rouge1'], 4),
                     'rouge2'   : round(r['rouge2'], 4),
                     'rougeL'   : round(r['rougeL'], 4),
                     'composite': round(r['composite'], 4)}
                    for n, r in scored.items()},
}
with open('../outputs/best_model_meta.json', 'w') as f:
    json.dump(best_config, f, indent=2)
print("\n  Saved: outputs/best_model_meta.json")

# ── Plot 1: Composite score bar ───────────────────────────────
names      = [n for n, _ in sorted_models]
composites = [r['composite'] for _, r in sorted_models]
colors     = ['green' if n == best_name else 'steelblue' for n in names]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(names, composites, color=colors, edgecolor='black', width=0.5)
ax.set_title('Best Model Selection — Composite ROUGE Score\n'
             '(40% ROUGE-1 + 35% ROUGE-2 + 25% ROUGE-L)')
ax.set_ylabel('Composite Score')
ax.set_ylim(0, max(composites) * 1.2)
for bar, val in zip(bars, composites):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{val:.4f}', ha='center', fontsize=9, fontweight='bold')
ax.tick_params(axis='x', rotation=20)
ax.annotate('BEST', xy=(0, composites[0]),
            xytext=(0, composites[0] + 0.005),
            ha='center', fontsize=11,
            color='green', fontweight='bold')
plt.tight_layout()
plt.savefig('../outputs/step6_best_model.png', dpi=150)
plt.close()
print("  Saved: outputs/step6_best_model.png")

# ── Plot 2: ROUGE comparison grouped bar ─────────────────────
x     = np.arange(len(names))
width = 0.25
r1    = [scored[n]['rouge1'] for n in names]
r2    = [scored[n]['rouge2'] for n in names]
rl    = [scored[n]['rougeL'] for n in names]

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - width, r1, width, label='ROUGE-1',
       color='steelblue', edgecolor='black')
ax.bar(x,         r2, width, label='ROUGE-2',
       color='green',     edgecolor='black')
ax.bar(x + width, rl, width, label='ROUGE-L',
       color='orange',    edgecolor='black')
ax.set_title('All Models — ROUGE Score Comparison')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=20, ha='right')
ax.set_ylabel('ROUGE Score')
ax.set_ylim(0, max(r1) * 1.25)
ax.legend()
plt.tight_layout()
plt.savefig('../outputs/step6_rouge_comparison.png', dpi=150)
plt.close()
print("  Saved: outputs/step6_rouge_comparison.png")

# ── Plot 3: Radar chart ───────────────────────────────────────
categories = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
N          = len(categories)
angles     = [n / float(N) * 2 * np.pi for n in range(N)]
angles    += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
colors_r  = plt.cm.tab10(np.linspace(0, 0.9, len(scored)))

for (name, r), color in zip(scored.items(), colors_r):
    values  = [r['rouge1'], r['rouge2'], r['rougeL']]
    values += values[:1]
    lw = 3 if name == best_name else 1.5
    ax.plot(angles, values, linewidth=lw, label=name, color=color)
    ax.fill(angles, values, alpha=0.04, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12)
ax.set_title('Model Comparison — Radar Chart', fontsize=13, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15), fontsize=7)
plt.tight_layout()
plt.savefig('../outputs/step6_radar_chart.png', dpi=150)
plt.close()
print("  Saved: outputs/step6_radar_chart.png")

print("\n" + "=" * 55)
print("STEP 6 COMPLETE")
print("=" * 55)