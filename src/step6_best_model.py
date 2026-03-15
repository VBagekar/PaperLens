"""
Step 6 - Best Model Selection
Compares ALL models using a composite weighted score
and selects the overall best one.
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs('../outputs', exist_ok=True)

print("=" * 50)
print("STEP 6: BEST MODEL SELECTION")
print("=" * 50)

# ── Load All Results ──────────────────────────────────────────
print("\nLoading all results...")
with open('../dataset/all_results.pkl', 'rb') as f:
    all_results = pickle.load(f)

print(f"  Total models evaluated : {len(all_results)}")

# ── Composite Score ───────────────────────────────────────────
# 0.40 x Accuracy + 0.35 x F1 + 0.15 x Precision + 0.10 x Speed
print("\nComputing composite scores...")

times     = [r['time'] for r in all_results.values()]
max_time  = max(times)
min_time  = min(times)
time_range = max_time - min_time or 1

scored = {}
for name, r in all_results.items():
    time_score = 1 - (r['time'] - min_time) / time_range
    composite  = (0.40 * r['accuracy']  +
                  0.35 * r['f1_macro']  +
                  0.15 * r['precision'] +
                  0.10 * time_score)
    scored[name] = {**r, 'time_score': time_score, 'composite': composite}

# ── Print Table ───────────────────────────────────────────────
print(f"\n  {'Model':<30} {'Accuracy':>10} {'F1 Macro':>10} {'Composite':>12}")
print("  " + "-" * 66)

sorted_models = sorted(scored.items(), key=lambda x: -x[1]['composite'])
for name, r in sorted_models:
    print(f"  {name:<30} {r['accuracy']*100:>9.2f}%  "
          f"{r['f1_macro']*100:>8.2f}%  {r['composite']*100:>10.2f}%")

# ── Best Model ────────────────────────────────────────────────
best_name = sorted_models[0][0]
best      = sorted_models[0][1]

print(f"\n  {'='*50}")
print(f"  BEST MODEL : {best_name}")
print(f"  Accuracy   : {best['accuracy']*100:.2f}%")
print(f"  F1 Macro   : {best['f1_macro']*100:.2f}%")
print(f"  Precision  : {best['precision']*100:.2f}%")
print(f"  Recall     : {best['recall']*100:.2f}%")
print(f"  Composite  : {best['composite']*100:.2f}%")
print(f"  {'='*50}")

# ── Save Best Model ───────────────────────────────────────────
print("\nSaving best model...")
with open('../models/linear_svc_tuned.pkl', 'rb') as f:
    best_model = pickle.load(f)
with open('../models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("  Saved: models/best_model.pkl")

# Save metadata
meta = {
    'best_model_name': best_name,
    'accuracy'       : round(best['accuracy']*100, 2),
    'f1_macro'       : round(best['f1_macro']*100, 2),
    'precision'      : round(best['precision']*100, 2),
    'recall'         : round(best['recall']*100, 2),
    'composite_score': round(best['composite']*100, 2),
    'all_models'     : {n: {'accuracy': round(r['accuracy']*100,2),
                             'f1_macro': round(r['f1_macro']*100,2)}
                        for n, r in scored.items()},
}
with open('../outputs/best_model_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
print("  Saved: outputs/best_model_meta.json")

# ── Plot: Composite Score Bar ─────────────────────────────────
names      = [n for n, _ in sorted_models]
composites = [r['composite']*100 for _, r in sorted_models]
colors     = ['green' if n == best_name else 'steelblue' for n in names]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(names, composites, color=colors, edgecolor='black', width=0.5)
ax.set_title('Best Model Selection — Composite Score\n(40% Accuracy + 35% F1 + 15% Precision + 10% Speed)')
ax.set_ylabel('Composite Score (%)')
ax.set_ylim(0, 105)
for bar, val in zip(bars, composites):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', fontsize=8, fontweight='bold')
ax.tick_params(axis='x', rotation=20, labelsize=8)
ax.annotate('BEST', xy=(0, composites[0]),
            xytext=(0, composites[0] + 3),
            ha='center', fontsize=10, color='green', fontweight='bold')
plt.tight_layout()
plt.savefig('../outputs/step6_best_model.png', dpi=150)
plt.close()
print("  Saved: outputs/step6_best_model.png")

# ── Plot: Radar Chart ─────────────────────────────────────────
categories = ['Accuracy', 'F1 Macro', 'Precision', 'Recall']
N          = len(categories)
angles     = [n / float(N) * 2 * np.pi for n in range(N)]
angles    += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
colors_r  = plt.cm.tab10(np.linspace(0, 0.9, len(scored)))

for (name, r), color in zip(scored.items(), colors_r):
    values  = [r['accuracy'], r['f1_macro'], r['precision'], r['recall']]
    values += values[:1]
    lw = 3 if name == best_name else 1.5
    ax.plot(angles, values, linewidth=lw, label=name, color=color)
    ax.fill(angles, values, alpha=0.03, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 1)
ax.set_title('All Models — Radar Chart', fontsize=13, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.5, 1.15), fontsize=7)
plt.tight_layout()
plt.savefig('../outputs/step6_radar_chart.png', dpi=150)
plt.close()
print("  Saved: outputs/step6_radar_chart.png")

print("\n" + "=" * 50)
print("STEP 6 COMPLETE")
print("=" * 50)