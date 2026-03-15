"""
Step 4d - Tuned LinearSVC + Voting Ensemble
Tries multiple C values for LinearSVC to find best,
then combines top 3 models with voting ensemble.
"""

import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm           import LinearSVC
from sklearn.linear_model  import LogisticRegression
from sklearn.naive_bayes   import MultinomialNB
from sklearn.calibration   import CalibratedClassifierCV
from sklearn.ensemble      import VotingClassifier
from sklearn.metrics       import (accuracy_score, f1_score,
                                    precision_score, recall_score,
                                    confusion_matrix)
warnings.filterwarnings('ignore')

os.makedirs('../outputs', exist_ok=True)
os.makedirs('../models',  exist_ok=True)

print("=" * 50)
print("STEP 4d: TUNED LinearSVC + VOTING ENSEMBLE")
print("=" * 50)

# ── Load Features ─────────────────────────────────────────────
print("\nLoading TF-IDF features...")
with open('../dataset/features.pkl', 'rb') as f:
    features = pickle.load(f)
with open('../dataset/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train        = features['tfidf']['X_train']
X_test         = features['tfidf']['X_test']
y_train        = np.array(data['train_labels'])
y_test         = np.array(data['test_labels'])
category_names = data['category_names']

# ── Part 1: Tune LinearSVC ────────────────────────────────────
print("\n--- Part 1: Tuning LinearSVC ---")
print(f"\n  {'C value':>10} {'Accuracy':>10} {'F1 Macro':>10} {'Time':>8}")
print("  " + "-" * 42)

C_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
svc_results = {}

for C in C_values:
    t0    = time.time()
    model = LinearSVC(C=C, max_iter=2000, random_state=42)
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average='macro')

    svc_results[C] = {'model': model, 'accuracy': acc, 'f1': f1, 'time': elapsed}
    print(f"  C={C:>8}  {acc*100:>9.2f}%  {f1*100:>8.2f}%  {elapsed:>6.1f}s")

best_C   = max(svc_results, key=lambda c: svc_results[c]['accuracy'])
best_svc = svc_results[best_C]
print(f"\n  Best C value : {best_C}")
print(f"  Accuracy     : {best_svc['accuracy']*100:.2f}%")
print(f"  F1 Macro     : {best_svc['f1']*100:.2f}%")

# ── Part 2: Voting Ensemble ───────────────────────────────────
print("\n--- Part 2: Voting Ensemble ---")

# LinearSVC needs calibration for soft voting
svc_calibrated = CalibratedClassifierCV(
    LinearSVC(C=best_C, max_iter=2000, random_state=42), cv=3)
lr  = LogisticRegression(max_iter=1000, C=5.0)
nb  = MultinomialNB(alpha=0.1)

ensemble = VotingClassifier(
    estimators=[
        ('svm', svc_calibrated),
        ('lr',  lr),
        ('nb',  nb),
    ],
    voting='soft',
    n_jobs=-1,
)

print("\n  Training Voting Ensemble (SVM + LR + NB)...")
t0 = time.time()
ensemble.fit(X_train, y_train)
elapsed = time.time() - t0

y_pred    = ensemble.predict(X_test)
acc       = accuracy_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall    = recall_score(y_test, y_pred, average='macro', zero_division=0)
cm        = confusion_matrix(y_test, y_pred)

print(f"  Time      : {elapsed:.1f}s")
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  F1 Macro  : {f1*100:.2f}%")
print(f"  Precision : {precision*100:.2f}%")
print(f"  Recall    : {recall*100:.2f}%")

# ── Final Comparison ──────────────────────────────────────────
print("\n" + "=" * 60)
print("UPDATED FULL RESULTS")
print("=" * 60)

with open('../dataset/all_results.pkl', 'rb') as f:
    all_results = pickle.load(f)

# Add new results
all_results[f'LinearSVC (C={best_C})'] = {
    'accuracy' : best_svc['accuracy'],
    'f1_macro' : best_svc['f1'],
    'precision': precision_score(y_test, svc_results[best_C]['model'].predict(X_test),
                                  average='macro', zero_division=0),
    'recall'   : recall_score(y_test, svc_results[best_C]['model'].predict(X_test),
                               average='macro', zero_division=0),
    'time'     : best_svc['time'],
    'cm'       : confusion_matrix(y_test, svc_results[best_C]['model'].predict(X_test)),
}
all_results['Voting Ensemble'] = {
    'accuracy' : acc,
    'f1_macro' : f1,
    'precision': precision,
    'recall'   : recall,
    'time'     : elapsed,
    'cm'       : cm,
}

# Save updated results
with open('../dataset/all_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)

print(f"\n  {'Model':<30} {'Accuracy':>10} {'F1 Macro':>10}")
print("  " + "-" * 54)
for name, r in sorted(all_results.items(), key=lambda x: -x[1]['accuracy']):
    marker = ' <-- BEST' if r['accuracy'] == max(v['accuracy'] for v in all_results.values()) else ''
    print(f"  {name:<30} {r['accuracy']*100:>9.2f}%  {r['f1_macro']*100:>8.2f}%{marker}")

# ── Save Models ───────────────────────────────────────────────
with open(f'../models/linear_svc_tuned.pkl', 'wb') as f:
    pickle.dump(svc_results[best_C]['model'], f)
with open('../models/voting_ensemble.pkl', 'wb') as f:
    pickle.dump(ensemble, f)
print(f"\n  Saved: models/linear_svc_tuned.pkl")
print(f"  Saved: models/voting_ensemble.pkl")

# ── Plot: C value tuning ──────────────────────────────────────
c_labels = [str(c) for c in C_values]
c_accs   = [svc_results[c]['accuracy']*100 for c in C_values]
best_color = ['green' if c == best_C else 'steelblue' for c in C_values]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(c_labels, c_accs, color=best_color, edgecolor='black', width=0.5)
ax.set_title('LinearSVC — Accuracy vs C value (Hyperparameter Tuning)')
ax.set_xlabel('C value')
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(0, 105)
for bar, val in zip(bars, c_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.2f}%', ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('../outputs/step4d_svc_tuning.png', dpi=150)
plt.close()
print("  Saved: outputs/step4d_svc_tuning.png")

# ── Confusion Matrix for best model ──────────────────────────
best_overall = max(all_results, key=lambda n: all_results[n]['accuracy'])
print(f"\nGenerating confusion matrix for {best_overall}...")
cm_norm = all_results[best_overall]['cm'].astype(float)
cm_norm = cm_norm / cm_norm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=category_names, yticklabels=category_names,
            linewidths=0.3, ax=ax)
ax.set_title(f'Confusion Matrix — {best_overall} (normalised)')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.tick_params(axis='x', rotation=45, labelsize=7)
ax.tick_params(axis='y', rotation=0,  labelsize=7)
plt.tight_layout()
plt.savefig('../outputs/step4d_best_confusion_matrix.png', dpi=130)
plt.close()
print(f"  Saved: outputs/step4d_best_confusion_matrix.png")

print("\n" + "=" * 50)
print("STEP 4d COMPLETE")
print("=" * 50)