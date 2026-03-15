"""
Step 4c - Classification using Sentence-BERT features
Trains multiple models on SBERT embeddings and adds
results to the full comparison table.
"""

import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model  import LogisticRegression
from sklearn.svm           import SVC, LinearSVC
from sklearn.ensemble      import RandomForestClassifier
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.metrics       import (accuracy_score, f1_score,
                                    precision_score, recall_score,
                                    confusion_matrix)
warnings.filterwarnings('ignore')

os.makedirs('../outputs', exist_ok=True)
os.makedirs('../models',  exist_ok=True)

print("=" * 50)
print("STEP 4c: SBERT CLASSIFICATION")
print("=" * 50)

# ── Load SBERT Features ───────────────────────────────────────
print("\nLoading SBERT features...")
with open('../dataset/sbert_features.pkl', 'rb') as f:
    sbert = pickle.load(f)

X_train        = sbert['X_train']
X_test         = sbert['X_test']
y_train        = np.array(sbert['train_labels'])
y_test         = np.array(sbert['test_labels'])
category_names = sbert['category_names']

print(f"  X_train shape : {X_train.shape}")
print(f"  X_test  shape : {X_test.shape}")

# ── Models ────────────────────────────────────────────────────
MODELS = {
    'LR + SBERT'     : LogisticRegression(max_iter=1000, C=5.0),
    'LinearSVC + SBERT' : LinearSVC(max_iter=2000, C=1.0),
    'RBF SVM + SBERT': SVC(kernel='rbf', C=10.0, gamma='scale'),
    'RF + SBERT'     : RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'kNN + SBERT'    : KNeighborsClassifier(n_neighbors=7, metric='cosine', algorithm='brute'),
}

# ── Train and Evaluate ────────────────────────────────────────
print("\nTraining models on SBERT features...")
print(f"  {'Model':<25} {'Time':>8} {'Accuracy':>10} {'F1 Macro':>10}")
print("  " + "-" * 58)

results = {}
for name, model in MODELS.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_pred    = model.predict(X_test)
    acc       = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall    = recall_score(y_test, y_pred, average='macro', zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)

    results[name] = {
        'model'    : model,
        'accuracy' : acc,
        'f1_macro' : f1,
        'precision': precision,
        'recall'   : recall,
        'cm'       : cm,
        'time'     : elapsed,
    }

    print(f"  {name:<25} {elapsed:>6.1f}s  {acc*100:>8.2f}%  {f1*100:>8.2f}%")

# ── Best SBERT Model ──────────────────────────────────────────
best_name  = max(results, key=lambda n: results[n]['accuracy'])
best_acc   = results[best_name]['accuracy']
print(f"\n  Best SBERT model : {best_name}")
print(f"  Accuracy         : {best_acc*100:.2f}%")

with open('../models/best_sbert_model.pkl', 'wb') as f:
    pickle.dump(results[best_name]['model'], f)
print(f"  Saved: models/best_sbert_model.pkl")

# ── Full Comparison Table ─────────────────────────────────────
print("\n" + "=" * 75)
print("COMPLETE MODEL COMPARISON — TF-IDF vs Word2Vec vs SBERT")
print("=" * 75)

with open('../dataset/all_results.pkl', 'rb') as f:
    tfidf_results = pickle.load(f)

print(f"\n  {'Model':<25} {'Features':<10} {'Accuracy':>10} {'F1 Macro':>10} {'Precision':>11} {'Recall':>8}")
print("  " + "-" * 78)

for name, r in tfidf_results.items():
    print(f"  {name:<25} {'TF-IDF':<10} {r['accuracy']*100:>9.2f}%  "
          f"{r['f1_macro']*100:>8.2f}%  {r['precision']*100:>9.2f}%  {r['recall']*100:>6.2f}%")

for name, r in results.items():
    marker = ' <-- BEST' if name == best_name and r['accuracy'] > 0.86 else ''
    print(f"  {name:<25} {'SBERT':<10} {r['accuracy']*100:>9.2f}%  "
          f"{r['f1_macro']*100:>8.2f}%  {r['precision']*100:>9.2f}%  {r['recall']*100:>6.2f}%{marker}")

# ── Save Full CSV ─────────────────────────────────────────────
rows = []
for name, r in tfidf_results.items():
    rows.append({'Model': name, 'Features': 'TF-IDF',
                 'Accuracy': round(r['accuracy']*100, 2),
                 'F1 Macro': round(r['f1_macro']*100, 2),
                 'Precision': round(r['precision']*100, 2),
                 'Recall': round(r['recall']*100, 2)})
for name, r in results.items():
    rows.append({'Model': name, 'Features': 'SBERT',
                 'Accuracy': round(r['accuracy']*100, 2),
                 'F1 Macro': round(r['f1_macro']*100, 2),
                 'Precision': round(r['precision']*100, 2),
                 'Recall': round(r['recall']*100, 2)})

df = pd.DataFrame(rows)
df.to_csv('../outputs/full_model_comparison.csv', index=False)
print("\n  Saved: outputs/full_model_comparison.csv")

# ── Plot: Full Comparison ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 7))
labels  = [r['Model'] + '\n(' + r['Features'] + ')' for _, r in df.iterrows()]
accs    = df['Accuracy'].tolist()
best_a  = max(accs)
colors  = []
for _, r in df.iterrows():
    if r['Accuracy'] == best_a:
        colors.append('green')
    elif r['Features'] == 'TF-IDF':
        colors.append('steelblue')
    elif r['Features'] == 'SBERT':
        colors.append('darkorange')
    else:
        colors.append('gray')

bars = ax.bar(labels, accs, color=colors, edgecolor='black', width=0.6)
ax.set_title('Complete Model Comparison — TF-IDF vs Word2Vec vs SBERT', fontsize=13)
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(0, 105)
for bar, val in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', fontsize=8, fontweight='bold')
ax.tick_params(axis='x', rotation=15, labelsize=7)

p1 = mpatches.Patch(color='steelblue',  label='TF-IDF')
p2 = mpatches.Patch(color='darkorange', label='SBERT')
p3 = mpatches.Patch(color='green',      label='Best Model')
ax.legend(handles=[p1, p2, p3], fontsize=10)
plt.tight_layout()
plt.savefig('../outputs/step4c_full_comparison.png', dpi=150)
plt.close()
print("  Saved: outputs/step4c_full_comparison.png")

# ── Confusion Matrix for Best SBERT Model ────────────────────
print(f"\nGenerating confusion matrix for {best_name}...")
cm_norm = results[best_name]['cm'].astype(float)
cm_norm = cm_norm / cm_norm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Oranges',
            xticklabels=category_names, yticklabels=category_names,
            linewidths=0.3, ax=ax)
ax.set_title(f'Confusion Matrix — {best_name} (normalised)')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.tick_params(axis='x', rotation=45, labelsize=7)
ax.tick_params(axis='y', rotation=0,  labelsize=7)
plt.tight_layout()
plt.savefig('../outputs/step4c_sbert_confusion_matrix.png', dpi=130)
plt.close()
print(f"  Saved: outputs/step4c_sbert_confusion_matrix.png")

print("\n" + "=" * 50)
print("STEP 4c COMPLETE")
print("=" * 50)