"""
Step 4b - Classification using Word2Vec features
Trains multiple models on Word2Vec document vectors
and compares with TF-IDF results.
"""

import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model     import LogisticRegression, SGDClassifier
from sklearn.svm              import SVC
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.metrics          import (accuracy_score, f1_score,
                                       precision_score, recall_score,
                                       confusion_matrix)
warnings.filterwarnings('ignore')

os.makedirs('../outputs', exist_ok=True)
os.makedirs('../models',  exist_ok=True)

print("=" * 50)
print("STEP 4b: WORD2VEC CLASSIFICATION")
print("=" * 50)

# ── Load Word2Vec Features ────────────────────────────────────
print("\nLoading Word2Vec features...")
with open('../dataset/w2v_features.pkl', 'rb') as f:
    w2v = pickle.load(f)

X_train        = w2v['X_train']
X_test         = w2v['X_test']
y_train        = np.array(w2v['train_labels'])
y_test         = np.array(w2v['test_labels'])
category_names = w2v['category_names']

print(f"  X_train shape : {X_train.shape}")
print(f"  X_test  shape : {X_test.shape}")

# ── Models ────────────────────────────────────────────────────
MODELS = {
    'LR + Word2Vec'   : LogisticRegression(max_iter=1000, C=5.0, n_jobs=-1),
    'SVM + Word2Vec'  : SVC(kernel='rbf', C=10.0, gamma='scale'),
    'RF + Word2Vec'   : RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'kNN + Word2Vec'  : KNeighborsClassifier(n_neighbors=7, metric='cosine', algorithm='brute'),
}

# ── Train and Evaluate ────────────────────────────────────────
print("\nTraining models on Word2Vec features...")
print(f"  {'Model':<20} {'Time':>8} {'Accuracy':>10} {'F1 Macro':>10}")
print("  " + "-" * 52)

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

    print(f"  {name:<20} {elapsed:>6.1f}s  {acc*100:>8.2f}%  {f1*100:>8.2f}%")

# ── Save Best W2V Model ───────────────────────────────────────
best_name  = max(results, key=lambda n: results[n]['accuracy'])
best_model = results[best_name]['model']
with open('../models/best_w2v_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"\n  Best Word2Vec model : {best_name}")
print(f"  Accuracy            : {results[best_name]['accuracy']*100:.2f}%")
print(f"  Saved: models/best_w2v_model.pkl")

# ── Full Comparison Table (TF-IDF + Word2Vec) ─────────────────
print("\n" + "=" * 70)
print("FULL MODEL COMPARISON TABLE (TF-IDF + Word2Vec)")
print("=" * 70)

# Load TF-IDF results
with open('../dataset/all_results.pkl', 'rb') as f:
    tfidf_results = pickle.load(f)

print(f"\n  {'Model':<25} {'Features':<12} {'Accuracy':>10} {'F1 Macro':>10} {'Precision':>11} {'Recall':>8}")
print("  " + "-" * 80)

# TF-IDF models
for name, r in tfidf_results.items():
    print(f"  {name:<25} {'TF-IDF':<12} {r['accuracy']*100:>9.2f}%  "
          f"{r['f1_macro']*100:>8.2f}%  {r['precision']*100:>9.2f}%  {r['recall']*100:>6.2f}%")

# Word2Vec models
for name, r in results.items():
    print(f"  {name:<25} {'Word2Vec':<12} {r['accuracy']*100:>9.2f}%  "
          f"{r['f1_macro']*100:>8.2f}%  {r['precision']*100:>9.2f}%  {r['recall']*100:>6.2f}%")

# ── Save Combined CSV ─────────────────────────────────────────
rows = []
for name, r in tfidf_results.items():
    rows.append({'Model': name, 'Features': 'TF-IDF',
                 'Accuracy': round(r['accuracy']*100, 2),
                 'F1 Macro': round(r['f1_macro']*100, 2),
                 'Precision': round(r['precision']*100, 2),
                 'Recall': round(r['recall']*100, 2)})
for name, r in results.items():
    rows.append({'Model': name, 'Features': 'Word2Vec',
                 'Accuracy': round(r['accuracy']*100, 2),
                 'F1 Macro': round(r['f1_macro']*100, 2),
                 'Precision': round(r['precision']*100, 2),
                 'Recall': round(r['recall']*100, 2)})

df = pd.DataFrame(rows)
df.to_csv('../outputs/full_model_comparison.csv', index=False)
print("\n  Saved: outputs/full_model_comparison.csv")

# ── Plot: Full Comparison Bar Chart ──────────────────────────
all_names  = [r['Model'] + '\n(' + r['Features'] + ')' for _, r in df.iterrows()]
all_acc    = df['Accuracy'].tolist()
best_acc   = max(all_acc)
colors     = ['green' if a == best_acc else
              ('steelblue' if 'TF-IDF' in n else 'orange')
              for a, n in zip(all_acc, all_names)]

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(all_names, all_acc, color=colors, edgecolor='black', width=0.6)
ax.set_title('Complete Model Comparison — TF-IDF vs Word2Vec', fontsize=13)
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(0, 105)
for bar, val in zip(bars, all_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', fontsize=8, fontweight='bold')
ax.tick_params(axis='x', rotation=15, labelsize=8)

# Legend
import matplotlib.patches as mpatches
p1 = mpatches.Patch(color='steelblue', label='TF-IDF models')
p2 = mpatches.Patch(color='orange',    label='Word2Vec models')
p3 = mpatches.Patch(color='green',     label='Best model')
ax.legend(handles=[p1, p2, p3])
plt.tight_layout()
plt.savefig('../outputs/step4b_full_comparison.png', dpi=150)
plt.close()
print("  Saved: outputs/step4b_full_comparison.png")

# ── Confusion Matrix for Best W2V Model ──────────────────────
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
plt.savefig('../outputs/step4b_w2v_confusion_matrix.png', dpi=130)
plt.close()
print(f"  Saved: outputs/step4b_w2v_confusion_matrix.png")

print("\n" + "=" * 50)
print("STEP 4b COMPLETE")
print("=" * 50)