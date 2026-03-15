"""
Steps 4 & 5 - Training Multiple Models + Evaluation
Models: Naive Bayes, Logistic Regression, Linear SVM, Random Forest, k-NN
"""

import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes      import MultinomialNB
from sklearn.linear_model     import LogisticRegression, SGDClassifier
from sklearn.ensemble         import RandomForestClassifier
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.metrics          import (accuracy_score, f1_score,
                                       precision_score, recall_score,
                                       confusion_matrix, classification_report)
warnings.filterwarnings('ignore')

os.makedirs('../outputs', exist_ok=True)
os.makedirs('../models',  exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────
print("=" * 50)
print("STEPS 4 & 5: MODEL TRAINING + EVALUATION")
print("=" * 50)

print("\nLoading features and labels...")
with open('../dataset/features.pkl', 'rb') as f:
    features = pickle.load(f)
with open('../dataset/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = features['tfidf']['X_train']
X_test  = features['tfidf']['X_test']
y_train = np.array(data['train_labels'])
y_test  = np.array(data['test_labels'])
category_names = data['category_names']

print(f"  X_train shape: {X_train.shape}")
print(f"  X_test  shape: {X_test.shape}")

# ── Define Models ─────────────────────────────────────────────
MODELS = {
    'Naive Bayes'        : MultinomialNB(alpha=0.1),
    'Logistic Regression': LogisticRegression(max_iter=1000, C=5.0, solver='lbfgs', n_jobs=-1),
    'Linear SVM'         : SGDClassifier(loss='hinge', alpha=1e-4, max_iter=200, random_state=42),
    'Random Forest'      : RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'k-NN'               : KNeighborsClassifier(n_neighbors=7, metric='cosine', algorithm='brute'),
}

# ── Train and Evaluate ────────────────────────────────────────
print("\nTraining models...")
print(f"  {'Model':<25} {'Time':>8} {'Accuracy':>10} {'F1 Macro':>10}")
print("  " + "-" * 56)

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

# ── Save Models ───────────────────────────────────────────────
print("\nSaving models...")
for name, r in results.items():
    fname = name.lower().replace(' ', '_').replace('-', '_')
    path  = f'../models/{fname}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(r['model'], f)
    print(f"  Saved: models/{fname}.pkl")

# Save results (without model objects)
results_save = {k: {kk: vv for kk, vv in v.items() if kk != 'model'}
                for k, v in results.items()}
with open('../dataset/all_results.pkl', 'wb') as f:
    pickle.dump(results_save, f)

# ── Results Table ─────────────────────────────────────────────
print("\nModel Comparison Table:")
print(f"  {'Model':<25} {'Accuracy':>10} {'F1 Macro':>10} {'Precision':>11} {'Recall':>8} {'Time':>8}")
print("  " + "-" * 76)
for name, r in results.items():
    print(f"  {name:<25} {r['accuracy']*100:>9.2f}%  {r['f1_macro']*100:>8.2f}%  "
          f"{r['precision']*100:>9.2f}%  {r['recall']*100:>6.2f}%  {r['time']:>6.1f}s")

# Save to CSV
rows = [{'Model': n, 'Accuracy': round(r['accuracy']*100,2),
         'F1 Macro': round(r['f1_macro']*100,2),
         'Precision': round(r['precision']*100,2),
         'Recall': round(r['recall']*100,2),
         'Time(s)': round(r['time'],2)}
        for n, r in results.items()]
pd.DataFrame(rows).to_csv('../outputs/step5_results.csv', index=False)
print("\n  Saved: outputs/step5_results.csv")

# ── Plot: Accuracy Bar Chart ──────────────────────────────────
names      = list(results.keys())
accuracies = [results[n]['accuracy'] * 100 for n in names]
best       = names[accuracies.index(max(accuracies))]
colors     = ['green' if n == best else 'steelblue' for n in names]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(names, accuracies, color=colors, edgecolor='black', width=0.5)
ax.set_title('Model Accuracy Comparison')
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(0, 105)
for bar, val in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.2f}%', ha='center', fontsize=10, fontweight='bold')
ax.tick_params(axis='x', rotation=15)
plt.tight_layout()
plt.savefig('../outputs/step5_accuracy_bar.png', dpi=150)
plt.close()
print("  Saved: outputs/step5_accuracy_bar.png")

# ── Plot: Grouped Bar Chart ───────────────────────────────────
x     = np.arange(len(names))
width = 0.2
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - 1.5*width, [results[n]['accuracy']*100  for n in names], width, label='Accuracy',  color='steelblue', edgecolor='black')
ax.bar(x - 0.5*width, [results[n]['f1_macro']*100  for n in names], width, label='F1 Macro',  color='green',     edgecolor='black')
ax.bar(x + 0.5*width, [results[n]['precision']*100 for n in names], width, label='Precision', color='orange',    edgecolor='black')
ax.bar(x + 1.5*width, [results[n]['recall']*100    for n in names], width, label='Recall',    color='red',       edgecolor='black')
ax.set_title('Model Comparison - All Metrics')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15)
ax.set_ylabel('Score (%)')
ax.set_ylim(0, 105)
ax.legend()
plt.tight_layout()
plt.savefig('../outputs/step5_model_comparison.png', dpi=150)
plt.close()
print("  Saved: outputs/step5_model_comparison.png")

# ── Plot: Confusion Matrix for Best Model ────────────────────
print(f"\nGenerating confusion matrix for best model: {best}")
cm_norm = results[best]['cm'].astype(float)
cm_norm = cm_norm / cm_norm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=category_names, yticklabels=category_names,
            linewidths=0.3, ax=ax)
ax.set_title(f'Confusion Matrix - {best} (normalised)')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.tick_params(axis='x', rotation=45, labelsize=7)
ax.tick_params(axis='y', rotation=0,  labelsize=7)
plt.tight_layout()
plt.savefig('../outputs/step5_confusion_matrix.png', dpi=130)
plt.close()
print("  Saved: outputs/step5_confusion_matrix.png")

print("\n" + "=" * 50)
print("STEPS 4 & 5 COMPLETE")
print("=" * 50)