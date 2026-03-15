"""
Step 4e - DistilBERT Fine-tuning
Fine-tunes distilbert-base-uncased on 20 Newsgroups
for text classification. Uses GPU acceleration.
"""

import os
import time
import json
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (DistilBertTokenizerFast,
                          DistilBertForSequenceClassification,
                          get_linear_schedule_with_warmup)
from torch.optim import AdamW
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics  import (accuracy_score, f1_score,
                               precision_score, recall_score,
                               confusion_matrix)
warnings.filterwarnings('ignore')

os.makedirs('../outputs', exist_ok=True)
os.makedirs('../models',  exist_ok=True)

print("=" * 50)
print("STEP 4e: DISTILBERT FINE-TUNING")
print("=" * 50)

# ── Config ────────────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME  = 'distilbert-base-uncased'
MAX_LEN     = 512
BATCH_SIZE  = 8
EPOCHS      = 5
LR          = 3e-5
GRAD_ACCUM  = 4
NUM_CLASSES = 20

print(f"\n  Device     : {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"  GPU        : {torch.cuda.get_device_name(0)}")
print(f"  Model      : {MODEL_NAME}")
print(f"  Max length : {MAX_LEN} tokens")
print(f"  Batch size : {BATCH_SIZE}")
print(f"  Epochs     : {EPOCHS}")
print(f"  LR         : {LR}")

# ── Load Raw Dataset ──────────────────────────────────────────
print("\nLoading dataset...")
train_data = fetch_20newsgroups(subset='train')
test_data  = fetch_20newsgroups(subset='test')

train_texts  = train_data.data
test_texts   = test_data.data
train_labels = train_data.target.tolist()
test_labels  = test_data.target.tolist()
category_names = train_data.target_names

print(f"  Train : {len(train_texts)}")
print(f"  Test  : {len(test_texts)}")

# ── Tokenizer ─────────────────────────────────────────────────
print("\nLoading tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

# ── Dataset Class ─────────────────────────────────────────────
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(
            texts,
            truncation    = True,
            padding       = True,
            max_length    = max_len,
            return_tensors = 'pt'
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids'     : self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels'        : torch.tensor(self.labels[idx], dtype=torch.long)
        }

print("Tokenizing datasets (this may take a moment)...")
t0 = time.time()
train_dataset = NewsDataset(train_texts, train_labels, tokenizer, MAX_LEN)
test_dataset  = NewsDataset(test_texts,  test_labels,  tokenizer, MAX_LEN)
print(f"  Done in {time.time()-t0:.1f}s")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ── Model ─────────────────────────────────────────────────────
print("\nLoading DistilBERT model...")
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels = NUM_CLASSES
)
model = model.to(DEVICE)
print(f"  Parameters : {sum(p.numel() for p in model.parameters()):,}")

# ── Optimizer + Scheduler ─────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler   = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps   = total_steps // 10,
    num_training_steps = total_steps
)

# ── Training Loop ─────────────────────────────────────────────
print(f"\nTraining for {EPOCHS} epochs...")
print(f"  Steps per epoch : {len(train_loader)}")

history = {'train_loss': [], 'val_acc': [], 'val_f1': []}
best_acc = 0.0

for epoch in range(EPOCHS):
    # ── Train ──
    model.train()
    total_loss = 0
    t0 = time.time()

    for step, batch in enumerate(train_loader):
        input_ids      = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels         = batch['labels'].to(DEVICE)

       
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss / GRAD_ACCUM
        loss.backward()
        total_loss += loss.item() * GRAD_ACCUM

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % 100 == 0:
            print(f"  Epoch {epoch+1} | Step {step+1}/{len(train_loader)} "
                  f"| Loss: {total_loss/(step+1):.4f}")

    avg_loss = total_loss / len(train_loader)

    # ── Evaluate ──
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['labels']

            outputs = model(input_ids=input_ids,
                           attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro')
    elapsed = time.time() - t0

    history['train_loss'].append(avg_loss)
    history['val_acc'].append(acc)
    history['val_f1'].append(f1)

    print(f"\n  Epoch {epoch+1}/{EPOCHS} complete")
    print(f"  Loss     : {avg_loss:.4f}")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  F1 Macro : {f1*100:.2f}%")
    print(f"  Time     : {elapsed:.1f}s\n")

    # Save best model
    if acc > best_acc:
        best_acc = acc
        model.save_pretrained('../models/distilbert_best')
        tokenizer.save_pretrained('../models/distilbert_best')
        print(f"  ✓ Best model saved (accuracy: {best_acc*100:.2f}%)")

# ── Final Evaluation ──────────────────────────────────────────
print("\n" + "=" * 50)
print("FINAL EVALUATION")
print("=" * 50)

precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
recall    = recall_score(all_labels, all_preds,    average='macro', zero_division=0)
cm        = confusion_matrix(all_labels, all_preds)

print(f"\n  Accuracy  : {best_acc*100:.2f}%")
print(f"  F1 Macro  : {f1*100:.2f}%")
print(f"  Precision : {precision*100:.2f}%")
print(f"  Recall    : {recall*100:.2f}%")

# ── Save Results ──────────────────────────────────────────────
with open('../dataset/all_results.pkl', 'rb') as f:
    all_results = pickle.load(f)

all_results['DistilBERT'] = {
    'accuracy' : best_acc,
    'f1_macro' : f1,
    'precision': precision,
    'recall'   : recall,
    'time'     : EPOCHS * len(train_loader) * BATCH_SIZE / 60,
    'cm'       : cm,
}
with open('../dataset/all_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)

# Save metrics
metrics = {
    'model'    : 'DistilBERT',
    'accuracy' : round(best_acc*100, 2),
    'f1_macro' : round(f1*100, 2),
    'precision': round(precision*100, 2),
    'recall'   : round(recall*100, 2),
    'epochs'   : EPOCHS,
}
with open('../outputs/distilbert_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("\n  Saved: outputs/distilbert_metrics.json")

# ── Plot: Training History ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('DistilBERT Training History', fontsize=13)

axes[0].plot(range(1, EPOCHS+1), history['train_loss'], 
             marker='o', color='steelblue', linewidth=2)
axes[0].set_title('Training Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_xticks(range(1, EPOCHS+1))

axes[1].plot(range(1, EPOCHS+1), [a*100 for a in history['val_acc']], 
             marker='o', color='green', linewidth=2, label='Accuracy')
axes[1].plot(range(1, EPOCHS+1), [f*100 for f in history['val_f1']], 
             marker='s', color='orange', linewidth=2, label='F1 Macro')
axes[1].set_title('Validation Metrics')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Score (%)')
axes[1].set_xticks(range(1, EPOCHS+1))
axes[1].legend()

plt.tight_layout()
plt.savefig('../outputs/step4e_distilbert_history.png', dpi=150)
plt.close()
print("  Saved: outputs/step4e_distilbert_history.png")

# ── Confusion Matrix ──────────────────────────────────────────
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
            xticklabels=category_names, yticklabels=category_names,
            linewidths=0.3, ax=ax)
ax.set_title(f'Confusion Matrix — DistilBERT (normalised)')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.tick_params(axis='x', rotation=45, labelsize=7)
ax.tick_params(axis='y', rotation=0,  labelsize=7)
plt.tight_layout()
plt.savefig('../outputs/step4e_distilbert_confusion_matrix.png', dpi=130)
plt.close()
print("  Saved: outputs/step4e_distilbert_confusion_matrix.png")

# ── Final Leaderboard ─────────────────────────────────────────
print("\n" + "=" * 55)
print("FINAL MODEL LEADERBOARD")
print("=" * 55)
print(f"\n  {'Model':<30} {'Accuracy':>10} {'F1 Macro':>10}")
print("  " + "-" * 54)
for name, r in sorted(all_results.items(), key=lambda x: -x[1]['accuracy']):
    marker = ' <-- BEST' if r['accuracy'] == max(v['accuracy'] for v in all_results.values()) else ''
    print(f"  {name:<30} {r['accuracy']*100:>9.2f}%  {r['f1_macro']*100:>8.2f}%{marker}")

print("\n" + "=" * 50)
print("STEP 4e COMPLETE")
print("=" * 50)