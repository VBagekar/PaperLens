"""
Step 7 - GUI Application
Desktop app for news article classification.
Built with tkinter (Python standard library).
"""

import os
import sys
import pickle
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import numpy as np

# ── Path setup ────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, 'models')
DATA_DIR  = os.path.join(ROOT, 'dataset')

# ── Category names ────────────────────────────────────────────
CATEGORIES = [
    'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
    'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
    'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med',
    'sci.space', 'soc.religion.christian', 'talk.politics.guns',
    'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'
]

# ── Preprocessing ─────────────────────────────────────────────
import re
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    for r in ['punkt', 'stopwords', 'wordnet', 'punkt_tab']:
        try: nltk.download(r, quiet=True)
        except: pass
    STOP_WORDS = set(stopwords.words('english'))
    LEMMATIZER = WordNetLemmatizer()
    USE_NLTK   = True
except:
    USE_NLTK   = False
    STOP_WORDS = set()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if USE_NLTK:
        tokens = word_tokenize(text)
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens
                  if t not in STOP_WORDS and len(t) > 2]
    else:
        tokens = [t for t in text.split() if len(t) > 2]
    return ' '.join(tokens)


# ── Main Application ──────────────────────────────────────────
class PaperLensApp:
    def __init__(self, root):
        self.root       = root
        self.model      = None
        self.vectorizer = None

        root.title("PaperLens — News Article Classifier")
        root.geometry("860x680")
        root.minsize(700, 500)
        root.configure(bg='#F0F0F0')

        self._build_ui()
        self._load_model()

    # ── Load model ────────────────────────────────────────────
    def _load_model(self):
        try:
            with open(os.path.join(MODEL_DIR, 'best_model.pkl'), 'rb') as f:
                self.model = pickle.load(f)
            with open(os.path.join(DATA_DIR, 'features.pkl'), 'rb') as f:
                features = pickle.load(f)
            self.vectorizer = features['tfidf']['vectorizer']
            self._status("Model loaded successfully. Ready to predict.", 'ok')
        except FileNotFoundError:
            self._status("Model not found. Please run the pipeline first (python run_all.py)", 'error')

    # ── Build UI ──────────────────────────────────────────────
    def _build_ui(self):
        # ── Header
        header = tk.Frame(self.root, bg='#1A237E', height=60)
        header.pack(fill='x')
        header.pack_propagate(False)
        tk.Label(header, text="PaperLens",
                 font=('Times New Roman', 22, 'bold'),
                 bg='#1A237E', fg='white').pack(side='left', padx=20, pady=10)
        tk.Label(header, text="News Article Classifier",
                 font=('Times New Roman', 12),
                 bg='#1A237E', fg='#90CAF9').pack(side='left', pady=15)
        tk.Label(header, text="Best Model: LinearSVC (C=0.5) | Accuracy: 86.07%",
                 font=('Times New Roman', 10),
                 bg='#1A237E', fg='#A5D6A7').pack(side='right', padx=20)

        # ── Status bar
        self.status_var = tk.StringVar(value="Loading...")
        self._status_bar = tk.Label(self.root, textvariable=self.status_var,
                                    font=('Times New Roman', 9),
                                    bg='#E0E0E0', fg='#333333',
                                    anchor='w', padx=10, pady=4)
        self._status_bar.pack(fill='x', side='bottom')

        # ── Notebook
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook.Tab', font=('Times New Roman', 10), padding=[12, 6])

        nb = ttk.Notebook(self.root)
        nb.pack(fill='both', expand=True, padx=10, pady=10)

        # Tab 1 — Single
        tab1 = tk.Frame(nb, bg='#F0F0F0')
        nb.add(tab1, text='  Single Predict  ')
        self._build_single_tab(tab1)

        # Tab 2 — Batch
        tab2 = tk.Frame(nb, bg='#F0F0F0')
        nb.add(tab2, text='  Batch Predict  ')
        self._build_batch_tab(tab2)

    def _build_single_tab(self, parent):
        # Input label
        tk.Label(parent, text="Enter news article text:",
                 font=('Times New Roman', 11, 'bold'),
                 bg='#F0F0F0').pack(anchor='w', padx=15, pady=(12, 2))

        # Input box
        self.input_box = scrolledtext.ScrolledText(
            parent, height=10, font=('Times New Roman', 10),
            wrap='word', relief='solid', borderwidth=1)
        self.input_box.pack(fill='x', padx=15, pady=(0, 8))

        # Buttons
        btn_frame = tk.Frame(parent, bg='#F0F0F0')
        btn_frame.pack(fill='x', padx=15, pady=4)

        tk.Button(btn_frame, text="Predict",
                  font=('Times New Roman', 11, 'bold'),
                  bg='#1A237E', fg='white', relief='flat',
                  padx=16, pady=6, cursor='hand2',
                  command=self._predict).pack(side='left', padx=(0, 8))

        tk.Button(btn_frame, text="Clear",
                  font=('Times New Roman', 10),
                  bg='#E0E0E0', fg='black', relief='flat',
                  padx=10, pady=6, cursor='hand2',
                  command=self._clear).pack(side='left', padx=4)

        tk.Button(btn_frame, text="Load File",
                  font=('Times New Roman', 10),
                  bg='#E0E0E0', fg='black', relief='flat',
                  padx=10, pady=6, cursor='hand2',
                  command=self._load_file).pack(side='left', padx=4)

        # Result frame
        result_frame = tk.LabelFrame(parent, text=" Prediction Result ",
                                     font=('Times New Roman', 10, 'bold'),
                                     bg='#F0F0F0', relief='solid', borderwidth=1)
        result_frame.pack(fill='both', expand=True, padx=15, pady=10)

        # Category
        self.result_var = tk.StringVar(value="—")
        tk.Label(result_frame, textvariable=self.result_var,
                 font=('Times New Roman', 20, 'bold'),
                 bg='#F0F0F0', fg='#1A237E').pack(pady=(16, 4))

        # Confidence label
        self.conf_var = tk.StringVar(value="Confidence: —")
        tk.Label(result_frame, textvariable=self.conf_var,
                 font=('Times New Roman', 10),
                 bg='#F0F0F0', fg='#555555').pack()

        # Confidence bar
        self.conf_canvas = tk.Canvas(result_frame, height=18,
                                     bg='#E0E0E0', highlightthickness=0)
        self.conf_canvas.pack(fill='x', padx=40, pady=(4, 8))

        # Top 3
        self.top3_var = tk.StringVar(value="")
        tk.Label(result_frame, textvariable=self.top3_var,
                 font=('Times New Roman', 9),
                 bg='#F0F0F0', fg='#777777').pack(pady=(0, 12))

    def _build_batch_tab(self, parent):
        tk.Label(parent, text="Enter one document per line:",
                 font=('Times New Roman', 11, 'bold'),
                 bg='#F0F0F0').pack(anchor='w', padx=15, pady=(12, 2))

        self.batch_input = scrolledtext.ScrolledText(
            parent, height=8, font=('Times New Roman', 9),
            wrap='word', relief='solid', borderwidth=1)
        self.batch_input.pack(fill='x', padx=15, pady=(0, 8))

        tk.Button(parent, text="Predict All",
                  font=('Times New Roman', 11, 'bold'),
                  bg='#1A237E', fg='white', relief='flat',
                  padx=16, pady=6, cursor='hand2',
                  command=self._batch_predict).pack(anchor='w', padx=15, pady=4)

        # Results table
        cols = ('Line Preview', 'Predicted Category', 'Confidence')
        self.tree = ttk.Treeview(parent, columns=cols, show='headings', height=12)
        self.tree.heading('Line Preview',       text='Line Preview')
        self.tree.heading('Predicted Category', text='Predicted Category')
        self.tree.heading('Confidence',         text='Confidence')
        self.tree.column('Line Preview',       width=300)
        self.tree.column('Predicted Category', width=250)
        self.tree.column('Confidence',         width=100)

        sb = ttk.Scrollbar(parent, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(fill='both', expand=True, padx=15, pady=8)
        sb.pack(side='right', fill='y')

    # ── Predict ───────────────────────────────────────────────
    def _predict(self):
        text = self.input_box.get('1.0', 'end').strip()
        if not text:
            messagebox.showwarning("Empty", "Please enter some text.")
            return
        if not self.model or not self.vectorizer:
            messagebox.showerror("Error", "Model not loaded.")
            return

        clean   = preprocess(text)
        vec     = self.vectorizer.transform([clean])
        pred    = self.model.predict(vec)[0]
        category = CATEGORIES[pred]

        self.result_var.set(category)

        # Confidence via decision function
        conf = None
        top3_text = ""
        if hasattr(self.model, 'decision_function'):
            scores = self.model.decision_function(vec)[0]
            exp_s  = np.exp(scores - scores.max())
            probs  = exp_s / exp_s.sum()
            conf   = probs[pred]
            top3   = probs.argsort()[-3:][::-1]
            top3_text = "Top 3:  " + "   |   ".join(
                f"{CATEGORIES[i]}: {probs[i]*100:.1f}%" for i in top3)
        elif hasattr(self.model, 'predict_proba'):
            probs  = self.model.predict_proba(vec)[0]
            conf   = probs[pred]
            top3   = probs.argsort()[-3:][::-1]
            top3_text = "Top 3:  " + "   |   ".join(
                f"{CATEGORIES[i]}: {probs[i]*100:.1f}%" for i in top3)

        if conf is not None:
            self.conf_var.set(f"Confidence: {conf*100:.1f}%")
            self._draw_bar(conf)

        self.top3_var.set(top3_text)
        self._status(f"Predicted: {category}", 'ok')

    def _draw_bar(self, conf):
        self.conf_canvas.update_idletasks()
        w = self.conf_canvas.winfo_width()
        fill_w = int(w * conf)
        color  = '#4CAF50' if conf > 0.6 else ('#FF9800' if conf > 0.35 else '#F44336')
        self.conf_canvas.delete('all')
        self.conf_canvas.create_rectangle(0, 0, w, 18, fill='#E0E0E0', outline='')
        self.conf_canvas.create_rectangle(0, 0, fill_w, 18, fill=color, outline='')

    def _batch_predict(self):
        lines = [l.strip() for l in
                 self.batch_input.get('1.0', 'end').strip().split('\n')
                 if l.strip()]
        if not lines:
            messagebox.showwarning("Empty", "Enter at least one line.")
            return
        if not self.model or not self.vectorizer:
            messagebox.showerror("Error", "Model not loaded.")
            return

        for row in self.tree.get_children():
            self.tree.delete(row)

        cleaned = [preprocess(l) for l in lines]
        vecs    = self.vectorizer.transform(cleaned)
        preds   = self.model.predict(vecs)

        for line, pred in zip(lines, preds):
            preview  = line[:60] + '...' if len(line) > 60 else line
            category = CATEGORIES[pred]
            conf     = '—'
            if hasattr(self.model, 'decision_function'):
                scores = self.model.decision_function(
                    self.vectorizer.transform([preprocess(line)]))[0]
                exp_s  = np.exp(scores - scores.max())
                probs  = exp_s / exp_s.sum()
                conf   = f"{probs[pred]*100:.1f}%"
            self.tree.insert('', 'end', values=(preview, category, conf))

        self._status(f"Batch predicted {len(lines)} documents.", 'ok')

    def _clear(self):
        self.input_box.delete('1.0', 'end')
        self.result_var.set('—')
        self.conf_var.set('Confidence: —')
        self.conf_canvas.delete('all')
        self.top3_var.set('')

    def _load_file(self):
        path = filedialog.askopenfilename(
            filetypes=[('Text Files', '*.txt'), ('All Files', '*.*')])
        if path:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            self.input_box.delete('1.0', 'end')
            self.input_box.insert('1.0', content)

    def _status(self, msg, kind='ok'):
        self.status_var.set(msg)
        colors = {'ok': ('#E8F5E9', '#2E7D32'),
                  'error': ('#FFEBEE', '#C62828')}
        bg, fg = colors.get(kind, ('#F5F5F5', '#333333'))
        self._status_bar.config(bg=bg, fg=fg)


if __name__ == '__main__':
    root = tk.Tk()
    app  = PaperLensApp(root)
    root.mainloop()