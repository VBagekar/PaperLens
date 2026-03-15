# PaperLens — AI-Powered News Article Classifier

Multi-class NLP text classification on the 20 Newsgroups dataset.

## Project Structure
```
PaperLens/
├── src/          # Pipeline scripts (steps 1-6)
├── gui/          # Desktop GUI application
├── models/       # Saved trained models
├── dataset/      # Preprocessed data cache
├── outputs/      # Charts, CSVs, results
├── report/       # Project report
└── requirements.txt
```

## How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python run_all.py

# Launch GUI
python gui/app.py
```

## Results
| Model | Accuracy |
|---|---|
| Logistic Regression | ~82% |
| Linear SVM | ~81% |
| Naive Bayes | ~77% |
| Random Forest | ~72% |
| k-NN | ~54% |
```

**File 2** — Create `requirements.txt` in the root folder and paste this:
```
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
nltk>=3.8.0
```

Save both files. Then come back to the terminal and run:
```
dir