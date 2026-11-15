# Detecting AI-Generated Text: TF-IDF Classifiers vs Fine-Tuned DistilBERT

University NLP project. Two detectors are trained on the same human-versus-AI essays, then compared on a held-out slice of that collection and on text from another source. The baseline is a TF-IDF linear classifier meant to run on CPU. DistilBERT is fine-tuned from the pretrained checkpoint `distilbert-base-uncased` and is meant for a GPU runtime (Google Colab or Kaggle). No model is trained from scratch. The random seed is 42 everywhere.

Label 0 is human and label 1 is AI. A predicted AI probability of 0.5 or higher is read as AI-generated. Precision, recall, and F1 use AI as the positive class.

## Research questions

1. On a held-out slice of one AI-versus-human essay collection, how well does a bigram TF-IDF linear classifier separate the two classes?
2. Does fine-tuning DistilBERT improve accuracy, F1, ranking (ROC-AUC), and calibration on that same test split, and are the paired errors different by McNemar's test?
3. How much does each model's F1 drop when the texts come from another domain, here Hello-SimpleAI/HC3 (human answers versus ChatGPT)?
4. Which mistakes remain, false accusations of human writing and missed AI text, and how do they vary with length and surface style?

## Datasets

### In-domain: AI vs Human Text

Source: the Kaggle dataset [shanegerami/ai-vs-human-text](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text). Place the file yourself at `data/raw/AI_Human.csv`. The loader expects columns `text` and `generated` (`0` = human, `1` = AI) and will rename close alternatives (`essay`, `content`, `label`, `class`, and similar) when those are what the file actually contains.

Cleaning, in `src/data_utils.py`:

- Drop rows with missing text or labels, strip whitespace, then drop duplicate texts.
- Drop texts shorter than 20 words.
- Draw a balanced subsample of at most 20,000 rows (seed 42).
- Split that subsample, stratified, into train / validation / test at 70% / 15% / 15%. The training split is therefore about 14,000 rows.

The test split is never used to choose a model.

### Cross-domain: HC3

`src/cross_domain.py` downloads [Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3). It tries the configs `all`, then `open_qa`, then `finance`. Each human answer becomes label 0 and each ChatGPT answer becomes label 1. The same cleaner runs, then a balanced sample of about 2,000 rows is drawn.

If every HC3 config fails, the script falls back to [Ateeqq/AI-and-Human-Generated-Text](https://huggingface.co/datasets/Ateeqq/AI-and-Human-Generated-Text): original scientific abstracts are label 0 and GPT-3 abstracts are label 1. The saved JSON records which source was used. That fallback is still outside the Kaggle essay mix. It is a different generator and a different genre.

## Methodology

### TF-IDF + linear classifier

`src/train_baseline.py` builds one scikit-learn pipeline:

- `TfidfVectorizer`: word unigrams and bigrams, at most 50,000 features, `sublinear_tf=True`, `min_df=2`.
- Two classifiers, each with `C` in `{0.1, 1.0, 10.0}`: `LogisticRegression` (`liblinear`) and `LinearSVC` wrapped in `CalibratedClassifierCV` (sigmoid, 5 stratified folds, seed 42) so both can emit probabilities.
- The candidate with the highest validation F1 is saved as `models/tfidf_best.joblib`. Ties keep the earlier candidate. The test set is scored only after that choice.
- For a linear model, a positive coefficient points toward AI. For the calibrated SVM, the reported feature weight is the mean coefficient across calibration folds.

### Fine-tuned DistilBERT

`src/train_transformer.py` starts from `distilbert-base-uncased` and fine-tunes a 2-label head (`0` = human, `1` = ai) with the Hugging Face `Trainer`:

- Maximum length 256 tokens, batch size 16, 2 epochs, learning rate `2e-5`.
- The best epoch by validation F1 is the checkpoint written to `models/distilbert_detector`.
- `--max_samples 500` caps each split for a CPU smoke test. The full run is intended for a GPU. `notebooks/02_train_distilbert_colab.ipynb` is the Colab/Kaggle entry point.

### Comparison, transfer, and errors

- `src/evaluate.py` scores both models on the same `data/processed/test.csv`. It writes accuracy, precision, recall, F1, ROC-AUC, seconds per 1,000 texts, and size on disk, plus McNemar's test with Edwards' continuity correction.
- `src/cross_domain.py` repeats those metrics out of domain and writes the drop (in-domain minus cross-domain).
- `src/error_analysis.py` lists false positives (human called AI) and false negatives (AI called human), compares style features of correct and incorrect texts, and breaks accuracy into short, medium, and long tertiles.
- `app.py` is a Gradio demo. It loads each saved model once and caches it.

## Install

Python 3.10 or newer.

```bash
python -m venv .venv
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Run every command below from the `ai-text-detector` directory.

## Run each step in order

1. Put `AI_Human.csv` in `data/raw/`.
2. Build the splits:

   ```bash
   python -m src.data_utils
   ```

3. Open `notebooks/01_eda.ipynb` and run all cells. This writes the EDA figures and refreshes `data/processed/train.csv`, `val.csv`, and `test.csv` with the same seed. Do this before training.
4. Train the TF-IDF baseline (CPU, several cross-validated fits):

   ```bash
   python -m src.train_baseline
   ```

5. Fine-tune DistilBERT. On a GPU machine, or in Colab via `notebooks/02_train_distilbert_colab.ipynb`:

   ```bash
   python -m src.train_transformer
   ```

   CPU smoke test only:

   ```bash
   python -m src.train_transformer --max_samples 500
   ```

6. Compare both models on the held-out test split:

   ```bash
   python -m src.evaluate
   ```

7. Score a cross-domain sample (needs the saved models and network access the first time):

   ```bash
   python -m src.cross_domain
   ```

8. Write the error lists and the narrative report:

   ```bash
   python -m src.error_analysis
   ```

9. Start the demo:

   ```bash
   python app.py
   ```

   One passage from the command line:

   ```bash
   python -m src.predict --text "Your passage here." --model tfidf
   ```

## Results

Fill these cells from `outputs/metrics/model_comparison.csv`, `outputs/metrics/comparison_summary.json`, `outputs/metrics/cross_domain_metrics.json`, and `outputs/error_analysis_report.md` after you run the scripts. The cells are placeholders.

### In-domain test set

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Seconds / 1,000 | Size (MB) | Device |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TF-IDF | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DistilBERT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### McNemar's test (paired errors on the same test texts)

| Statistic | p-value | TF-IDF only correct | DistilBERT only correct | Both correct | Both wrong | Significant at 5% |
| --- | --- | --- | --- | --- | --- | --- |
| TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### Cross-domain drop

Drop = in-domain score minus cross-domain score. A positive drop means the model did worse outside the training collection. Record the dataset id from `cross_domain_metrics.json` (HC3 config, or the abstract fallback).

Dataset used: TBD

| Model | Split | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| TF-IDF | In-domain | TBD | TBD | TBD | TBD | TBD |
| TF-IDF | Cross-domain | TBD | TBD | TBD | TBD | TBD |
| TF-IDF | Drop | TBD | TBD | TBD | TBD | TBD |
| DistilBERT | In-domain | TBD | TBD | TBD | TBD | TBD |
| DistilBERT | Cross-domain | TBD | TBD | TBD | TBD | TBD |
| DistilBERT | Drop | TBD | TBD | TBD | TBD | TBD |

### Error rates on the in-domain test set

A false positive is a human text called AI. A false negative is an AI text called human.

| Model | False positives | False-positive rate | False negatives | False-negative rate |
| --- | --- | --- | --- | --- |
| TF-IDF | TBD | TBD | TBD | TBD |
| DistilBERT | TBD | TBD | TBD | TBD |

| Overlap | Count | Share |
| --- | --- | --- |
| Both correct | TBD | TBD |
| Both wrong | TBD | TBD |
| TF-IDF only wrong | TBD | TBD |
| DistilBERT only wrong | TBD | TBD |

## Folder structure

```
ai-text-detector/
├── app.py                          Gradio demo
├── REPORT_OUTLINE.md               slide outline for the course presentation
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                        AI_Human.csv and cached HC3 JSONL (gitignored)
│   └── processed/                  train.csv, val.csv, test.csv
├── models/                         tfidf_best.joblib, distilbert_detector/ (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_train_distilbert_colab.ipynb
├── outputs/
│   ├── error_analysis_report.md
│   ├── figures/                    plots listed below
│   └── metrics/                    CSV and JSON tables
└── src/
    ├── config.py                   seed, paths, hyperparameters
    ├── data_utils.py
    ├── train_baseline.py
    ├── train_transformer.py
    ├── evaluate.py
    ├── cross_domain.py
    ├── error_analysis.py
    └── predict.py
```

Figures written under `outputs/figures/`:

| File | What it shows |
| --- | --- |
| `class_balance.png` | Class counts after cleaning, before the 20,000-row subsample |
| `length_distribution.png` | Word-count distributions by class |
| `top_words_per_class.png` | Top 20 content words per class |
| `length_only_baseline.png` | A length-threshold classifier chosen on train and scored on test |
| `cm_baseline.png` | TF-IDF confusion matrix on the test set |
| `cm_transformer.png` | DistilBERT confusion matrix on the test set |
| `loss_curve.png` | DistilBERT training and validation loss |
| `metric_comparison.png` | Side-by-side accuracy, precision, recall, F1, ROC-AUC, speed, and size |
| `roc_curves.png` | Both ROC curves |
| `calibration_curves.png` | Both calibration curves |
| `cross_domain.png` | In-domain versus cross-domain scores |
| `error_boxplots.png` | Style features for correct versus incorrect texts |
| `accuracy_by_length.png` | Accuracy on short, medium, and long texts |

## Limitations

- The in-domain test set is a held-out slice of the same Kaggle collection used for training. A high score can mean the models learned that collection's prompts, topics, or length mix.
- HC3 changes both the genre (question-answer text) and the generator (ChatGPT). A drop there is real, and it is not a test of every later model. Claude, Gemini, and newer open models are unseen.
- DistilBERT reads the first 256 tokens. TF-IDF uses the full text. The two systems do not always see the same span.
- The TF-IDF `C` grid is small, and DistilBERT is fine-tuned for two epochs. These are course-scale settings, not a full hyperparameter search.
- Paraphrase and other evasion attacks are not in the test set. Accuracy here is not a measure of how hard the detector is to evade.
- The error-analysis style features (type-token ratio, punctuation, a fixed formal-word list) are descriptive. They are not a causal account of why a text was mislabeled.
- The cross-domain sample is about 2,000 balanced rows. If HC3 fails to download, the fallback abstracts answer a narrower version of the transfer question, and the report must say so.

## Ethical considerations

A false positive is the system telling a human author that their writing is machine-generated. That accusation is especially risky for non-native English speakers: classroom English, formal connectors, and a smaller vocabulary are ordinary features of second-language writing, and they can resemble the features these models treat as AI. This dataset has no native-speaker label, so the project cannot measure that gap, and it also cannot dismiss it.

The demo states this on screen: **AI-text detectors are imperfect and should not be used as sole evidence of misconduct.** A grade, a hiring decision, or an academic-integrity charge needs evidence beyond a probability from this model. Detectors also drift as new generators appear, and they can be steered off by a paraphrase. Publishing the model weights and the demo is for coursework and error analysis, not for deploying a misconduct detector.
