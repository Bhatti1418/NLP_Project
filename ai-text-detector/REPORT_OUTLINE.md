# Course presentation outline

Title: Detecting AI-Generated Text: TF-IDF Classifiers vs Fine-Tuned DistilBERT, with Cross-Domain Robustness Analysis.

About nine slides. Each block is one slide: what to say, then the figure to leave on screen. Fill the spoken numbers from `outputs/metrics/` after the scripts have been run. The README tables are the same placeholders.

## 1. Problem and motivation

- AI writing is now cheap, and courses, publishers, and employers want a way to tell it from a person's draft.
- A detector that is only right on one dataset can still accuse a student, so the question is both accuracy and what happens off that dataset.
- This project compares a CPU TF-IDF linear classifier with DistilBERT fine-tuned from `distilbert-base-uncased`, on the same labels: 0 = human, 1 = AI.
- The decision rule is an AI probability of 0.5 or higher. The talk will separate in-domain score, cross-domain drop, and the cost of a false accusation.

Figure: none. Title slide only. The disclaimer in `app.py` is the line to preview: "AI-text detectors are imperfect and should not be used as sole evidence of misconduct."

## 2. Dataset and preprocessing

- In-domain data is the Kaggle AI vs Human Text collection (`text`, `generated`), placed locally at `data/raw/AI_Human.csv`.
- Cleaning drops empty rows, duplicate texts, and passages under 20 words, then keeps a balanced subsample of at most 20,000 rows, seed 42.
- That subsample is split 70% / 15% / 15% into train, validation, and test. Validation chooses the model. The test split is scored once.
- Say the class counts and the length gap you actually measured (median words, human vs AI). Mention the length-only baseline so the audience knows a trivial cue was checked.

Figure: `outputs/figures/class_balance.png`, then `outputs/figures/length_distribution.png`. If you have time for a third plot, `outputs/figures/length_only_baseline.png`.

## 3. Methods (TF-IDF + classifier vs DistilBERT)

- TF-IDF uses word unigrams and bigrams, at most 50,000 features, sublinear tf, and `min_df=2`. Logistic regression and a calibrated linear SVM share that vectorizer. `C` is chosen from `{0.1, 1, 10}` by validation F1.
- DistilBERT starts from `distilbert-base-uncased`. Fine-tuning uses length 256, batch size 16, 2 epochs, and learning rate `2e-5`. The epoch with the best validation F1 is kept.
- Both emit probabilities. The positive class for precision, recall, and F1 is AI. The same threshold, 0.5, is used everywhere after this slide.
- Name the practical difference: TF-IDF sees the whole essay and trains on CPU; DistilBERT sees the first 256 tokens and is trained on a GPU.

Figure: `outputs/figures/loss_curve.png` for the DistilBERT run. The TF-IDF side has no loss curve; point at the pipeline on the slide while this plot is up.

## 4. Results (in-domain)

- Read accuracy, precision, recall, F1, and ROC-AUC for both models from `outputs/metrics/model_comparison.csv`. Say which model leads on F1 and by how much.
- Give McNemar's statistic and p-value from `outputs/metrics/comparison_summary.json`: whether the models are wrong on different test texts more often than chance.
- Mention speed and disk size (seconds per 1,000 texts, megabytes). A small F1 gain can be expensive.
- Use the confusion matrices to say whether the remaining errors are mostly false accusations of human writers or missed AI text. Paste your counts into the README table before the talk.

Figure: `outputs/figures/metric_comparison.png` as the main plot. Switch to `outputs/figures/roc_curves.png` and `outputs/figures/calibration_curves.png` if you discuss ranking or overconfidence. Keep `outputs/figures/cm_baseline.png` and `outputs/figures/cm_transformer.png` for the error mix.

## 5. Cross-domain robustness

- The transfer set is Hello-SimpleAI/HC3: human answers versus ChatGPT, cleaned the same way, about 2,000 balanced rows. Say the config you actually loaded (`all`, `open_qa`, or `finance`). If the fallback abstracts were used, say that explicitly.
- Show in-domain versus cross-domain F1, and the drop (in-domain minus cross-domain), for both models. A positive drop means the model got worse off the Kaggle essays.
- Interpret the winner on cross-domain F1, not on the in-domain number. A model can win at home and lose on HC3.
- One sentence on why a drop is expected: the genre is question-answer text, and the generator is ChatGPT, which the Kaggle mix does not cover by itself.

Figure: `outputs/figures/cross_domain.png`.

## 6. Error analysis (FP/FN)

- Define the two mistakes in plain language. A false positive is a human text called AI. A false negative is an AI text called human. Quote your false-positive rate and false-negative rate for each model.
- Say how many test texts both models miss, and how many only one model misses (`outputs/metrics/error_overlap.csv`). Shared mistakes are the harder cases.
- Describe the style comparison only where the median gap is real in your run: length, type-token ratio, words per sentence, punctuation rate, formal-word rate. Skip a feature if the report says the medians are similar.
- Close with length. Say whether accuracy is weakest on short, medium, or long texts, and read the word ranges printed in `outputs/error_analysis_report.md`.

Figure: `outputs/figures/error_boxplots.png`, then `outputs/figures/accuracy_by_length.png`. If you want one example on the slide, take it from `outputs/metrics/errors_distilbert_fp.csv` or `errors_tfidf_fp.csv` (highest-confidence human text called AI).

## 7. Limitations and ethics

- In-domain accuracy is a score on one Kaggle collection. It can reward topic and prompt cues that will not exist in another course or another generator.
- New models (and paraphrases of old ones) move the writing style. HC3 is one ChatGPT check, not a standing test of every later LLM.
- A false positive accuses a person. Formal classroom English and a smaller vocabulary, common for non-native speakers, can look like the machine features in this project, and the data has no native-speaker label.
- The demo line is the policy: do not use this detector as the only evidence of misconduct.

Figure: put `outputs/figures/cross_domain.png` back up, because the drop is the empirical limit. The spoken ethics point does not need a second plot.

## 8. Live demo

- Open `app.py` (`python app.py`). Show the disclaimer before any prediction.
- Paste a human example and an AI-style example from the built-in list. Switch the dropdown from DistilBERT to TF-IDF to Both.
- On TF-IDF, point at the highlighted n-grams: blue pushes toward human, red toward AI. Those are contributions, not proof.
- Delete the text down to a sentence under 50 words and show the warning that short predictions are unreliable.

Figure: none from `outputs/figures/`. The Gradio window is the slide. If the network fails, fall back to a screenshot of the Detect tab with the disclaimer visible.

## 9. Conclusion and future work

- Answer the four research questions in one sentence each, using your F1, your McNemar result, your cross-domain drop, and your false-positive rate.
- State which model you would keep for this course setting, and the cost (speed, size, or brittleness) that comes with it.
- Future work that this repo does not do: a paraphrase attack, more than one outside generator, a native-speaker label, and a longer context than 256 tokens.
- End on the same constraint you opened with: a probability from this model is a clue about a text, not a finding about a person.

Figure: `outputs/figures/metric_comparison.png` beside the one-line cross-domain drop from `outputs/figures/cross_domain.png`. Leave the cross-domain plot up for the last sentence.
