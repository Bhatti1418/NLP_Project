"""Error analysis for the TF-IDF and DistilBERT test predictions.

False positives are human texts called AI. False negatives are AI texts called
human. The style comparisons describe this test split.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, METRICS_DIR, MODELS_DIR, OUTPUTS_DIR, PROJECT_ROOT, SEED
from src.data_utils import set_seed
from src.evaluate import (
    DISTILBERT_DIRNAME,
    MODEL_COLORS,
    load_test_set,
    load_tfidf,
    score_distilbert,
    score_tfidf,
    warn_if_partial_finetune,
)

logger = logging.getLogger(__name__)

TOP_K = 20
EXAMPLE_COUNT = 3
EXCERPT_WORDS = 70
WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
FORMAL_WORDS: frozenset[str] = frozenset(
    {
        "additionally",
        "aforementioned",
        "algorithm",
        "analysis",
        "consequently",
        "correlation",
        "criteria",
        "criterion",
        "dataset",
        "demonstrate",
        "demonstrated",
        "demonstrates",
        "distribution",
        "empirical",
        "framework",
        "furthermore",
        "hence",
        "herein",
        "hypothesis",
        "implementation",
        "indicate",
        "indicates",
        "infrastructure",
        "methodology",
        "methodological",
        "moreover",
        "nevertheless",
        "nonetheless",
        "notwithstanding",
        "optimization",
        "parameter",
        "parameters",
        "probability",
        "qualitative",
        "quantitative",
        "respectively",
        "significant",
        "significantly",
        "statistical",
        "subsequently",
        "theoretical",
        "thereby",
        "therefore",
        "thus",
        "utilize",
        "utilized",
        "utilizes",
        "utilizing",
        "utilization",
        "variable",
        "variables",
    }
)
FEATURE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("n_words", "Words"),
    ("type_token_ratio", "Type-token ratio"),
    ("avg_sentence_length", "Words per sentence"),
    ("punctuation_rate", "Punctuation rate"),
    ("formal_rate", "Formal-word rate"),
)
MODEL_SPECS: tuple[tuple[str, str, str], ...] = (
    ("TF-IDF", "tfidf", "tfidf"),
    ("DistilBERT", "distilbert", "distilbert"),
)


def attach_style_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add length, richness, sentence, punctuation, and formal-language features."""
    styled = frame.copy()
    features = styled["text"].map(text_style_features).apply(pd.Series)
    return pd.concat([styled.reset_index(drop=True), features], axis=1)


def text_style_features(text: str) -> dict[str, float]:
    """Style measures for one text. Rates are zero when the text has no words."""
    raw = "" if text is None else str(text)
    tokens = WORD_PATTERN.findall(raw)
    lowered = [token.lower() for token in tokens]
    n_words = len(lowered)
    unique = len(set(lowered))
    sentence_lengths = [
        len(WORD_PATTERN.findall(part))
        for part in re.split(r"[.!?]+", raw)
        if WORD_PATTERN.search(part)
    ]
    punctuation = sum(1 for char in raw if unicodedata.category(char).startswith("P"))
    formal_hits = sum(1 for token in lowered if token in FORMAL_WORDS)
    return {
        "n_words": float(n_words),
        "type_token_ratio": float(unique / n_words) if n_words else 0.0,
        "avg_sentence_length": float(np.mean(sentence_lengths)) if sentence_lengths else 0.0,
        "punctuation_rate": float(punctuation / max(len(raw), 1)),
        "formal_rate": float(formal_hits / n_words) if n_words else 0.0,
        "formal_present": float(formal_hits > 0),
    }


def top_confident_errors(frame: pd.DataFrame, model_prefix: str, kind: str, k: int = TOP_K) -> pd.DataFrame:
    """Return up to ``k`` highest-confidence mistakes of one kind.

    ``kind`` is ``fp`` (human predicted as AI) or ``fn`` (AI predicted as human).
    Confidence is the probability of the predicted, wrong class.
    """
    if kind not in {"fp", "fn"}:
        raise ValueError(f"kind must be 'fp' or 'fn', got {kind!r}.")
    prediction = frame[f"{model_prefix}_prediction"].to_numpy()
    probability_ai = frame[f"{model_prefix}_probability_ai"].to_numpy()
    labels = frame["label"].to_numpy()
    if kind == "fp":
        mask = (labels == 0) & (prediction == 1)
        confidence = probability_ai
    else:
        mask = (labels == 1) & (prediction == 0)
        confidence = 1.0 - probability_ai
    selected = frame.loc[mask].copy()
    selected["confidence"] = confidence[mask]
    selected["error_type"] = kind
    selected = selected.sort_values(["confidence", "row_id"], ascending=[False, True])
    return selected.head(k)


def assign_length_buckets(n_words: pd.Series) -> tuple[pd.Series, str]:
    """Label texts short, medium, or long from test-set tertiles."""
    try:
        buckets = pd.qcut(n_words, 3, labels=["short", "medium", "long"])
        note = "Short, medium, and long are tertiles of word count on this test set."
    except ValueError:
        buckets = pd.cut(
            n_words,
            bins=[-np.inf, 100, 250, np.inf],
            labels=["short", "medium", "long"],
        )
        note = (
            "Tertiles were not unique, so these buckets use fixed cuts: "
            "short is 100 words or fewer, medium is 101 to 250, and long is above 250."
        )
    return buckets.astype(str), note


def accuracy_by_length(frame: pd.DataFrame) -> pd.DataFrame:
    """Accuracy of each model inside each length bucket."""
    rows: list[dict[str, Any]] = []
    for display_name, prefix, _slug in MODEL_SPECS:
        correct = frame["label"].to_numpy() == frame[f"{prefix}_prediction"].to_numpy()
        grouped = (
            pd.DataFrame({"bucket": frame["bucket"], "correct": correct, "n_words": frame["n_words"]})
            .groupby("bucket", observed=True)
            .agg(accuracy=("correct", "mean"), n=("correct", "size"), median_words=("n_words", "median"))
        )
        for bucket in ("short", "medium", "long"):
            if bucket not in grouped.index:
                continue
            record = grouped.loc[bucket]
            rows.append(
                {
                    "model": display_name,
                    "bucket": bucket,
                    "accuracy": float(record["accuracy"]),
                    "n": int(record["n"]),
                    "median_words": float(record["median_words"]),
                }
            )
    return pd.DataFrame(rows)


def overlap_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Count texts both models miss, and texts only one model misses."""
    categories = overlap_categories(frame)
    n = len(frame)
    rows = [
        {
            "category": name,
            "count": int((categories == name).sum()),
            "share": float((categories == name).mean()) if n else 0.0,
        }
        for name in ("both_correct", "both_wrong", "tfidf_only", "distilbert_only")
    ]
    return pd.DataFrame(rows)


def overlap_categories(frame: pd.DataFrame) -> pd.Series:
    """Label each row both_correct, both_wrong, tfidf_only, or distilbert_only."""
    tfidf_wrong = frame["label"].to_numpy() != frame["tfidf_prediction"].to_numpy()
    distilbert_wrong = frame["label"].to_numpy() != frame["distilbert_prediction"].to_numpy()
    categories = np.full(len(frame), "both_correct", dtype=object)
    categories[tfidf_wrong & distilbert_wrong] = "both_wrong"
    categories[tfidf_wrong & ~distilbert_wrong] = "tfidf_only"
    categories[~tfidf_wrong & distilbert_wrong] = "distilbert_only"
    return pd.Series(categories, index=frame.index, name="category")


def feature_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Median style features for correct vs incorrect texts, per model."""
    rows: list[dict[str, Any]] = []
    for display_name, prefix, _slug in MODEL_SPECS:
        correct = frame["label"].to_numpy() == frame[f"{prefix}_prediction"].to_numpy()
        for outcome, mask in (("correct", correct), ("incorrect", ~correct)):
            subset = frame.loc[mask]
            record: dict[str, Any] = {
                "model": display_name,
                "outcome": outcome,
                "n": int(len(subset)),
            }
            for column, _label in FEATURE_COLUMNS:
                record[column] = float(subset[column].median()) if len(subset) else float("nan")
            record["formal_present_share"] = float(subset["formal_present"].mean()) if len(subset) else float("nan")
            rows.append(record)
    return pd.DataFrame(rows)


def plot_feature_boxplots(frame: pd.DataFrame, path: Path) -> None:
    """Boxplots of style features for correct vs incorrect texts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(len(FEATURE_COLUMNS), 2, figsize=(10.5, 14.5), sharex=False)
    for column_index, (display_name, prefix, _slug) in enumerate(MODEL_SPECS):
        correct = frame["label"].to_numpy() == frame[f"{prefix}_prediction"].to_numpy()
        for row_index, (column, label) in enumerate(FEATURE_COLUMNS):
            axis = axes[row_index, column_index]
            panel = pd.DataFrame(
                {
                    "outcome": np.where(correct, "Correct", "Incorrect"),
                    "value": frame[column].to_numpy(),
                }
            )
            sns.boxplot(
                data=panel,
                x="outcome",
                y="value",
                order=["Correct", "Incorrect"],
                color=MODEL_COLORS[display_name],
                ax=axis,
            )
            axis.set_xlabel("")
            axis.set_ylabel(label if column_index == 0 else "")
            if row_index == 0:
                axis.set_title(display_name)
    fig.suptitle("Correct vs incorrect texts", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", _display_path(path))


def plot_accuracy_by_length(bucket_accuracy: pd.DataFrame, path: Path) -> None:
    """Grouped bars of accuracy for short, medium, and long texts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axis = plt.subplots(figsize=(7.4, 4.4))
    sns.barplot(
        data=bucket_accuracy,
        x="bucket",
        y="accuracy",
        hue="model",
        order=["short", "medium", "long"],
        hue_order=["TF-IDF", "DistilBERT"],
        palette=MODEL_COLORS,
        ax=axis,
    )
    for container in axis.containers:
        axis.bar_label(container, fmt="%.2f", padding=2)
    axis.set_ylim(0, 1.15)
    axis.set_xlabel("")
    axis.set_ylabel("Accuracy")
    axis.set_title("Accuracy by text-length bucket")
    axis.legend(title="", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", _display_path(path))


def build_report(
    frame: pd.DataFrame,
    feature_table: pd.DataFrame,
    bucket_accuracy: pd.DataFrame,
    overlap: pd.DataFrame,
    bucket_note: str,
    saved_errors: dict[str, pd.DataFrame],
) -> str:
    """Markdown report: findings, examples, and the limits of this kind of detector."""
    lines: list[str] = [
        "# Error analysis",
        "",
        "Both models are scored on the same held-out test texts. "
        "A probability of 0.5 or higher is read as AI (label 1). "
        "A false positive is a human text called AI. A false negative is an AI text called human. "
        "Confidence is the probability of the predicted class, so the lists below are the mistakes "
        "the model was most sure about.",
        "",
        f"Test size: {len(frame)} texts "
        f"(human={int((frame['label'] == 0).sum())}, AI={int((frame['label'] == 1).sum())}).",
        "",
        "## Error counts",
        "",
        _counts_table(frame),
        "",
        "## Where the models disagree",
        "",
        _overlap_prose(overlap),
        "",
        _markdown_table(
            overlap.assign(share=lambda data: data["share"].map(lambda value: f"{value:.1%}")),
            ["category", "count", "share"],
        ),
        "",
        "Rows where only one model fails, and rows where both fail, are listed in "
        "`metrics/error_overlap_samples.csv`.",
        "",
        *_example_block(
            "Both models wrong",
            _both_wrong_examples(frame, EXAMPLE_COUNT),
            "These are the shared misses with the highest average confidence.",
        ),
        "",
        "## Patterns in the mistakes",
        "",
        bucket_note,
        " Medians below compare texts each model got right with texts it got wrong. "
        "A gap is reported only when it is larger than about 10 percent, and it describes this test set only.",
        "",
        _markdown_table(_format_feature_table(feature_table), list(_format_feature_table(feature_table).columns)),
        "",
        *_finding_lines(frame, feature_table, bucket_accuracy),
        "",
        "### Accuracy by length",
        "",
        _markdown_table(_format_bucket_table(bucket_accuracy), ["model", "bucket", "n", "median_words", "accuracy"]),
        "",
        "Figures: `figures/error_boxplots.png` and `figures/accuracy_by_length.png`.",
        "",
        "## Highest-confidence mistakes",
        "",
    ]
    for display_name, prefix, slug in MODEL_SPECS:
        for kind, title in (("fp", "false positives"), ("fn", "false negatives")):
            subset = saved_errors[f"{slug}_{kind}"]
            lines.extend(
                _example_block(
                    f"{display_name} {title}",
                    subset.head(EXAMPLE_COUNT),
                    f"Showing {min(EXAMPLE_COUNT, len(subset))} of {len(subset)} saved "
                    f"(up to {TOP_K} highest-confidence). Full rows: "
                    f"`metrics/errors_{slug}_{kind}.csv`.",
                )
            )
            lines.append("")
    lines.extend(
        [
            "## Limitations of AI-text detection",
            "",
            _limitations(frame),
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def run_analysis(
    frame: pd.DataFrame,
    metrics_dir: Path,
    figures_dir: Path,
    report_path: Path,
) -> str:
    """Write error lists, plots, summary tables, and the markdown report."""
    analyzed = attach_style_features(frame)
    analyzed["bucket"], bucket_note = assign_length_buckets(analyzed["n_words"])
    ranges = (
        analyzed.groupby("bucket")["n_words"]
        .agg(min_words="min", max_words="max")
        .reindex(["short", "medium", "long"])
        .dropna(how="all")
    )
    range_bits = [
        f"{bucket} {int(row.min_words)}–{int(row.max_words)} words"
        for bucket, row in ranges.iterrows()
        if pd.notna(row.min_words)
    ]
    if range_bits:
        bucket_note = bucket_note + " Observed ranges: " + "; ".join(range_bits) + "."

    metrics_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, pd.DataFrame] = {}
    export_columns = [
        "row_id",
        "text",
        "label",
        "prediction",
        "probability_human",
        "probability_ai",
        "confidence",
        "error_type",
        "n_words",
        "type_token_ratio",
        "avg_sentence_length",
        "punctuation_rate",
        "formal_rate",
    ]
    for _display_name, prefix, slug in MODEL_SPECS:
        for kind in ("fp", "fn"):
            mistakes = top_confident_errors(analyzed, prefix, kind, TOP_K)
            export = _export_frame(mistakes, prefix)
            path = metrics_dir / f"errors_{slug}_{kind}.csv"
            export.reindex(columns=export_columns).to_csv(path, index=False)
            saved[f"{slug}_{kind}"] = export
            logger.info("Wrote %s (%d rows).", _display_path(path), len(export))

    feature_table = feature_comparison(analyzed)
    feature_path = metrics_dir / "error_feature_comparison.csv"
    feature_table.to_csv(feature_path, index=False)
    logger.info("Wrote %s", _display_path(feature_path))

    bucket_accuracy = accuracy_by_length(analyzed)
    bucket_path = metrics_dir / "accuracy_by_length.csv"
    bucket_accuracy.to_csv(bucket_path, index=False)
    logger.info("Wrote %s", _display_path(bucket_path))

    overlap = overlap_summary(analyzed)
    overlap_path = metrics_dir / "error_overlap.csv"
    overlap.to_csv(overlap_path, index=False)
    logger.info("Wrote %s", _display_path(overlap_path))
    sample_path = metrics_dir / "error_overlap_samples.csv"
    _overlap_samples(analyzed).to_csv(sample_path, index=False)
    logger.info("Wrote %s", _display_path(sample_path))

    plot_feature_boxplots(analyzed, figures_dir / "error_boxplots.png")
    plot_accuracy_by_length(bucket_accuracy, figures_dir / "accuracy_by_length.png")

    report = build_report(analyzed, feature_table, bucket_accuracy, overlap, bucket_note, saved)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    logger.info("Wrote %s", _display_path(report_path))
    return report


def main(argv: list[str] | None = None) -> None:
    """Load test predictions for both models and write the error analysis."""
    _configure_logging()
    args = _build_parser().parse_args(argv)
    set_seed(SEED)

    test = load_test_set(args.test_csv)
    frame = _prediction_frame(test, args.predictions_csv, args.tfidf_path, args.distilbert_dir)
    metrics_dir = METRICS_DIR if args.metrics_dir is None else Path(args.metrics_dir)
    figures_dir = FIGURES_DIR if args.figures_dir is None else Path(args.figures_dir)
    report_path = (
        OUTPUTS_DIR / "error_analysis_report.md" if args.report_path is None else Path(args.report_path)
    )
    report = run_analysis(frame, metrics_dir, figures_dir, report_path)
    print()
    print(_console_summary(report))


def _prediction_frame(
    test: pd.DataFrame,
    predictions_csv: Path | None,
    tfidf_path: Path | None,
    distilbert_dir: Path | None,
) -> pd.DataFrame:
    logger.info("Scoring the TF-IDF pipeline on the test set.")
    tfidf = score_tfidf(load_tfidf(tfidf_path), test["text"])
    model_dir = MODELS_DIR / DISTILBERT_DIRNAME if distilbert_dir is None else Path(distilbert_dir)
    distilbert = _distilbert_scores(test, predictions_csv, model_dir)
    frame = pd.DataFrame(
        {
            "row_id": np.arange(len(test)),
            "text": test["text"].astype(str).to_numpy(),
            "label": test["label"].astype(int).to_numpy(),
            "tfidf_prediction": tfidf["predictions"].astype(int),
            "tfidf_probability_ai": tfidf["probabilities"].astype(float),
            "distilbert_prediction": distilbert["predictions"].astype(int),
            "distilbert_probability_ai": distilbert["probabilities"].astype(float),
        }
    )
    if len(frame) != len(test):
        raise RuntimeError("Prediction frame is not aligned with the test split.")
    return frame


def _distilbert_scores(test: pd.DataFrame, predictions_csv: Path | None, model_dir: Path) -> dict[str, Any]:
    explicit = predictions_csv is not None
    saved_path = METRICS_DIR / "transformer_test_predictions.csv" if predictions_csv is None else Path(predictions_csv)
    if saved_path.is_file():
        saved = pd.read_csv(saved_path)
        if _predictions_match_test(test, saved):
            logger.info("Using saved DistilBERT test predictions from %s.", _display_path(saved_path))
            probabilities = np.clip(saved["probability_ai"].to_numpy(dtype=float), 0.0, 1.0)
            predictions = saved["prediction"].to_numpy(dtype=int)
            return {"probabilities": probabilities, "predictions": predictions}
        logger.warning(
            "%s does not line up with the current test split. Rescoring DistilBERT.",
            _display_path(saved_path),
        )
    elif explicit:
        raise FileNotFoundError(f"DistilBERT predictions not found at {_display_path(saved_path)}.")
    else:
        logger.info("No saved DistilBERT test predictions. Scoring the checkpoint.")
    return score_distilbert(model_dir, test["text"])


def _predictions_match_test(test: pd.DataFrame, saved: pd.DataFrame) -> bool:
    required = {"text", "label", "prediction", "probability_ai"}
    if not required <= set(saved.columns) or len(saved) != len(test):
        return False
    same_text = test["text"].astype(str).str.strip().equals(saved["text"].astype(str).str.strip())
    same_label = test["label"].astype(int).equals(saved["label"].astype(int))
    return bool(same_text and same_label)


def _export_frame(mistakes: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if mistakes.empty:
        return pd.DataFrame(
            columns=[
                "row_id",
                "text",
                "label",
                "prediction",
                "probability_human",
                "probability_ai",
                "confidence",
                "error_type",
                "n_words",
                "type_token_ratio",
                "avg_sentence_length",
                "punctuation_rate",
                "formal_rate",
            ]
        )
    export = mistakes.copy()
    export["prediction"] = export[f"{prefix}_prediction"].astype(int)
    export["probability_ai"] = export[f"{prefix}_probability_ai"].astype(float)
    export["probability_human"] = 1.0 - export["probability_ai"]
    export["label"] = export["label"].astype(int)
    return export


def _overlap_samples(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows where at least one model is wrong, with the disagreement category."""
    categories = overlap_categories(frame)
    samples = frame.loc[categories != "both_correct"].copy()
    samples.insert(0, "category", categories.loc[samples.index].to_numpy())
    samples["text"] = samples["text"].map(lambda text: _excerpt(str(text)))
    columns = [
        "row_id",
        "category",
        "label",
        "tfidf_prediction",
        "tfidf_probability_ai",
        "distilbert_prediction",
        "distilbert_probability_ai",
        "n_words",
        "text",
    ]
    return samples.sort_values(["category", "row_id"]).reindex(columns=columns)


def _both_wrong_examples(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    tfidf_wrong = frame["label"].to_numpy() != frame["tfidf_prediction"].to_numpy()
    distilbert_wrong = frame["label"].to_numpy() != frame["distilbert_prediction"].to_numpy()
    selected = frame.loc[tfidf_wrong & distilbert_wrong].copy()
    if selected.empty:
        return selected
    tfidf_confidence = np.where(
        selected["tfidf_prediction"].to_numpy() == 1,
        selected["tfidf_probability_ai"],
        1.0 - selected["tfidf_probability_ai"],
    )
    distilbert_confidence = np.where(
        selected["distilbert_prediction"].to_numpy() == 1,
        selected["distilbert_probability_ai"],
        1.0 - selected["distilbert_probability_ai"],
    )
    selected["confidence"] = (tfidf_confidence + distilbert_confidence) / 2.0
    selected["prediction"] = selected["tfidf_prediction"]
    selected["probability_ai"] = selected["tfidf_probability_ai"]
    return selected.sort_values(["confidence", "row_id"], ascending=[False, True]).head(n)


def _counts_table(frame: pd.DataFrame) -> str:
    rows: list[dict[str, Any]] = []
    for display_name, prefix, _slug in MODEL_SPECS:
        labels = frame["label"].to_numpy()
        prediction = frame[f"{prefix}_prediction"].to_numpy()
        human = int((labels == 0).sum())
        ai = int((labels == 1).sum())
        fp = int(((labels == 0) & (prediction == 1)).sum())
        fn = int(((labels == 1) & (prediction == 0)).sum())
        rows.append(
            {
                "model": display_name,
                "false_positives": fp,
                "false_positive_rate": f"{fp / human:.1%}" if human else "n/a",
                "false_negatives": fn,
                "false_negative_rate": f"{fn / ai:.1%}" if ai else "n/a",
                "accuracy": f"{float((labels == prediction).mean()):.3f}",
            }
        )
    return _markdown_table(pd.DataFrame(rows), list(rows[0].keys()))


def _overlap_prose(overlap: pd.DataFrame) -> str:
    counts = {row.category: int(row.count) for row in overlap.itertuples(index=False)}
    total = sum(counts.values())
    return (
        f"Of {total} test texts, both models are wrong on {counts.get('both_wrong', 0)}. "
        f"TF-IDF alone is wrong on {counts.get('tfidf_only', 0)}. "
        f"DistilBERT alone is wrong on {counts.get('distilbert_only', 0)}. "
        f"Both are right on {counts.get('both_correct', 0)}."
    )


def _example_block(title: str, examples: pd.DataFrame, intro: str) -> list[str]:
    lines = [f"### {title}", "", intro, ""]
    if examples.empty:
        lines.append("None on this test split.")
        return lines
    for rank, (_, row) in enumerate(examples.iterrows(), start=1):
        confidence = float(row["confidence"]) if "confidence" in row and pd.notna(row["confidence"]) else float("nan")
        lines.append(
            f"{rank}. row {int(row['row_id'])}, "
            f"gold={'AI' if int(row['label']) == 1 else 'human'}, "
            f"confidence={confidence:.3f}, words={int(row['n_words'])}"
        )
        lines.append("")
        lines.append("> " + _excerpt(str(row["text"])))
        lines.append("")
    return lines


def _finding_lines(
    frame: pd.DataFrame,
    feature_table: pd.DataFrame,
    bucket_accuracy: pd.DataFrame,
) -> list[str]:
    lines = ["### Findings", ""]
    for display_name, prefix, _slug in MODEL_SPECS:
        correct_n = int((frame["label"].to_numpy() == frame[f"{prefix}_prediction"].to_numpy()).sum())
        incorrect_n = len(frame) - correct_n
        model_rows = feature_table.loc[feature_table["model"] == display_name]
        if incorrect_n < 5:
            lines.append(
                f"- {display_name} has {incorrect_n} mistakes, too few to treat style differences as a pattern."
            )
            continue
        correct = model_rows.loc[model_rows["outcome"] == "correct"].iloc[0]
        incorrect = model_rows.loc[model_rows["outcome"] == "incorrect"].iloc[0]
        shifts = []
        for column, label in FEATURE_COLUMNS:
            sentence = _median_shift(label, float(correct[column]), float(incorrect[column]), column)
            if sentence:
                shifts.append(sentence)
        if shifts:
            lines.append(f"- {display_name} ({incorrect_n} wrong, {correct_n} right): " + " ".join(shifts))
        else:
            lines.append(
                f"- {display_name}: median length, richness, sentence length, punctuation, "
                "and formal-word rate are similar for correct and incorrect texts."
            )
        formal_gap = float(incorrect["formal_present_share"]) - float(correct["formal_present_share"])
        if abs(formal_gap) >= 0.05:
            direction = "more often" if formal_gap > 0 else "less often"
            lines.append(
                f"- {display_name} mistakes contain a formal or technical word {direction} "
                f"({incorrect['formal_present_share']:.1%} of mistakes vs "
                f"{correct['formal_present_share']:.1%} of correct texts)."
            )
    if not bucket_accuracy.empty:
        for display_name, _prefix, _slug in MODEL_SPECS:
            subset = bucket_accuracy.loc[bucket_accuracy["model"] == display_name]
            if subset.empty:
                continue
            worst = subset.loc[subset["accuracy"].idxmin()]
            best = subset.loc[subset["accuracy"].idxmax()]
            if worst["bucket"] != best["bucket"]:
                lines.append(
                    f"- {display_name} accuracy is lowest on {worst['bucket']} texts "
                    f"({worst['accuracy']:.3f}, n={int(worst['n'])}) and highest on {best['bucket']} texts "
                    f"({best['accuracy']:.3f}, n={int(best['n'])})."
                )
    return lines


def _median_shift(label: str, correct_value: float, incorrect_value: float, column: str) -> str:
    baseline = max(abs(correct_value), 1e-6)
    if abs(incorrect_value - correct_value) / baseline < 0.10:
        return ""
    if column == "n_words":
        correct_text, incorrect_text = f"{correct_value:.0f}", f"{incorrect_value:.0f}"
    else:
        correct_text, incorrect_text = f"{correct_value:.3f}", f"{incorrect_value:.3f}"
    direction = "higher" if incorrect_value > correct_value else "lower"
    return (
        f"mistakes have a {direction} median {label.lower()} "
        f"({incorrect_text} vs {correct_text} on correct texts)."
    )


def _format_feature_table(feature_table: pd.DataFrame) -> pd.DataFrame:
    formatted = feature_table.copy()
    for column, _label in FEATURE_COLUMNS:
        formatted[column] = formatted[column].map(lambda value: "n/a" if pd.isna(value) else f"{value:.3f}")
    formatted["formal_present_share"] = formatted["formal_present_share"].map(
        lambda value: "n/a" if pd.isna(value) else f"{value:.1%}"
    )
    return formatted


def _format_bucket_table(bucket_accuracy: pd.DataFrame) -> pd.DataFrame:
    formatted = bucket_accuracy.copy()
    formatted["accuracy"] = formatted["accuracy"].map(lambda value: f"{value:.3f}")
    formatted["median_words"] = formatted["median_words"].map(lambda value: f"{value:.0f}")
    return formatted


def _limitations(frame: pd.DataFrame) -> str:
    human = frame["label"].to_numpy() == 0
    rates = []
    for display_name, prefix, _slug in MODEL_SPECS:
        if not human.any():
            continue
        called_ai = frame[f"{prefix}_prediction"].to_numpy() == 1
        rate = float((human & called_ai).sum() / human.sum())
        rates.append(f"{display_name} {rate:.1%}")
    rate_sentence = (
        "On this test split the false-positive rate on human text is " + " and ".join(rates) + ". "
        if rates
        else ""
    )
    return "\n\n".join(
        [
            "### False accusations of human writers",
            rate_sentence
            + "A false positive is the system telling a human author that their writing is machine-generated. "
            "That accusation is especially risky for non-native English speakers. Classroom English, "
            "more formal connectors, and a smaller vocabulary are ordinary features of second-language writing, "
            "and detectors often treat those same features as signs of a model. This test set has no "
            "native-speaker label, so the boxplots leave that risk unmeasured. "
            "A detector used for grading or hiring can punish a person for how their English looks.",
            "### Easy evasion via paraphrasing",
            "These accuracies leave evasion untested. "
            "A second model, or a person, can rephrase an AI draft until the stock phrases and the "
            "length cues in this test set are gone. The label stays AI, while the text stops "
            "looking like the generations in the training set. Paraphrase attacks are cheap relative to "
            "retraining the detector.",
            "### Drift as new LLMs appear",
            "These models were fit to one mix of human essays and AI essays. ChatGPT, Claude, Gemini, "
            "and later open models each have their own style. A fine-tune that caught yesterday's generator "
            "can miss today's, and a TF-IDF weight on a phrase that model liked becomes a dead feature. "
            "`src/cross_domain.py` checks one outside collection. Each new generator needs its own check.",
            "### Dataset bias",
            "The in-domain test set is a held-out slice of the same Kaggle AI-versus-human collection "
            "used for training. High accuracy there can mean the models learned the prompts, the topics, "
            "or the length mix of that collection. A general difference between people and models is a "
            "stronger claim than this split can support. Other genres and other generators sit outside "
            "that scrape. Read the error examples as failures on this collection.",
        ]
    )


def _excerpt(text: str, n_words: int = EXCERPT_WORDS) -> str:
    words = str(text).replace("\n", " ").split()
    body = " ".join(words[:n_words])
    if len(words) > n_words:
        body += " ..."
    return body


def _console_summary(report: str) -> str:
    findings = []
    capture = False
    for line in report.splitlines():
        if line.startswith("### Findings"):
            capture = True
            continue
        if capture and line.startswith("### "):
            break
        if capture and line.startswith("- "):
            findings.append(line[2:])
    overlap = ""
    for line in report.splitlines():
        if line.startswith("Of "):
            overlap = line
            break
    parts = [overlap, *findings]
    return " ".join(part for part in parts if part)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, rule, *body])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze test-set errors for TF-IDF and DistilBERT.")
    parser.add_argument("--test-csv", type=Path, default=None, help="Defaults to data/processed/test.csv.")
    parser.add_argument("--tfidf-path", type=Path, default=None, help="Defaults to models/tfidf_best.joblib.")
    parser.add_argument(
        "--distilbert-dir",
        type=Path,
        default=None,
        help="Defaults to models/distilbert_detector. Used when saved predictions are missing or do not match the test split.",
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=None,
        help="Optional DistilBERT predictions. Defaults to outputs/metrics/transformer_test_predictions.csv when it matches the test split.",
    )
    parser.add_argument("--metrics-dir", type=Path, default=None, help="Defaults to outputs/metrics.")
    parser.add_argument("--figures-dir", type=Path, default=None, help="Defaults to outputs/figures.")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Defaults to outputs/error_analysis_report.md.",
    )
    return parser


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    main()
