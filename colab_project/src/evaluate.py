"""Compare the saved TF-IDF pipeline and DistilBERT on the same test split.

Both models are scored on ``data/processed/test.csv``. The report is a metric
table, three figures, and McNemar's test on paired correctness.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import load
from scipy.stats import chi2
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import (
    BATCH_SIZE,
    FIGURES_DIR,
    MAX_LEN,
    METRICS_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    SEED,
)
from src.data_utils import set_seed

logger = logging.getLogger(__name__)

POSITIVE_LABEL = 1
TFIDF_FILENAME = "tfidf_best.joblib"
DISTILBERT_DIRNAME = "distilbert_detector"
CLASSIFICATION_METRICS: tuple[str, ...] = ("accuracy", "precision", "recall", "f1", "roc_auc")
METRIC_LABELS: dict[str, str] = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
}
MODEL_COLORS: dict[str, str] = {"TF-IDF": "#4C78A8", "DistilBERT": "#F58518"}


def load_test_set(path: Path | None = None) -> pd.DataFrame:
    """Load the held-out test CSV.

    Parameters
    ----------
    path:
        CSV location. Defaults to ``data/processed/test.csv``.

    Returns
    -------
    pd.DataFrame
        Columns ``text`` and ``label`` (0 = human, 1 = AI).
    """
    csv_path = PROCESSED_DATA_DIR / "test.csv" if path is None else Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing test split at {_display_path(csv_path)}. "
            "Run `python -m src.data_utils` before evaluation."
        )
    frame = pd.read_csv(csv_path)
    if "text" not in frame.columns or "label" not in frame.columns:
        raise ValueError(
            f"{_display_path(csv_path)} must contain 'text' and 'label' columns. "
            f"Found {list(frame.columns)}."
        )
    before = len(frame)
    frame = frame.dropna(subset=["text", "label"]).copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame.loc[frame["text"].ne("")].copy()
    frame["label"] = frame["label"].astype(int)
    unexpected = sorted(set(frame["label"].unique()) - {0, 1})
    if unexpected:
        raise ValueError(f"Test split has labels other than 0 and 1: {unexpected}.")
    if frame["label"].nunique() < 2:
        raise ValueError("Test split must contain both human (0) and AI (1) rows.")
    dropped = before - len(frame)
    if dropped:
        logger.warning("Dropped %d empty rows from the test split.", dropped)
    frame = frame.reset_index(drop=True)
    logger.info("Loaded %s: %d rows.", _display_path(csv_path), len(frame))
    return frame


def load_tfidf(path: Path | None = None) -> Any:
    """Load the saved scikit-learn TF-IDF pipeline."""
    model_path = MODELS_DIR / TFIDF_FILENAME if path is None else Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"TF-IDF model not found at {_display_path(model_path)}. "
            "Run `python -m src.train_baseline` first."
        )
    logger.info("Loading TF-IDF pipeline from %s.", _display_path(model_path))
    return load(model_path)


def score_tfidf(pipeline: Any, texts: pd.Series) -> dict[str, Any]:
    """Return AI probabilities, hard labels, and seconds per 1,000 texts.

    Timing covers ``predict_proba`` on the full series after one warmup batch.
    Loading the joblib file is not included.
    """
    text_list = texts.astype(str).tolist()
    if not text_list:
        raise ValueError("score_tfidf received no texts.")
    warmup = text_list[: min(32, len(text_list))]
    pipeline.predict_proba(warmup)
    started = time.perf_counter()
    probabilities = np.asarray(pipeline.predict_proba(text_list), dtype=float)
    elapsed = time.perf_counter() - started
    positive = _positive_column(probabilities, _classifier_classes(pipeline))
    logger.info("TF-IDF inference: %.3fs for %d texts.", elapsed, len(text_list))
    return {
        "name": "TF-IDF",
        "probabilities": positive,
        "predictions": (positive >= 0.5).astype(int),
        "seconds_per_1000": elapsed / len(text_list) * 1000.0,
        "device": "cpu",
    }


def warn_if_partial_finetune(model_dir: Path) -> None:
    """Warn when the checkpoint was fit on a capped smoke-test sample.

    ``python -m src.train_transformer --max_samples 500`` writes the weights to
    the same folder as a full run. Later scripts would otherwise report those
    weights as the project result.
    """
    meta_path = Path(model_dir) / "training_meta.json"
    if not meta_path.is_file():
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Could not read %s.", _display_path(meta_path))
        return
    cap = meta.get("max_samples")
    if cap:
        logger.warning(
            "DistilBERT at %s was fine-tuned with --max_samples %s. "
            "That checkpoint is a smoke test. Run `python -m src.train_transformer` "
            "without --max_samples before reporting test, cross-domain, or demo results.",
            _display_path(model_dir),
            cap,
        )


def score_distilbert(model_dir: Path, texts: pd.Series, batch_size: int = BATCH_SIZE) -> dict[str, Any]:
    """Return AI probabilities from the fine-tuned DistilBERT checkpoint.

    Timing covers tokenization and the forward pass for the full test set
    after one warmup batch. Moving the model onto the device is not included.
    """
    if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
        raise FileNotFoundError(
            f"DistilBERT model not found at {_display_path(model_dir)}. "
            "Run `python -m src.train_transformer` first."
        )
    warn_if_partial_finetune(model_dir)
    text_list = texts.astype(str).tolist()
    if not text_list:
        raise ValueError("score_distilbert received no texts.")
    torch, tokenizer_cls, model_cls = _import_torch_stack()
    logger.info("Loading DistilBERT from %s.", _display_path(model_dir))
    tokenizer = tokenizer_cls.from_pretrained(model_dir)
    model = model_cls.from_pretrained(model_dir)
    if int(model.config.num_labels) != 2:
        raise RuntimeError(
            f"Expected a 2-label DistilBERT head, found num_labels={model.config.num_labels}."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    if device.type == "cuda":
        logger.info("DistilBERT inference device: %s.", torch.cuda.get_device_name(0))
    else:
        logger.info("DistilBERT inference device: CPU.")

    positive_index = _positive_label_index(model.config.id2label)
    warmup = text_list[: min(batch_size, len(text_list))]
    _distilbert_probabilities(torch, model, tokenizer, warmup, device, batch_size, positive_index)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    positive = _distilbert_probabilities(
        torch, model, tokenizer, text_list, device, batch_size, positive_index
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    logger.info("DistilBERT inference: %.3fs for %d texts.", elapsed, len(text_list))
    return {
        "name": "DistilBERT",
        "probabilities": positive,
        "predictions": (positive >= 0.5).astype(int),
        "seconds_per_1000": elapsed / len(text_list) * 1000.0,
        "device": device.type,
    }


def classification_scores(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Accuracy, precision, recall, F1, and ROC-AUC. AI (1) is the positive class."""
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        "recall": float(recall_score(labels, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        "f1": float(f1_score(labels, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def mcnemar_test(labels: np.ndarray, tfidf_pred: np.ndarray, distilbert_pred: np.ndarray) -> dict[str, Any]:
    """McNemar's test with Edwards' continuity correction.

    The null hypothesis is that the two models are wrong on the same texts
    equally often. ``tfidf_only_correct`` and ``distilbert_only_correct`` are
    the off-diagonal counts.
    """
    tfidf_correct = np.asarray(tfidf_pred) == np.asarray(labels)
    distilbert_correct = np.asarray(distilbert_pred) == np.asarray(labels)
    both_correct = int(np.sum(tfidf_correct & distilbert_correct))
    both_wrong = int(np.sum(~tfidf_correct & ~distilbert_correct))
    tfidf_only = int(np.sum(tfidf_correct & ~distilbert_correct))
    distilbert_only = int(np.sum(~tfidf_correct & distilbert_correct))
    discordant = tfidf_only + distilbert_only
    if discordant == 0 or tfidf_only == distilbert_only:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = (abs(tfidf_only - distilbert_only) - 1) ** 2 / discordant
        p_value = float(chi2.sf(statistic, df=1))
    return {
        "test": "mcnemar_continuity_corrected",
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "tfidf_only_correct": tfidf_only,
        "distilbert_only_correct": distilbert_only,
        "statistic": float(statistic),
        "p_value": p_value,
        "alpha": 0.05,
        "significant": bool(p_value < 0.05),
    }


def plain_summary(rows: list[dict[str, Any]], test: dict[str, Any]) -> str:
    """One short paragraph: who wins on F1, whether the gap is significant, and the cost."""
    by_name = {row["model"]: row for row in rows}
    tfidf = by_name["TF-IDF"]
    distilbert = by_name["DistilBERT"]
    gap = float(distilbert["f1"] - tfidf["f1"])
    if gap > 0:
        lead = (
            f"DistilBERT wins on F1 by {gap:.3f} "
            f"({distilbert['f1']:.3f} vs {tfidf['f1']:.3f} for TF-IDF)."
        )
    elif gap < 0:
        lead = (
            f"TF-IDF wins on F1 by {-gap:.3f} "
            f"({tfidf['f1']:.3f} vs {distilbert['f1']:.3f} for DistilBERT)."
        )
    else:
        lead = (
            f"The models tie on F1 at {tfidf['f1']:.3f}."
        )
    disagree = int(test["tfidf_only_correct"] + test["distilbert_only_correct"])
    if disagree == 0:
        paired = "They are right or wrong together on every test text."
    else:
        paired = (
            f"On the {disagree} test texts where they disagree, "
            f"DistilBERT is right on {test['distilbert_only_correct']} "
            f"and TF-IDF is right on {test['tfidf_only_correct']}."
        )
    if test["significant"]:
        significance = (
            f"McNemar's test says that difference in errors is statistically significant "
            f"(p = {_format_p(test['p_value'])})."
        )
    else:
        significance = (
            f"McNemar's test does not call the difference significant "
            f"(p = {_format_p(test['p_value'])})."
        )
    cost = (
        f"TF-IDF takes {tfidf['seconds_per_1000']:.3f}s per 1,000 texts on CPU "
        f"and uses {tfidf['size_mb']:.1f} MB. "
        f"DistilBERT takes {distilbert['seconds_per_1000']:.3f}s per 1,000 on {distilbert['device']} "
        f"and uses {distilbert['size_mb']:.1f} MB."
    )
    return " ".join([lead, paired, significance, cost])


def save_comparison_table(rows: list[dict[str, Any]], summary: str, test: dict[str, Any], metrics_dir: Path) -> None:
    """Write the comparison as CSV and Markdown."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "seconds_per_1000",
        "size_mb",
        "device",
    ]
    csv_path = metrics_dir / "model_comparison.csv"
    table.loc[:, columns].to_csv(csv_path, index=False)
    logger.info("Wrote %s", _display_path(csv_path))

    markdown_path = metrics_dir / "model_comparison.md"
    markdown_path.write_text(_comparison_markdown(table, summary, test), encoding="utf-8")
    logger.info("Wrote %s", _display_path(markdown_path))


def save_summary_json(payload: dict[str, Any], metrics_dir: Path) -> None:
    """Write the machine-readable comparison, including McNemar's test."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / "comparison_summary.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s", _display_path(path))


def plot_metric_bars(rows: list[dict[str, Any]], path: Path) -> None:
    """Grouped bars for the shared classification metrics, plus time and size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    by_name = {row["model"]: row for row in rows}
    names = ["TF-IDF", "DistilBERT"]
    colors = [MODEL_COLORS[name] for name in names]
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.3), gridspec_kw={"width_ratios": [2.5, 1.0, 1.0]})

    x = np.arange(len(CLASSIFICATION_METRICS))
    width = 0.36
    for offset, name in ((-width / 2, "TF-IDF"), (width / 2, "DistilBERT")):
        values = [by_name[name][metric] for metric in CLASSIFICATION_METRICS]
        axes[0].bar(x + offset, values, width=width, label=name, color=MODEL_COLORS[name])
    axes[0].set_xticks(x, [METRIC_LABELS[metric] for metric in CLASSIFICATION_METRICS])
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Classification metrics")
    axes[0].legend(frameon=False)

    axes[1].bar(names, [by_name[name]["seconds_per_1000"] for name in names], color=colors)
    axes[1].set_title("Seconds / 1,000 texts")
    axes[1].tick_params(axis="x", labelrotation=15)

    axes[2].bar(names, [by_name[name]["size_mb"] for name in names], color=colors)
    axes[2].set_title("Size on disk (MB)")
    axes[2].tick_params(axis="x", labelrotation=15)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", _display_path(path))


def plot_roc_curves(labels: np.ndarray, scores: list[dict[str, Any]], path: Path) -> None:
    """ROC curves for both models on one set of axes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for score in scores:
        fpr, tpr, _ = roc_curve(labels, score["probabilities"])
        auc = float(roc_auc_score(labels, score["probabilities"]))
        ax.plot(fpr, tpr, color=MODEL_COLORS[score["name"]], label=f"{score['name']} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on the held-out test set")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", _display_path(path))


def plot_calibration_curves(labels: np.ndarray, scores: list[dict[str, Any]], path: Path) -> None:
    """Reliability diagram for both models. AI is the positive class."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", label="Perfect")
    for score in scores:
        fraction_positive, mean_predicted = calibration_curve(
            labels,
            score["probabilities"],
            n_bins=10,
            strategy="uniform",
        )
        ax.plot(
            mean_predicted,
            fraction_positive,
            marker="o",
            color=MODEL_COLORS[score["name"]],
            label=score["name"],
        )
    ax.set_xlabel("Predicted probability of AI")
    ax.set_ylabel("Fraction of texts that are AI")
    ax.set_title("Calibration curves")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", _display_path(path))


def main(argv: list[str] | None = None) -> None:
    """Score both saved models on the test split and write the comparison."""
    _configure_logging()
    args = _build_parser().parse_args(argv)
    set_seed(SEED)

    test = load_test_set(args.test_csv)
    tfidf_path = MODELS_DIR / TFIDF_FILENAME if args.tfidf_path is None else Path(args.tfidf_path)
    distilbert_dir = MODELS_DIR / DISTILBERT_DIRNAME if args.distilbert_dir is None else Path(args.distilbert_dir)
    metrics_dir = METRICS_DIR if args.metrics_dir is None else Path(args.metrics_dir)
    figures_dir = FIGURES_DIR if args.figures_dir is None else Path(args.figures_dir)

    pipeline = load_tfidf(tfidf_path)
    tfidf_score = score_tfidf(pipeline, test["text"])
    distilbert_score = score_distilbert(distilbert_dir, test["text"], batch_size=args.batch_size)
    labels = test["label"].to_numpy()
    _check_alignment(labels, tfidf_score["probabilities"], distilbert_score["probabilities"])

    rows = [
        _result_row(tfidf_score, labels, path_size_bytes(tfidf_path)),
        _result_row(distilbert_score, labels, path_size_bytes(distilbert_dir)),
    ]
    paired = mcnemar_test(labels, tfidf_score["predictions"], distilbert_score["predictions"])
    summary = plain_summary(rows, paired)
    logger.info(
        "TF-IDF F1=%.4f ROC-AUC=%.4f | DistilBERT F1=%.4f ROC-AUC=%.4f | McNemar p=%s",
        rows[0]["f1"],
        rows[0]["roc_auc"],
        rows[1]["f1"],
        rows[1]["roc_auc"],
        _format_p(paired["p_value"]),
    )

    save_comparison_table(rows, summary, paired, metrics_dir)
    save_summary_json(
        {
            "seed": SEED,
            "n_test": int(len(test)),
            "positive_label": POSITIVE_LABEL,
            "positive_class": "AI",
            "models": rows,
            "mcnemar": paired,
            "summary": summary,
        },
        metrics_dir,
    )
    plot_metric_bars(rows, figures_dir / "metric_comparison.png")
    plot_roc_curves(labels, [tfidf_score, distilbert_score], figures_dir / "roc_curves.png")
    plot_calibration_curves(labels, [tfidf_score, distilbert_score], figures_dir / "calibration_curves.png")
    print()
    print(summary)


def _format_p(p_value: float) -> str:
    if p_value < 0.0001:
        return f"{p_value:.2e}"
    return f"{p_value:.4f}"


def _result_row(score: dict[str, Any], labels: np.ndarray, size_bytes: int) -> dict[str, Any]:
    metrics = classification_scores(labels, score["predictions"], score["probabilities"])
    return {
        "model": score["name"],
        **metrics,
        "seconds_per_1000": float(score["seconds_per_1000"]),
        "size_bytes": int(size_bytes),
        "size_mb": float(size_bytes / (1024 * 1024)),
        "device": score["device"],
    }


def _comparison_markdown(table: pd.DataFrame, summary: str, test: dict[str, Any]) -> str:
    header = (
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Seconds / 1000 | Size (MB) | Device |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n"
    )
    lines = [header]
    for _, row in table.iterrows():
        lines.append(
            "| {model} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} | "
            "{roc_auc:.4f} | {seconds_per_1000:.3f} | {size_mb:.2f} | {device} |\n".format(**row.to_dict())
        )
    significance = "yes" if test["significant"] else "no"
    body = (
        "# Held-out test comparison\n\n"
        "Both models are scored on the same `data/processed/test.csv` rows. "
        "The positive class is AI (label 1). Seconds per 1,000 texts is inference only, "
        "after one warmup batch, and does not include loading the model. "
        "Disk size is `models/tfidf_best.joblib` and the `models/distilbert_detector` folder.\n\n"
        f"{summary}\n\n"
        + "".join(lines)
        + "\n## McNemar's test\n\n"
        "Paired test of whether the models are wrong on different texts more often than chance. "
        "The statistic uses Edwards' continuity correction.\n\n"
        f"- TF-IDF only correct: {test['tfidf_only_correct']}\n"
        f"- DistilBERT only correct: {test['distilbert_only_correct']}\n"
        f"- Both correct: {test['both_correct']}\n"
        f"- Both wrong: {test['both_wrong']}\n"
        f"- Statistic: {test['statistic']:.4f}\n"
        f"- p-value: {_format_p(test['p_value'])}\n"
        f"- Significant at 5%: {significance}\n"
    )
    return body


def _distilbert_probabilities(
    torch: Any,
    model: Any,
    tokenizer: Any,
    texts: list[str],
    device: Any,
    batch_size: int,
    positive_index: int,
) -> np.ndarray:
    batches: list[np.ndarray] = []
    total = max(1, (len(texts) + batch_size - 1) // batch_size)
    for batch_number, start in enumerate(range(0, len(texts), batch_size), start=1):
        chunk = texts[start : start + batch_size]
        encoded = tokenizer(
            chunk,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)[:, positive_index]
        batches.append(probabilities.detach().cpu().numpy())
        if batch_number == 1 or batch_number == total or batch_number % 50 == 0:
            logger.info("DistilBERT batch %d/%d.", batch_number, total)
    return np.clip(np.concatenate(batches), 0.0, 1.0)


def _positive_label_index(id2label: dict[Any, Any]) -> int:
    for key, name in id2label.items():
        if str(name).strip().lower() in {"ai", "generated", "1"}:
            return int(key)
    return POSITIVE_LABEL


def _classifier_classes(pipeline: Any) -> list[int]:
    classifier = pipeline.named_steps.get("classifier", pipeline.steps[-1][1])
    if not hasattr(classifier, "classes_"):
        raise RuntimeError("The TF-IDF pipeline classifier has no classes_.")
    return [int(label) for label in classifier.classes_]


def _positive_column(probabilities: np.ndarray, classes: list[int]) -> np.ndarray:
    if probabilities.ndim != 2:
        raise RuntimeError(f"Expected predict_proba with shape (n, n_classes), got {probabilities.shape}.")
    if POSITIVE_LABEL not in classes:
        raise RuntimeError(f"TF-IDF classes are {classes}; expected label {POSITIVE_LABEL}.")
    column = classes.index(POSITIVE_LABEL)
    return np.clip(probabilities[:, column], 0.0, 1.0)


def _check_alignment(labels: np.ndarray, tfidf_probabilities: np.ndarray, distilbert_probabilities: np.ndarray) -> None:
    count = len(labels)
    if len(tfidf_probabilities) != count or len(distilbert_probabilities) != count:
        raise RuntimeError(
            "Prediction lengths do not match the test set: "
            f"test={count}, tfidf={len(tfidf_probabilities)}, distilbert={len(distilbert_probabilities)}."
        )


def path_size_bytes(path: Path) -> int:
    """Bytes used by a file, or by every file under a directory."""
    if path.is_file():
        return int(path.stat().st_size)
    if not path.exists():
        raise FileNotFoundError(f"Cannot measure a missing path: {_display_path(path)}.")
    return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))


def _import_torch_stack() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "DistilBERT evaluation needs torch and transformers. "
            "Install them with `pip install -r requirements.txt`."
        ) from exc
    return torch, AutoTokenizer, AutoModelForSequenceClassification


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the saved TF-IDF and DistilBERT models on the held-out test set."
    )
    parser.add_argument("--test-csv", type=Path, default=None, help="Defaults to data/processed/test.csv.")
    parser.add_argument("--tfidf-path", type=Path, default=None, help="Defaults to models/tfidf_best.joblib.")
    parser.add_argument(
        "--distilbert-dir",
        type=Path,
        default=None,
        help="Defaults to models/distilbert_detector.",
    )
    parser.add_argument("--metrics-dir", type=Path, default=None, help="Defaults to outputs/metrics.")
    parser.add_argument("--figures-dir", type=Path, default=None, help="Defaults to outputs/figures.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"DistilBERT inference batch size. Defaults to {BATCH_SIZE}.",
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
