"""Train TF-IDF baselines and keep the model that wins on the validation split.

Logistic regression and a calibrated linear SVM share one bigram TF-IDF
pipeline. ``C`` is chosen from a small grid by validation F1. The test split
is used only after that choice, for the reported metrics.
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
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import FIGURES_DIR, METRICS_DIR, MODELS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, SEED
from src.data_utils import assert_no_text_leakage, set_seed

logger = logging.getLogger(__name__)

C_GRID: tuple[float, ...] = (0.1, 1.0, 10.0)
TOP_FEATURES: int = 20
POSITIVE_LABEL: int = 1
CALIBRATION_FOLDS: int = 5

TFIDF_PARAMS: dict[str, Any] = {
    "ngram_range": (1, 2),
    "max_features": 50_000,
    "sublinear_tf": True,
    "min_df": 2,
}


def load_splits(data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the processed train, validation, and test CSVs.

    Parameters
    ----------
    data_dir:
        Directory that contains ``train.csv``, ``val.csv``, and ``test.csv``.
        Defaults to ``data/processed``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Train, validation, and test frames with columns ``text`` and ``label``.
    """
    directory = PROCESSED_DATA_DIR if data_dir is None else Path(data_dir)
    splits: list[pd.DataFrame] = []
    for name in ("train", "val", "test"):
        path = directory / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {name} split at {_display_path(path)}. "
                "Run `python -m src.data_utils` before training the baseline."
            )
        frame = pd.read_csv(path)
        if "text" not in frame.columns or "label" not in frame.columns:
            raise ValueError(
                f"{_display_path(path)} must contain 'text' and 'label' columns. "
                f"Found {list(frame.columns)}."
            )
        before = len(frame)
        frame = frame.dropna(subset=["text", "label"]).copy()
        frame["text"] = frame["text"].astype(str).str.strip()
        frame = frame.loc[frame["text"].ne("")].copy()
        frame["label"] = frame["label"].astype(int)
        unexpected = sorted(set(frame["label"].unique()) - {0, 1})
        if unexpected:
            raise ValueError(f"{name} split has labels other than 0 and 1: {unexpected}.")
        if frame["label"].nunique() < 2:
            raise ValueError(f"{name} split must contain both human (0) and AI (1) rows.")
        dropped = before - len(frame)
        if dropped:
            logger.warning("Dropped %d empty rows from the %s split.", dropped, name)
        logger.info("Loaded %s: %d rows.", _display_path(path), len(frame))
        splits.append(frame.reset_index(drop=True))
    assert_no_text_leakage(splits[0], splits[1], splits[2])
    logger.info("No text is shared across train, validation, and test.")
    return splits[0], splits[1], splits[2]


def build_pipeline(classifier: Any) -> Pipeline:
    """Pair a fresh TF-IDF vectorizer with ``classifier``."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
            ("classifier", classifier),
        ]
    )


def select_model(
    train_text: pd.Series,
    train_labels: pd.Series,
    val_text: pd.Series,
    val_labels: pd.Series,
) -> tuple[Pipeline, dict[str, Any], list[dict[str, Any]]]:
    """Fit each ``C`` on train and keep the pipeline with the best validation F1.

    The test split is not read here. LinearSVC is wrapped in
    ``CalibratedClassifierCV`` so every candidate can emit probabilities.

    Returns
    -------
    tuple[Pipeline, dict[str, Any], list[dict[str, Any]]]
        Fitted winning pipeline, its validation record, and every candidate.
    """
    candidates = _candidate_specs()
    records: list[dict[str, Any]] = []
    best_pipeline: Pipeline | None = None
    best_record: dict[str, Any] | None = None
    best_key: tuple[float, float, float] | None = None

    logger.info(
        "Tuning %d candidates on the validation set (selection metric: F1 for AI).",
        len(candidates),
    )
    for index, (name, c_value, classifier) in enumerate(candidates, start=1):
        pipeline = build_pipeline(classifier)
        logger.info("Fitting %d/%d: %s C=%s", index, len(candidates), name, c_value)
        started = time.perf_counter()
        pipeline.fit(train_text, train_labels)
        elapsed = time.perf_counter() - started
        scores = _classification_scores(val_labels, pipeline.predict(val_text), _positive_proba(pipeline, val_text))
        record: dict[str, Any] = {
            "model": name,
            "C": c_value,
            "fit_seconds": round(elapsed, 2),
            **{f"val_{key}": value for key, value in scores.items()},
        }
        records.append(record)
        logger.info(
            "  val accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f (%.1fs)",
            scores["accuracy"],
            scores["precision"],
            scores["recall"],
            scores["f1"],
            scores["roc_auc"],
            elapsed,
        )
        key = (scores["f1"], scores["roc_auc"], scores["accuracy"])
        if best_key is None or key > best_key:
            best_key = key
            best_pipeline = pipeline
            best_record = record

    if best_pipeline is None or best_record is None:
        raise RuntimeError("Model selection produced no candidate.")
    logger.info(
        "Selected %s with C=%s by validation F1 (%.4f). Test data was not used.",
        best_record["model"],
        best_record["C"],
        best_record["val_f1"],
    )
    return best_pipeline, best_record, records


def evaluate_model(
    pipeline: Pipeline,
    text: pd.Series,
    labels: pd.Series,
) -> dict[str, Any]:
    """Score a fitted pipeline. Precision, recall, and F1 use AI as the positive class."""
    predictions = pipeline.predict(text)
    probabilities = _positive_proba(pipeline, text)
    scores = _classification_scores(labels, predictions, probabilities)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        **scores,
        "confusion_matrix": matrix.astype(int).tolist(),
        "labels": ["human", "ai"],
    }


def top_predictive_features(pipeline: Pipeline, k: int = TOP_FEATURES) -> dict[str, list[dict[str, float | str]]]:
    """Return the ``k`` tokens with the strongest weight toward each class.

    Positive weights point at AI (label 1). For the calibrated SVM, the weight
    is the mean coefficient of the LinearSVC models fitted inside each fold.
    """
    names = np.asarray(pipeline.named_steps["tfidf"].get_feature_names_out())
    weights = _decision_weights(pipeline)
    if names.shape[0] != weights.shape[0]:
        raise RuntimeError(
            f"Feature count ({names.shape[0]}) does not match weight count ({weights.shape[0]})."
        )
    limit = min(k, int(weights.shape[0]))
    order = np.argsort(weights)
    human_index = order[:limit]
    ai_index = order[-limit:][::-1]
    return {
        "human": [_feature_row(names[index], weights[index]) for index in human_index],
        "ai": [_feature_row(names[index], weights[index]) for index in ai_index],
    }


def save_confusion_matrix(matrix: list[list[int]], path: Path, title: str) -> None:
    """Write a labeled confusion-matrix heatmap."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    sns.heatmap(
        np.asarray(matrix),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Human", "AI"],
        yticklabels=["Human", "AI"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", _display_path(path))


def save_metrics(payload: dict[str, Any], path: Path) -> None:
    """Write ``payload`` as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s", _display_path(path))


def save_model(pipeline: Pipeline, path: Path) -> None:
    """Persist the fitted pipeline with joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(pipeline, path)
    logger.info("Wrote %s", _display_path(path))


def main(argv: list[str] | None = None) -> None:
    """Tune the TF-IDF baselines, score the winner on test, and save artifacts."""
    _configure_logging()
    args = _build_parser().parse_args(argv)
    set_seed(SEED)

    train, val, test = load_splits(args.data_dir)
    pipeline, winner, candidates = select_model(train["text"], train["label"], val["text"], val["label"])
    logger.info("Evaluating the selected model on the test set.")
    test_metrics = evaluate_model(pipeline, test["text"], test["label"])
    logger.info(
        "Test accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f",
        test_metrics["accuracy"],
        test_metrics["precision"],
        test_metrics["recall"],
        test_metrics["f1"],
        test_metrics["roc_auc"],
    )
    logger.info(
        "Confusion matrix [rows true human/AI, cols predicted human/AI]: %s",
        test_metrics["confusion_matrix"],
    )

    features = top_predictive_features(pipeline)
    _log_features(features)

    model_path = MODELS_DIR / "tfidf_best.joblib" if args.model_path is None else Path(args.model_path)
    metrics_path = METRICS_DIR / "baseline_metrics.json" if args.metrics_path is None else Path(args.metrics_path)
    figure_path = FIGURES_DIR / "cm_baseline.png" if args.figure_path is None else Path(args.figure_path)

    payload = {
        "seed": SEED,
        "positive_label": POSITIVE_LABEL,
        "positive_class": "AI",
        "selection_metric": "val_f1",
        "vectorizer": {
            "ngram_range": list(TFIDF_PARAMS["ngram_range"]),
            "max_features": TFIDF_PARAMS["max_features"],
            "sublinear_tf": TFIDF_PARAMS["sublinear_tf"],
            "min_df": TFIDF_PARAMS["min_df"],
        },
        "c_grid": list(C_GRID),
        "calibration_folds": CALIBRATION_FOLDS,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "best_model": winner["model"],
        "best_C": winner["C"],
        "validation": {
            "accuracy": winner["val_accuracy"],
            "precision": winner["val_precision"],
            "recall": winner["val_recall"],
            "f1": winner["val_f1"],
            "roc_auc": winner["val_roc_auc"],
        },
        "candidates": candidates,
        "test": test_metrics,
        "top_features": features,
    }
    save_model(pipeline, model_path)
    save_metrics(payload, metrics_path)
    save_confusion_matrix(
        test_metrics["confusion_matrix"],
        figure_path,
        title=f"TF-IDF {winner['model']} (C={winner['C']})",
    )


def _candidate_specs() -> list[tuple[str, float, Any]]:
    specs: list[tuple[str, float, Any]] = []
    for c_value in C_GRID:
        specs.append(("logistic_regression", c_value, _logistic(c_value)))
        specs.append(("linear_svc", c_value, _linear_svc(c_value)))
    return specs


def _logistic(c_value: float) -> LogisticRegression:
    return LogisticRegression(
        C=c_value,
        solver="liblinear",
        max_iter=2000,
        random_state=SEED,
    )


def _linear_svc(c_value: float) -> CalibratedClassifierCV:
    estimator = LinearSVC(
        C=c_value,
        dual="auto",
        max_iter=5000,
        random_state=SEED,
    )
    cv = StratifiedKFold(n_splits=CALIBRATION_FOLDS, shuffle=True, random_state=SEED)
    return CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=cv)


def _positive_proba(pipeline: Pipeline, text: pd.Series) -> np.ndarray:
    classifier = pipeline.named_steps["classifier"]
    probabilities = pipeline.predict_proba(text)
    class_list = [int(label) for label in classifier.classes_]
    if POSITIVE_LABEL not in class_list:
        raise RuntimeError(f"Classifier classes are {class_list}; expected label {POSITIVE_LABEL}.")
    return np.asarray(probabilities)[:, class_list.index(POSITIVE_LABEL)]


def _classification_scores(
    labels: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        "recall": float(recall_score(labels, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        "f1": float(f1_score(labels, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def _decision_weights(pipeline: Pipeline) -> np.ndarray:
    classifier = pipeline.named_steps["classifier"]
    classes = [int(label) for label in classifier.classes_]
    if hasattr(classifier, "coef_"):
        weights = np.asarray(classifier.coef_, dtype=float).ravel()
    else:
        fold_weights = [
            np.asarray(calibrated.estimator.coef_, dtype=float).ravel()
            for calibrated in classifier.calibrated_classifiers_
        ]
        if not fold_weights:
            raise RuntimeError("Calibrated LinearSVC did not expose fold coefficients.")
        weights = np.mean(fold_weights, axis=0)
        logger.info("LinearSVC feature weights are the mean coefficient across %d calibration folds.", len(fold_weights))
    if classes == [POSITIVE_LABEL, 0]:
        weights = -weights
    elif classes != [0, POSITIVE_LABEL]:
        raise RuntimeError(f"Unexpected class order {classes}.")
    return weights


def _feature_row(feature: str, weight: float) -> dict[str, float | str]:
    return {"feature": str(feature), "weight": round(float(weight), 6)}


def _log_features(features: dict[str, list[dict[str, float | str]]]) -> None:
    for class_name in ("human", "ai"):
        logger.info("Top %d features signaling %s:", len(features[class_name]), class_name)
        for rank, row in enumerate(features[class_name], start=1):
            logger.info("  %2d. %-24s % .6f", rank, row["feature"], float(row["weight"]))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune TF-IDF logistic regression and linear SVM baselines on the validation split."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory with train.csv, val.csv, and test.csv. Defaults to data/processed.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Where to save the best pipeline. Defaults to models/tfidf_best.joblib.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=None,
        help="Where to save metrics JSON. Defaults to outputs/metrics/baseline_metrics.json.",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=None,
        help="Where to save the confusion matrix. Defaults to outputs/figures/cm_baseline.png.",
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
