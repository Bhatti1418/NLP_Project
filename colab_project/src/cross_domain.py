"""Score TF-IDF and DistilBERT on text they were not trained on.

The out-of-domain set is Hello-SimpleAI/HC3: human answers and ChatGPT
answers to the same questions. It is cleaned with the same rules as the
Kaggle essays and cut to a balanced sample of about 2,000 rows. In-domain
numbers come from the saved test comparison when that file exists.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, METRICS_DIR, MODELS_DIR, PROJECT_ROOT, RAW_DATA_DIR, SEED
from src.data_utils import clean_dataframe, set_seed, subsample_balanced
from src.evaluate import (
    CLASSIFICATION_METRICS,
    DISTILBERT_DIRNAME,
    METRIC_LABELS,
    classification_scores,
    load_test_set,
    load_tfidf,
    score_distilbert,
    score_tfidf,
)

logger = logging.getLogger(__name__)

CROSS_DOMAIN_ROWS = 2_000
HC3_REPO = "Hello-SimpleAI/HC3"
HC3_CONFIGS: tuple[str, ...] = ("all", "open_qa", "finance")
FALLBACK_DATASET = "Ateeqq/AI-and-Human-Generated-Text"
IN_DOMAIN_COLOR = "#4C78A8"
CROSS_DOMAIN_COLOR = "#F58518"
COMPARISON_JSON = METRICS_DIR / "comparison_summary.json"


def load_cross_domain_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load HC3, or a public fallback, as ``text`` / ``label`` rows.

    Label 0 is human and label 1 is AI. HC3 configs are tried in order:
    ``all``, ``open_qa``, then ``finance``.
    """
    errors: list[str] = []
    for config in HC3_CONFIGS:
        frame, error = _try_hc3_config(config)
        if frame is not None:
            try:
                labeled = hc3_frame_to_labeled(frame, config)
            except ValueError as exc:
                error = str(exc)
            else:
                description = (
                    "Human answers (label 0) and ChatGPT answers (label 1) to the same "
                    f"questions, from {HC3_REPO} config '{config}'. This question-answer "
                    "corpus is outside the Kaggle essay mix used for training."
                )
                logger.info(
                    "Using %s config %s (%d answer rows before cleaning).",
                    HC3_REPO,
                    config,
                    len(labeled),
                )
                return labeled, {
                    "id": HC3_REPO,
                    "config": config,
                    "fallback": False,
                    "description": description,
                    "hc3_errors": errors,
                }
        errors.append(f"{config}: {error}")
        logger.warning("HC3 config %s failed: %s", config, error)

    logger.warning("HC3 could not be loaded. Falling back to %s.", FALLBACK_DATASET)
    labeled = _load_aiga_fallback()
    description = (
        f"HC3 failed to load ({'; '.join(errors)}). "
        f"Used {FALLBACK_DATASET} instead: human scientific abstracts are label 0 and "
        "GPT-3 abstracts are label 1. That corpus is still outside the Kaggle essay "
        "training mix, but it is not the HC3 question-answer set."
    )
    logger.info(description)
    return labeled, {
        "id": FALLBACK_DATASET,
        "config": "default",
        "fallback": True,
        "description": description,
        "hc3_errors": errors,
    }


def hc3_frame_to_labeled(frame: pd.DataFrame, config: str) -> pd.DataFrame:
    """Expand HC3 answer lists into one row per human or ChatGPT answer."""
    missing = {"human_answers", "chatgpt_answers"} - set(frame.columns)
    if missing:
        raise ValueError(f"HC3 config {config} is missing {sorted(missing)}. Found {list(frame.columns)}.")
    working = frame.copy()
    if "source" not in working.columns:
        working["source"] = config
    working["human_answers"] = working["human_answers"].map(_ensure_list)
    working["chatgpt_answers"] = working["chatgpt_answers"].map(_ensure_list)
    pieces = [
        _explode_answers(working, "human_answers", 0),
        _explode_answers(working, "chatgpt_answers", 1),
    ]
    labeled = pd.concat(pieces, ignore_index=True)
    labeled["text"] = labeled["text"].map(_clean_answer)
    labeled = labeled.loc[labeled["text"].ne("")].reset_index(drop=True)
    if labeled.empty:
        raise ValueError(f"HC3 config {config} produced no answers.")
    return labeled


def prepare_sample(frame: pd.DataFrame, n_rows: int) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean with the training rules and draw a balanced sample of ``n_rows``."""
    if n_rows < 2:
        raise ValueError(f"--n-samples must be at least 2, got {n_rows}.")
    source_counts: dict[str, int] = {}
    if "source" in frame.columns:
        source_counts = {str(key): int(value) for key, value in frame["source"].value_counts().head(8).items()}
    cleaned = clean_dataframe(frame)
    sampled = subsample_balanced(cleaned, n_rows, seed=SEED)
    if len(sampled) < min(100, n_rows):
        logger.warning(
            "Cross-domain sample has only %d rows after cleaning (requested %d).",
            len(sampled),
            n_rows,
        )
    return sampled, source_counts


def score_labeled_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Score both saved models on a frame with ``text`` and ``label``."""
    labels = frame["label"].to_numpy()
    tfidf_score = score_tfidf(load_tfidf(), frame["text"])
    distilbert_score = score_distilbert(MODELS_DIR / DISTILBERT_DIRNAME, frame["text"])
    rows: list[dict[str, Any]] = []
    for score in (tfidf_score, distilbert_score):
        metrics = classification_scores(labels, score["predictions"], score["probabilities"])
        rows.append(
            {
                "model": score["name"],
                **metrics,
                "seconds_per_1000": float(score["seconds_per_1000"]),
                "device": score["device"],
            }
        )
    return rows


def load_in_domain_metrics() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use the saved test comparison, or score ``test.csv`` if that file is absent."""
    payload = None
    if COMPARISON_JSON.is_file():
        try:
            payload = json.loads(COMPARISON_JSON.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("%s is not valid JSON. Rescoring the test split.", COMPARISON_JSON.name)
            payload = None
    if isinstance(payload, dict):
        models = payload.get("models") or []
        names = {row.get("model") for row in models}
        needed = set(CLASSIFICATION_METRICS)
        complete = {"TF-IDF", "DistilBERT"} <= names and all(
            needed <= set(row) for row in models if row.get("model") in {"TF-IDF", "DistilBERT"}
        )
        if complete:
            logger.info("In-domain metrics loaded from %s.", _display_path(COMPARISON_JSON))
            return [_slim_metrics(row) for row in models], {
                "source": "outputs/metrics/comparison_summary.json",
                "n_rows": payload.get("n_test"),
            }
        logger.warning(
            "comparison_summary.json is missing a model or a classification metric. Rescoring the test split."
        )
    logger.info("Scoring the in-domain test split so the drop has a baseline.")
    test = load_test_set()
    return score_labeled_frame(test), {
        "source": "data/processed/test.csv",
        "n_rows": int(len(test)),
    }


def build_drop_table(in_domain: list[dict[str, Any]], cross_domain: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """In-domain minus cross-domain for each classification metric.

    A positive drop means the model scored worse on the new source.
    """
    in_by_model = {row["model"]: row for row in in_domain}
    cross_by_model = {row["model"]: row for row in cross_domain}
    missing = set(in_by_model) ^ set(cross_by_model)
    if missing or not in_by_model:
        raise RuntimeError(f"In-domain and cross-domain model names do not match: {sorted(missing)}.")
    rows: list[dict[str, Any]] = []
    for model in ("TF-IDF", "DistilBERT"):
        for metric in CLASSIFICATION_METRICS:
            before = float(in_by_model[model][metric])
            after = float(cross_by_model[model][metric])
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "in_domain": before,
                    "cross_domain": after,
                    "drop": before - after,
                }
            )
    return rows


def interpret_generalization(
    dataset: dict[str, Any],
    drop_rows: list[dict[str, Any]],
    cross_rows: list[dict[str, Any]],
) -> str:
    """Say which model holds up better out of domain, and a reason that fits the scores."""
    cross_f1 = {row["model"]: float(row["f1"]) for row in cross_rows}
    f1_drop = {
        row["model"]: float(row["drop"])
        for row in drop_rows
        if row["metric"] == "f1"
    }
    tfidf_f1 = cross_f1["TF-IDF"]
    bert_f1 = cross_f1["DistilBERT"]
    tfidf_drop = f1_drop["TF-IDF"]
    bert_drop = f1_drop["DistilBERT"]
    if abs(bert_f1 - tfidf_f1) < 0.01:
        leader = (
            f"On {dataset['id']}, the models are about equally accurate out of domain "
            f"(TF-IDF F1 {tfidf_f1:.3f}, DistilBERT F1 {bert_f1:.3f})."
        )
    elif bert_f1 > tfidf_f1:
        leader = (
            f"DistilBERT generalizes better here, with cross-domain F1 {bert_f1:.3f} "
            f"against {tfidf_f1:.3f} for TF-IDF."
        )
    else:
        leader = (
            f"TF-IDF generalizes better here, with cross-domain F1 {tfidf_f1:.3f} "
            f"against {bert_f1:.3f} for DistilBERT."
        )
    changes = (
        f"F1 drop from the in-domain test set: TF-IDF {_drop_phrase(tfidf_drop)}, "
        f"DistilBERT {_drop_phrase(bert_drop)}."
    )
    if dataset.get("fallback"):
        reason = (
            "The fallback set is scientific abstracts, half of them written by GPT-3, "
            "not the essays either model was fit on. A smaller drop means the model's "
            "cues were less tied to that training mix."
        )
    elif bert_f1 >= tfidf_f1:
        reason = (
            "HC3 is question-answer text, and the AI half is ChatGPT, so both the topic "
            "and the generator differ from training. DistilBERT started from a general "
            "English checkpoint, so more of its cues can survive that shift. TF-IDF can "
            "only reuse word weights learned on the training essays."
        )
    else:
        reason = (
            "HC3 changes both the topic and the generator. TF-IDF still separates the "
            "answers, which suggests it is using shallow cues such as function words and "
            "punctuation that ChatGPT still gives away. The fine-tune looks more tied "
            "to the training domain than those n-gram weights are."
        )
    return " ".join([dataset["description"], leader, changes, reason])


def plot_cross_domain(
    in_domain: list[dict[str, Any]],
    cross_domain: list[dict[str, Any]],
    dataset_name: str,
    path: Path,
) -> None:
    """Grouped bars of in-domain vs cross-domain scores, one panel per model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    in_by_model = {row["model"]: row for row in in_domain}
    cross_by_model = {row["model"]: row for row in cross_domain}
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), sharey=True)
    x = np.arange(len(CLASSIFICATION_METRICS))
    width = 0.36
    tick_labels = [METRIC_LABELS[metric] for metric in CLASSIFICATION_METRICS]
    for axis, model in zip(axes, ("TF-IDF", "DistilBERT"), strict=True):
        in_values = [float(in_by_model[model][metric]) for metric in CLASSIFICATION_METRICS]
        cross_values = [float(cross_by_model[model][metric]) for metric in CLASSIFICATION_METRICS]
        axis.bar(x - width / 2, in_values, width=width, label="In-domain", color=IN_DOMAIN_COLOR)
        axis.bar(x + width / 2, cross_values, width=width, label="Cross-domain", color=CROSS_DOMAIN_COLOR)
        axis.set_xticks(x, tick_labels)
        axis.set_ylim(0, 1.08)
        axis.set_title(model)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Score")
    fig.suptitle(f"In-domain test vs {dataset_name}")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", _display_path(path))


def main(argv: list[str] | None = None) -> None:
    """Download the out-of-domain set, score both models, and save the drop."""
    _configure_logging()
    args = _build_parser().parse_args(argv)
    set_seed(SEED)

    labeled, dataset = load_cross_domain_frame()
    sampled, source_counts = prepare_sample(labeled, args.n_samples)
    dataset["n_rows"] = int(len(sampled))
    dataset["n_human"] = int((sampled["label"] == 0).sum())
    dataset["n_ai"] = int((sampled["label"] == 1).sum())
    dataset["sources_before_cleaning"] = source_counts
    logger.info(
        "Cross-domain sample: %d rows | human=%d | ai=%d.",
        dataset["n_rows"],
        dataset["n_human"],
        dataset["n_ai"],
    )

    in_domain, in_meta = load_in_domain_metrics()
    logger.info("Scoring both models on the cross-domain sample.")
    cross_domain = score_labeled_frame(sampled.loc[:, ["text", "label"]])
    drop_rows = build_drop_table(in_domain, cross_domain)
    _log_drop_table(drop_rows)
    interpretation = interpret_generalization(dataset, drop_rows, cross_domain)

    metrics_path = METRICS_DIR / "cross_domain_metrics.json" if args.metrics_path is None else Path(args.metrics_path)
    figure_path = FIGURES_DIR / "cross_domain.png" if args.figure_path is None else Path(args.figure_path)
    payload = {
        "seed": SEED,
        "dataset": dataset,
        "in_domain": {**in_meta, "models": [_slim_metrics(row) for row in in_domain]},
        "cross_domain": {"n_rows": dataset["n_rows"], "models": cross_domain},
        "drop": drop_rows,
        "interpretation": interpretation,
    }
    _write_json(payload, metrics_path)
    plot_cross_domain(in_domain, cross_domain, dataset["id"], figure_path)
    print()
    print(interpretation)


def _try_hc3_config(config: str) -> tuple[pd.DataFrame | None, str]:
    dataset_error = _hc3_via_datasets(config)
    if isinstance(dataset_error, pd.DataFrame):
        return dataset_error, ""
    logger.info("HC3 config %s is not loading through datasets (%s). Trying the JSONL file.", config, dataset_error)
    try:
        path = _download_hc3_jsonl(config)
        frame = pd.read_json(path, lines=True)
        logger.info("Read %s (%d questions).", _display_path(path), len(frame))
        return frame, ""
    except Exception as exc:
        jsonl_error = f"{type(exc).__name__}: {exc}"
        return None, f"datasets: {dataset_error}; jsonl: {jsonl_error}"


def _hc3_via_datasets(config: str) -> pd.DataFrame | str:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        return f"ImportError: {exc}"
    try:
        try:
            dataset = load_dataset(HC3_REPO, config, split="train")
        except TypeError:
            dataset = load_dataset(HC3_REPO, config, split="train", trust_remote_code=True)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    if not hasattr(dataset, "to_pandas"):
        return "load_dataset did not return a tabular dataset."
    return dataset.to_pandas()


def _download_hc3_jsonl(config: str) -> Path:
    destination = RAW_DATA_DIR / f"hc3_{config}.jsonl"
    if destination.is_file() and destination.stat().st_size > 0:
        logger.info("Using cached %s.", _display_path(destination))
        return destination
    url = f"https://huggingface.co/datasets/{HC3_REPO}/resolve/main/{config}.jsonl"
    logger.info("Downloading %s", url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".jsonl.part")
    request = urllib.request.Request(url, headers={"User-Agent": "ai-text-detector"})
    try:
        with urllib.request.urlopen(request, timeout=300) as response, temporary.open("wb") as handle:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                handle.write(block)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    temporary.replace(destination)
    return destination


def _load_aiga_fallback() -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The HC3 fallback needs the datasets package. Install it with `pip install -r requirements.txt`."
        ) from exc
    dataset = load_dataset(FALLBACK_DATASET, split="train")
    frame = dataset.to_pandas()
    if "abstract" not in frame.columns or "label" not in frame.columns:
        raise ValueError(
            f"{FALLBACK_DATASET} did not have abstract and label columns. Found {list(frame.columns)}."
        )
    return pd.DataFrame({"text": frame["abstract"], "label": frame["label"]})


def _explode_answers(frame: pd.DataFrame, column: str, label: int) -> pd.DataFrame:
    piece = frame.loc[:, ["source", column]].explode(column)
    piece = piece.rename(columns={column: "text"})
    piece["label"] = label
    return piece


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    return [value]


def _clean_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def _slim_metrics(row: dict[str, Any]) -> dict[str, Any]:
    kept = ["model", *CLASSIFICATION_METRICS, "seconds_per_1000", "device"]
    return {key: row[key] for key in kept if key in row}


def _drop_phrase(drop: float) -> str:
    if drop > 0.005:
        return f"falls by {drop:.3f}"
    if drop < -0.005:
        return f"rises by {-drop:.3f}"
    return f"stays about the same ({drop:+.3f})"


def _log_drop_table(rows: list[dict[str, Any]]) -> None:
    logger.info("Performance drop (in-domain minus cross-domain):")
    for row in rows:
        logger.info(
            "  %-12s %-10s in=%.4f cross=%.4f drop=%+.4f",
            row["model"],
            row["metric"],
            row["in_domain"],
            row["cross_domain"],
            row["drop"],
        )


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s", _display_path(path))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare TF-IDF and DistilBERT on HC3, outside the training domain."
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=CROSS_DOMAIN_ROWS,
        help=f"Balanced cross-domain sample size. Defaults to {CROSS_DOMAIN_ROWS}.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=None,
        help="Defaults to outputs/metrics/cross_domain_metrics.json.",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=None,
        help="Defaults to outputs/figures/cross_domain.png.",
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
