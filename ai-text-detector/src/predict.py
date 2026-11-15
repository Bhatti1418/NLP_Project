"""Run a saved detector on one text.

Models are loaded on first use and kept in memory. ``predict`` returns the
class name, the probability of that class, and both class probabilities.
A probability of 0.5 or higher for AI is read as AI-generated, matching the
training and evaluation scripts.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import threading
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from src.config import MAX_LEN, MODELS_DIR, SEED
from src.data_utils import set_seed
from src.evaluate import (
    DISTILBERT_DIRNAME,
    _import_torch_stack,
    _positive_label_index,
    load_tfidf,
    warn_if_partial_finetune,
)
from src.train_baseline import _decision_weights

logger = logging.getLogger(__name__)

RELIABLE_WORD_COUNT = 50
TOP_CONTRIBUTIONS = 10
HUMAN_LABEL = "Human"
AI_LABEL = "AI-generated"
_AI_FEATURE = "AI"
_HUMAN_FEATURE = "Human"

_MODEL_ALIASES = {
    "distilbert": "distilbert",
    "tfidf": "tfidf",
    "tf-idf": "tfidf",
}

_CACHE: dict[str, Any] = {}
_LOCK = threading.Lock()


class Prediction(TypedDict):
    """One text scored by one model."""

    label: str
    label_id: int
    confidence: float
    probabilities: dict[str, float]


def clear_model_cache() -> None:
    """Drop cached models so the next call loads them again."""
    with _LOCK:
        _CACHE.clear()


def word_count(text: str) -> int:
    """Count whitespace-separated words, the same split used when cleaning data."""
    return len(str(text).split())


def short_text_warning(text: str) -> str:
    """Warn when a text is too short for a stable prediction. Empty when it is long enough."""
    count = word_count(text)
    if count >= RELIABLE_WORD_COUNT:
        return ""
    return (
        f"This text has {count} words. "
        f"Predictions on texts shorter than {RELIABLE_WORD_COUNT} words are unreliable."
    )


def normalize_model_name(model: str) -> str:
    """Return ``distilbert`` or ``tfidf``."""
    key = str(model).strip().lower()
    if key not in _MODEL_ALIASES:
        raise ValueError("model must be 'distilbert' or 'tfidf'.")
    return _MODEL_ALIASES[key]


def predict(text: str, model: str = "distilbert") -> Prediction:
    """Classify ``text`` with the cached TF-IDF pipeline or DistilBERT checkpoint.

    Parameters
    ----------
    text:
        The passage to score.
    model:
        ``distilbert`` or ``tfidf``.

    Returns
    -------
    Prediction
        ``label`` is ``Human`` or ``AI-generated``. ``confidence`` is the
        probability of that label. ``probabilities`` has keys ``human`` and ``ai``.
    """
    if text is None or not str(text).strip():
        raise ValueError("Enter some text to classify.")
    choice = normalize_model_name(model)
    if choice == "tfidf":
        return _predict_tfidf(str(text))
    return _predict_distilbert(str(text))


def top_contributions(text: str, k: int = TOP_CONTRIBUTIONS) -> list[dict[str, float | str]]:
    """Highest-magnitude TF-IDF n-grams in ``text``.

    Contribution is the token's TF-IDF value times its linear weight.
    A positive weight points toward AI, the same sign used at training time.
    """
    if text is None or not str(text).strip():
        raise ValueError("Enter some text to classify.")
    if k < 1:
        raise ValueError("k must be at least 1.")
    pipeline = get_tfidf_pipeline()
    weights = _CACHE["tfidf_weights"]
    vectorizer = pipeline.named_steps["tfidf"]
    names = np.asarray(vectorizer.get_feature_names_out())
    if names.shape[0] != np.asarray(weights).shape[0]:
        raise RuntimeError(
            f"Feature count ({names.shape[0]}) does not match weight count ({len(weights)})."
        )
    row = vectorizer.transform([str(text)]).tocsr()
    if row.nnz == 0:
        return []
    contributions = row.data * np.asarray(weights, dtype=float)[row.indices]
    order = np.argsort(-np.abs(contributions))
    rows: list[dict[str, float | str]] = []
    for position in order:
        value = float(contributions[position])
        if value == 0.0:
            continue
        rows.append(
            {
                "feature": str(names[int(row.indices[position])]),
                "contribution": value,
                "direction": _AI_FEATURE if value > 0 else _HUMAN_FEATURE,
            }
        )
        if len(rows) >= k:
            break
    return rows


def tfidf_highlights(text: str, k: int = TOP_CONTRIBUTIONS) -> list[tuple[str, str | None]]:
    """Split ``text`` into spans tagged Human, AI, or None for the untagged pieces.

    Tags follow ``top_contributions``. A stronger n-gram keeps the span when two
    matches overlap.
    """
    raw = str(text)
    contributions = top_contributions(raw, k=k)
    if not contributions:
        return [(raw, None)]
    candidates: list[tuple[int, int, str, float]] = []
    for row in contributions:
        feature = str(row["feature"])
        direction = str(row["direction"])
        strength = abs(float(row["contribution"]))
        pattern = _feature_pattern(feature)
        for match in pattern.finditer(raw):
            if match.start() == match.end():
                continue
            candidates.append((match.start(), match.end(), direction, strength))
    candidates.sort(key=lambda item: (-item[3], -(item[1] - item[0]), item[0]))
    chosen: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []
    for start, end, direction, _strength in candidates:
        if any(start < right and end > left for left, right in occupied):
            continue
        occupied.append((start, end))
        chosen.append((start, end, direction))
    chosen.sort(key=lambda item: item[0])
    pieces: list[tuple[str, str | None]] = []
    cursor = 0
    for start, end, direction in chosen:
        if start > cursor:
            pieces.append((raw[cursor:start], None))
        pieces.append((raw[start:end], direction))
        cursor = end
    if cursor < len(raw):
        pieces.append((raw[cursor:], None))
    return pieces or [(raw, None)]


def get_tfidf_pipeline() -> Any:
    """Load the TF-IDF pipeline once and cache its class weights."""
    with _LOCK:
        if "tfidf" not in _CACHE:
            pipeline = load_tfidf()
            _CACHE["tfidf"] = pipeline
            _CACHE["tfidf_weights"] = _decision_weights(pipeline)
        return _CACHE["tfidf"]


def get_distilbert() -> dict[str, Any]:
    """Load the fine-tuned DistilBERT checkpoint once."""
    with _LOCK:
        if "distilbert" not in _CACHE:
            _CACHE["distilbert"] = _load_distilbert()
        return _CACHE["distilbert"]


def main(argv: list[str] | None = None) -> None:
    """Score one passage from the command line."""
    _configure_logging()
    args = _build_parser().parse_args(argv)
    set_seed(SEED)
    text = _read_cli_text(args.text, args.text_file)
    result = predict(text, args.model)
    print(json.dumps(result, indent=2))
    warning = short_text_warning(text)
    if warning:
        print(warning)
    if normalize_model_name(args.model) == "tfidf":
        rows = top_contributions(text)
        if rows:
            print("Top contributing n-grams:")
            for row in rows:
                print(f"  {row['direction']:<6} {float(row['contribution']):+.4f}  {row['feature']}")


def _predict_tfidf(text: str) -> Prediction:
    pipeline = get_tfidf_pipeline()
    probabilities = np.asarray(pipeline.predict_proba([text]), dtype=float)
    classes = [int(label) for label in pipeline.named_steps["classifier"].classes_]
    if probabilities.shape != (1, len(classes)):
        raise RuntimeError(f"Expected one probability row, got shape {probabilities.shape}.")
    if 0 not in classes or 1 not in classes:
        raise RuntimeError(f"TF-IDF classes are {classes}; expected labels 0 and 1.")
    human = float(np.clip(probabilities[0, classes.index(0)], 0.0, 1.0))
    ai = float(np.clip(probabilities[0, classes.index(1)], 0.0, 1.0))
    return _pack_prediction(human, ai)


def _predict_distilbert(text: str) -> Prediction:
    bundle = get_distilbert()
    torch = bundle["torch"]
    device = bundle["device"]
    with _LOCK:
        encoded = bundle["tokenizer"](
            [text],
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = bundle["model"](**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
    if int(probabilities.shape[0]) != 2:
        raise RuntimeError(f"Expected 2 class probabilities, got shape {probabilities.shape}.")
    ai = float(np.clip(probabilities[bundle["positive_index"]], 0.0, 1.0))
    human = float(np.clip(probabilities[1 - bundle["positive_index"]], 0.0, 1.0))
    return _pack_prediction(human, ai)


def _pack_prediction(human: float, ai: float) -> Prediction:
    if ai >= 0.5:
        label = AI_LABEL
        label_id = 1
        confidence = ai
    else:
        label = HUMAN_LABEL
        label_id = 0
        confidence = human
    return {
        "label": label,
        "label_id": label_id,
        "confidence": float(confidence),
        "probabilities": {"human": float(human), "ai": float(ai)},
    }


def _load_distilbert() -> dict[str, Any]:
    model_dir = MODELS_DIR / DISTILBERT_DIRNAME
    if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
        raise FileNotFoundError(
            f"DistilBERT model not found at {_display_path(model_dir)}. "
            "Run `python -m src.train_transformer` first."
        )
    warn_if_partial_finetune(model_dir)
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
    positive_index = _positive_label_index(model.config.id2label)
    if positive_index not in (0, 1):
        raise RuntimeError(f"AI class index must be 0 or 1, found {positive_index}.")
    logger.info("DistilBERT is cached on %s.", device.type)
    return {
        "torch": torch,
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "positive_index": positive_index,
    }


def _feature_pattern(feature: str) -> re.Pattern[str]:
    parts = [re.escape(part) for part in feature.split()]
    body = r"\b" + r"\W+".join(parts) + r"\b"
    return re.compile(body, flags=re.IGNORECASE)


def _read_cli_text(text: str | None, text_file: Path | None) -> str:
    if text and text_file:
        raise ValueError("Pass --text or --text-file, not both.")
    if text_file is not None:
        return Path(text_file).read_text(encoding="utf-8")
    if text:
        return text
    raise ValueError("Pass --text or --text-file.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify one text with a saved detector.")
    parser.add_argument("--text", default=None, help="The passage to classify.")
    parser.add_argument("--text-file", type=Path, default=None, help="UTF-8 file containing the passage.")
    parser.add_argument(
        "--model",
        choices=("distilbert", "tfidf"),
        default="distilbert",
        help="Which cached model to run. Defaults to distilbert.",
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
        from src.config import PROJECT_ROOT

        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    main()
