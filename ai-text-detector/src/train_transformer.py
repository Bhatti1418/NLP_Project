"""Fine-tune distilbert-base-uncased for human vs AI text.

The checkpoint is the pretrained Hugging Face model named in ``src/config.py``.
Training uses the validation split to keep the epoch with the best F1. The test
split is scored only after that choice. ``--max_samples 500`` caps each split
for a CPU smoke test.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
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
from sklearn.metrics import confusion_matrix

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    FIGURES_DIR,
    LR,
    MAX_LEN,
    METRICS_DIR,
    MODEL_NAME,
    MODELS_DIR,
    PROJECT_ROOT,
    SEED,
)
from src.data_utils import set_seed, subsample_balanced
from src.train_baseline import load_splits

logger = logging.getLogger(__name__)

FINAL_DIR_NAME = "distilbert_detector"
CHECKPOINT_DIR_NAME = "distilbert_checkpoints"
POSITIVE_LABEL = 1


def _import_training_stack() -> None:
    """Import torch and Hugging Face only when a training run starts.

    ``--help`` stays usable on a machine that has not installed the GPU stack.
    """
    global AutoModelForSequenceClassification
    global AutoTokenizer
    global DataCollatorWithPadding
    global Dataset
    global EvalPrediction
    global Trainer
    global TrainingArguments
    global evaluate
    global set_hf_seed
    global torch
    try:
        import evaluate as evaluate_mod
        import torch as torch_mod
        from datasets import Dataset as dataset_cls
        from transformers import AutoModelForSequenceClassification as model_cls
        from transformers import AutoTokenizer as tokenizer_cls
        from transformers import DataCollatorWithPadding as collator_cls
        from transformers import EvalPrediction as eval_prediction_cls
        from transformers import Trainer as trainer_cls
        from transformers import TrainingArguments as training_arguments_cls
        from transformers import set_seed as hf_seed
    except ImportError as exc:
        raise ImportError(
            "DistilBERT training needs torch, transformers, datasets, and evaluate. "
            "Install them with `pip install -r requirements.txt`."
        ) from exc
    evaluate = evaluate_mod
    torch = torch_mod
    Dataset = dataset_cls
    AutoModelForSequenceClassification = model_cls
    AutoTokenizer = tokenizer_cls
    DataCollatorWithPadding = collator_cls
    EvalPrediction = eval_prediction_cls
    Trainer = trainer_cls
    TrainingArguments = training_arguments_cls
    set_hf_seed = hf_seed


def cap_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    max_samples: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified cap on each split.

    ``max_samples`` is the maximum number of rows kept in train, in val, and
    in test. ``None`` keeps every row.
    """
    if max_samples is None:
        return train, val, test
    if max_samples < 2:
        raise ValueError(f"--max_samples must be at least 2, got {max_samples}.")

    capped: list[pd.DataFrame] = []
    for name, frame in (("train", train), ("val", val), ("test", test)):
        if len(frame) <= max_samples:
            logger.info("%s split already has %d rows (cap is %d).", name, len(frame), max_samples)
            capped.append(frame)
            continue
        logger.info("Capping %s from %d rows to %d.", name, len(frame), max_samples)
        capped.append(subsample_balanced(frame, max_samples, seed=SEED))
    return capped[0], capped[1], capped[2]


def to_hf_dataset(frame: pd.DataFrame) -> Dataset:
    """Convert a split to a Hugging Face dataset with a ``labels`` column."""
    table = frame.loc[:, ["text", "label"]].reset_index(drop=True)
    dataset = Dataset.from_pandas(table, preserve_index=False)
    return dataset.rename_column("label", "labels")


def tokenize_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizerBase, split_name: str) -> Dataset:
    """Truncate to ``MAX_LEN`` without padding. The collator pads each batch."""

    def _tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

    tokenized = dataset.map(_tokenize, batched=True, desc=f"Tokenizing {split_name}")
    unused = [name for name in tokenized.column_names if name not in {"input_ids", "attention_mask", "labels"}]
    if unused:
        tokenized = tokenized.remove_columns(unused)
    return tokenized


def build_compute_metrics() -> Any:
    """Load ``evaluate`` metrics once and return the Trainer callback.

    Precision, recall, and F1 treat AI (label 1) as the positive class, matching
    the TF-IDF baseline.
    """
    logger.info("Loading evaluate metrics: accuracy, precision, recall, f1.")
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.argmax(np.asarray(logits), axis=-1)
        references = np.asarray(labels)
        scores = {
            "accuracy": accuracy.compute(predictions=predictions, references=references)["accuracy"],
            "precision": precision.compute(
                predictions=predictions,
                references=references,
                average="binary",
                pos_label=POSITIVE_LABEL,
            )["precision"],
            "recall": recall.compute(
                predictions=predictions,
                references=references,
                average="binary",
                pos_label=POSITIVE_LABEL,
            )["recall"],
            "f1": f1.compute(
                predictions=predictions,
                references=references,
                average="binary",
                pos_label=POSITIVE_LABEL,
            )["f1"],
        }
        return {key: float(value) for key, value in scores.items()}

    return compute_metrics


def build_trainer(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    use_fp16: bool,
) -> Trainer:
    """Trainer that evaluates every epoch and reloads the best F1 checkpoint."""
    checkpoint_dir = MODELS_DIR / CHECKPOINT_DIR_NAME
    if checkpoint_dir.exists():
        logger.info("Removing previous checkpoints in %s.", _display_path(checkpoint_dir))
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    arguments = TrainingArguments(
        output_dir=str(checkpoint_dir),
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        fp16=use_fp16,
        seed=SEED,
        data_seed=SEED,
        report_to="none",
        dataloader_pin_memory=use_fp16,
    )
    return Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=build_compute_metrics(),
    )


def plot_loss_curve(log_history: list[dict[str, Any]], path: Path) -> None:
    """Save train and validation loss against epoch."""
    train_epochs: list[float] = []
    train_loss: list[float] = []
    eval_epochs: list[float] = []
    eval_loss: list[float] = []
    for event in log_history:
        epoch = event.get("epoch")
        if epoch is None:
            continue
        if "loss" in event and "eval_loss" not in event:
            train_epochs.append(float(epoch))
            train_loss.append(float(event["loss"]))
        if "eval_loss" in event:
            eval_epochs.append(float(epoch))
            eval_loss.append(float(event["eval_loss"]))
    if not train_loss and not eval_loss:
        logger.warning("Trainer log has no loss entries. Skipping %s.", path.name)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    if train_loss:
        ax.plot(train_epochs, train_loss, marker="o", color="#4C78A8", label="Train")
    if eval_loss:
        ax.plot(eval_epochs, eval_loss, marker="o", color="#F58518", label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("DistilBERT training and validation loss")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", _display_path(path))


def plot_confusion_matrix(matrix: np.ndarray, path: Path) -> None:
    """Save the test-set confusion matrix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Human", "AI"],
        yticklabels=["Human", "AI"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("DistilBERT test confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", _display_path(path))


def save_predictions(frame: pd.DataFrame, probabilities: np.ndarray, path: Path) -> None:
    """Write test text, gold label, prediction, and class probabilities."""
    predicted = np.argmax(probabilities, axis=1).astype(int)
    table = pd.DataFrame(
        {
            "text": frame["text"].to_numpy(),
            "label": frame["label"].astype(int).to_numpy(),
            "prediction": predicted,
            "probability_human": probabilities[:, 0],
            "probability_ai": probabilities[:, 1],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    logger.info("Wrote %s (%d rows).", _display_path(path), len(table))


def main(argv: list[str] | None = None) -> None:
    """Fine-tune DistilBERT and write the model, metrics, figures, and predictions."""
    _configure_logging()
    args = _build_parser().parse_args(argv)
    _import_training_stack()
    set_seed(SEED)
    set_hf_seed(SEED)

    use_fp16 = torch.cuda.is_available()
    if use_fp16:
        logger.info("CUDA device: %s. Mixed precision (fp16) is on.", torch.cuda.get_device_name(0))
    else:
        logger.info("CUDA is not available. Training on CPU with fp16 off.")

    train_frame, val_frame, test_frame = load_splits()
    train_frame, val_frame, test_frame = cap_splits(train_frame, val_frame, test_frame, args.max_samples)
    logger.info(
        "Split sizes: train=%d val=%d test=%d.",
        len(train_frame),
        len(val_frame),
        len(test_frame),
    )

    logger.info("Loading pretrained tokenizer and model: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "human", 1: "ai"},
        label2id={"human": 0, "ai": 1},
    )

    train_dataset = tokenize_dataset(to_hf_dataset(train_frame), tokenizer, "train")
    val_dataset = tokenize_dataset(to_hf_dataset(val_frame), tokenizer, "val")

    trainer = build_trainer(model, tokenizer, train_dataset, val_dataset, use_fp16)
    logger.info(
        "Training for %s epoch(s), batch size %d, learning rate %s, max length %d.",
        EPOCHS,
        BATCH_SIZE,
        LR,
        MAX_LEN,
    )
    started = time.perf_counter()
    trainer.train()
    logger.info("Training finished in %.1f minutes.", (time.perf_counter() - started) / 60.0)
    best_f1 = trainer.state.best_metric
    if best_f1 is not None:
        logger.info("Best validation F1: %.4f (%s).", float(best_f1), trainer.state.best_model_checkpoint)

    logger.info("Evaluating the reloaded best model on the test set.")
    test_dataset = tokenize_dataset(to_hf_dataset(test_frame), tokenizer, "test")
    prediction = trainer.predict(test_dataset)
    logits = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
    probabilities = _softmax(np.asarray(logits))
    predicted = np.argmax(probabilities, axis=1)
    matrix = confusion_matrix(test_frame["label"], predicted, labels=[0, 1])
    test_metrics = _test_metrics(prediction.metrics)
    test_metrics["confusion_matrix"] = matrix.astype(int).tolist()
    test_metrics["labels"] = ["human", "ai"]
    logger.info(
        "Test accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
        test_metrics["accuracy"],
        test_metrics["precision"],
        test_metrics["recall"],
        test_metrics["f1"],
    )

    final_dir = MODELS_DIR / FINAL_DIR_NAME
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    _write_json(
        {
            "seed": SEED,
            "selection_metric": "val_f1",
            "max_samples": args.max_samples,
            "n_train": int(len(train_frame)),
            "n_val": int(len(val_frame)),
            "n_test": int(len(test_frame)),
        },
        final_dir / "training_meta.json",
    )
    if args.max_samples is not None:
        logger.warning(
            "Saved a smoke-test checkpoint (--max_samples %d). "
            "Re-run without --max_samples before `python -m src.evaluate`.",
            args.max_samples,
        )
    logger.info("Wrote model and tokenizer to %s", _display_path(final_dir))

    metrics_path = METRICS_DIR / "transformer_metrics.json"
    figure_cm = FIGURES_DIR / "cm_transformer.png"
    figure_loss = FIGURES_DIR / "loss_curve.png"
    predictions_path = METRICS_DIR / "transformer_test_predictions.csv"
    payload = {
        "seed": SEED,
        "model_name": MODEL_NAME,
        "max_length": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "fp16": use_fp16,
        "max_samples": args.max_samples,
        "positive_label": POSITIVE_LABEL,
        "positive_class": "AI",
        "selection_metric": "val_f1",
        "n_train": int(len(train_frame)),
        "n_val": int(len(val_frame)),
        "n_test": int(len(test_frame)),
        "best_val_f1": None if best_f1 is None else float(best_f1),
        "best_checkpoint": None
        if trainer.state.best_model_checkpoint is None
        else _display_path(Path(trainer.state.best_model_checkpoint)),
        "test": test_metrics,
        "history": _compact_history(trainer.state.log_history),
    }
    _write_json(payload, metrics_path)
    plot_confusion_matrix(matrix, figure_cm)
    plot_loss_curve(trainer.state.log_history, figure_loss)
    save_predictions(test_frame, probabilities, predictions_path)


def _test_metrics(raw: dict[str, Any] | None) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key, value in (raw or {}).items():
        name = key.removeprefix("test_")
        if name in {"accuracy", "precision", "recall", "f1", "loss"}:
            metrics[name] = float(value)
    missing = {"accuracy", "precision", "recall", "f1"} - metrics.keys()
    if missing:
        raise RuntimeError(f"Test evaluation did not return {sorted(missing)}.")
    return metrics


def _compact_history(log_history: list[dict[str, Any]]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    keys = ("epoch", "loss", "eval_loss", "eval_accuracy", "eval_precision", "eval_recall", "eval_f1")
    for event in log_history:
        row = {key: float(event[key]) for key in keys if key in event}
        if row:
            rows.append(row)
    return rows


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s", _display_path(path))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune distilbert-base-uncased on the processed AI vs human splits."
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help=(
            "Maximum rows to keep in each of train, val, and test (stratified, seed 42). "
            "Use 500 for a CPU smoke test. The default uses every row."
        ),
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
