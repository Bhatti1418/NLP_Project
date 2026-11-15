"""Load, clean, and split the AI vs Human text corpus.

The command-line entry point runs the full pipeline: read
``data/raw/AI_Human.csv``, clean it, draw a balanced sample, and write
stratified ``train.csv``, ``val.csv``, and ``test.csv`` under ``data/processed``.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    MAX_TRAIN_SAMPLES,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    SEED,
    TEST_SIZE,
    VAL_SIZE,
)

logger = logging.getLogger(__name__)

MIN_WORD_COUNT: int = 20
RAW_FILENAME: str = "AI_Human.csv"

_TEXT_CANDIDATES: tuple[str, ...] = (
    "text",
    "essay",
    "content",
    "document",
    "sentence",
    "paragraph",
)
_LABEL_CANDIDATES: tuple[str, ...] = (
    "generated",
    "label",
    "class",
    "target",
    "ai_generated",
    "is_ai",
)
_LABEL_MAP: dict[str, int] = {
    "0": 0,
    "0.0": 0,
    "human": 0,
    "false": 0,
    "real": 0,
    "no": 0,
    "1": 1,
    "1.0": 1,
    "ai": 1,
    "generated": 1,
    "true": 1,
    "machine": 1,
    "yes": 1,
}


def set_seed(seed: int = SEED) -> None:
    """Seed the Python and NumPy RNGs.

    Splitters and samplers also receive ``seed`` directly, so results stay
    reproducible even if another library reseeds the global RNGs later.
    """
    random.seed(seed)
    np.random.seed(seed)


def load_raw_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the raw Kaggle CSV.

    Parameters
    ----------
    path:
        CSV location. Defaults to ``data/raw/AI_Human.csv``.

    Returns
    -------
    pd.DataFrame
        Table as stored on disk, with the file's original column names.

    Raises
    ------
    FileNotFoundError
        If the CSV has not been placed at the expected path.
    """
    csv_path = RAW_DATA_DIR / RAW_FILENAME if path is None else Path(path)
    if not csv_path.is_file():
        others = sorted(item.name for item in csv_path.parent.glob("*.csv")) if csv_path.parent.is_dir() else []
        extra = f" CSV files already in that folder: {others}." if others else ""
        raise FileNotFoundError(
            f"Raw dataset not found at {_display_path(csv_path)}. "
            "Download the Kaggle dataset shanegerami/ai-vs-human-text and place "
            f"the CSV at data/raw/{RAW_FILENAME}.{extra}"
        )

    logger.info("Loading %s", _display_path(csv_path))
    try:
        frame = pd.read_csv(csv_path, low_memory=False, encoding="utf-8-sig")
    except UnicodeDecodeError:
        logger.warning("UTF-8 decode failed for %s; retrying with latin-1.", csv_path.name)
        frame = pd.read_csv(csv_path, low_memory=False, encoding="latin-1")
    logger.info("Loaded %d rows. Columns: %s", len(frame), list(frame.columns))
    return frame


def clean_dataframe(df: pd.DataFrame, min_words: int = MIN_WORD_COUNT) -> pd.DataFrame:
    """Clean text and labels for classification.

    Rows with null text or labels are dropped, whitespace is stripped, duplicate
    texts are removed, and texts shorter than ``min_words`` are discarded. The
    label column is renamed to ``label`` (0 = human, 1 = AI). Stripping happens
    before duplicate removal so the same essay with extra spaces counts once.

    Parameters
    ----------
    df:
        Raw dataframe from :func:`load_raw_dataset`.
    min_words:
        Minimum number of whitespace-separated words to keep a text.

    Returns
    -------
    pd.DataFrame
        Columns ``text`` and ``label`` only.

    Raises
    ------
    ValueError
        If the text or label column cannot be identified, or if fewer than
        two classes remain after cleaning.
    """
    if min_words < 1:
        raise ValueError(f"min_words must be at least 1, got {min_words}.")

    selected = _select_text_and_label(df)
    n_conflicts = int((selected.groupby("text")["label"].nunique() > 1).sum())
    if n_conflicts:
        logger.warning(
            "%d distinct texts have both labels. Keeping the first copy of each.",
            n_conflicts,
        )

    deduped = selected.drop_duplicates(subset=["text"], keep="first")
    n_duplicates = len(selected) - len(deduped)
    word_counts = deduped["text"].str.split().str.len()
    cleaned = deduped.loc[word_counts >= min_words].reset_index(drop=True)
    n_short = len(deduped) - len(cleaned)
    logger.info(
        "Removed %d duplicate texts and %d texts shorter than %d words. %d rows remain.",
        n_duplicates,
        n_short,
        min_words,
        len(cleaned),
    )
    if cleaned.empty or cleaned["label"].nunique() < 2:
        raise ValueError(
            "Cleaning left fewer than two classes. Check the label values and the "
            f"minimum length of {min_words} words."
        )
    return cleaned


def subsample_balanced(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """Draw a class-balanced sample of at most ``n`` rows.

    Parameters
    ----------
    df:
        Cleaned dataframe with a ``label`` column of 0s and 1s.
    n:
        Target number of rows, split evenly across the two classes. An odd
        ``n`` is rounded down so the classes stay equal. If either class is
        smaller than ``n // 2``, both classes are limited to the smaller one.
    seed:
        Sample seed. Defaults to the project seed.

    Returns
    -------
    pd.DataFrame
        Shuffled balanced sample with the same columns as ``df``.
    """
    if n < 2:
        raise ValueError(f"n must be at least 2 so both classes can be sampled, got {n}.")
    if "label" not in df.columns:
        raise ValueError("subsample_balanced expects a 'label' column. Run clean_dataframe first.")

    unexpected = sorted(set(df["label"].unique()) - {0, 1})
    if unexpected:
        raise ValueError(f"Expected labels 0 (human) and 1 (AI); found {unexpected}.")
    if df["label"].nunique() < 2:
        raise ValueError("subsample_balanced needs both classes.")

    per_class = n // 2
    if n % 2 == 1:
        logger.info("n=%d is odd; using %d rows so the two classes stay equal.", n, per_class * 2)

    minority = int(df["label"].value_counts().min())
    if minority < per_class:
        logger.warning(
            "Requested %d rows per class but the smaller class has %d. Using %d per class.",
            per_class,
            minority,
            minority,
        )
        per_class = minority
    if per_class < 1:
        raise ValueError("Not enough rows to draw one sample from each class.")

    parts = [
        df.loc[df["label"] == label].sample(n=per_class, random_state=seed)
        for label in (0, 1)
    ]
    sampled = pd.concat(parts, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    logger.info("Balanced subsample: %d rows | %s", len(sampled), _format_balance(sampled))
    return sampled


def make_splits(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Write a stratified train, validation, and test split.

    ``test_size`` and ``val_size`` are fractions of the full dataframe, so the
    default config values keep 70% train, 15% validation, and 15% test.

    Parameters
    ----------
    df:
        Cleaned (and usually subsampled) dataframe with ``text`` and ``label``.
    output_dir:
        Directory for ``train.csv``, ``val.csv``, and ``test.csv``. Defaults
        to ``data/processed``.
    test_size:
        Fraction of rows assigned to the test split.
    val_size:
        Fraction of rows assigned to the validation split.
    seed:
        Stratification seed.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Train, validation, and test dataframes in that order.
    """
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("make_splits expects columns 'text' and 'label'.")
    if df["label"].nunique() < 2:
        raise ValueError("make_splits needs both classes so the splits can be stratified.")
    if test_size <= 0 or val_size <= 0 or test_size + val_size >= 1:
        raise ValueError(
            f"TEST_SIZE ({test_size}) and VAL_SIZE ({val_size}) must both be > 0 and sum to < 1."
        )

    destination = PROCESSED_DATA_DIR if output_dir is None else Path(output_dir)
    relative_val = val_size / (1.0 - test_size)
    working = df.loc[:, ["text", "label"]].copy().reset_index(drop=True)
    working["text"] = working["text"].astype(str).str.strip()
    duplicated = int(working["text"].duplicated().sum())
    if duplicated:
        raise ValueError(
            f"make_splits received {duplicated} duplicate texts. "
            "Run clean_dataframe first so the same essay cannot fall into two splits."
        )
    train_val, test = train_test_split(
        working,
        test_size=test_size,
        random_state=seed,
        stratify=working["label"],
    )
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        random_state=seed,
        stratify=train_val["label"],
    )

    destination.mkdir(parents=True, exist_ok=True)
    written: dict[str, pd.DataFrame] = {}
    for name, split in (("train", train), ("val", val), ("test", test)):
        split = split.reset_index(drop=True)
        csv_path = destination / f"{name}.csv"
        split.to_csv(csv_path, index=False)
        written[name] = split
        logger.info("Wrote %s (%d rows).", _display_path(csv_path), len(split))
    assert_no_text_leakage(written["train"], written["val"], written["test"])
    return written["train"], written["val"], written["test"]


def assert_no_text_leakage(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Raise if the same stripped text appears in more than one split."""
    sets = {
        "train": set(train["text"].astype(str).str.strip()),
        "val": set(val["text"].astype(str).str.strip()),
        "test": set(test["text"].astype(str).str.strip()),
    }
    leaked = {
        "train/val": len(sets["train"] & sets["val"]),
        "train/test": len(sets["train"] & sets["test"]),
        "val/test": len(sets["val"] & sets["test"]),
    }
    found = {name: count for name, count in leaked.items() if count}
    if found:
        detail = ", ".join(f"{name}={count}" for name, count in found.items())
        raise ValueError(
            "The same text appears in more than one split "
            f"({detail}). Rebuild the splits with `python -m src.data_utils`."
        )


def main(argv: list[str] | None = None) -> None:
    """Run cleaning, balanced subsampling, and the stratified split."""
    _configure_logging()
    args = _build_parser().parse_args(argv)
    set_seed(SEED)

    n_samples = args.n_samples
    if n_samples < 2:
        raise ValueError(f"--n-samples must be at least 2 so both classes can be drawn, got {n_samples}.")
    if n_samples > MAX_TRAIN_SAMPLES:
        logger.warning(
            "Requested %d rows, above MAX_TRAIN_SAMPLES=%d. Capping at %d.",
            n_samples,
            MAX_TRAIN_SAMPLES,
            MAX_TRAIN_SAMPLES,
        )
        n_samples = MAX_TRAIN_SAMPLES

    raw = load_raw_dataset(args.input)
    cleaned = clean_dataframe(raw, min_words=args.min_words)
    logger.info("Cleaned corpus: %d rows | %s", len(cleaned), _format_balance(cleaned))
    sampled = subsample_balanced(cleaned, n_samples, seed=SEED)
    train, val, test = make_splits(sampled)

    logger.info("Split sizes and class balance (0 = human, 1 = AI):")
    for name, split in (("train", train), ("val", val), ("test", test)):
        logger.info("  %s: %d rows | %s", name, len(split), _format_balance(split))
    if len(train) > MAX_TRAIN_SAMPLES:
        logger.warning(
            "Training split has %d rows, above the %d-row cap.",
            len(train),
            MAX_TRAIN_SAMPLES,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Clean the AI vs Human CSV, draw a balanced sample, and write "
            "stratified train/val/test splits."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=f"Raw CSV path. Defaults to data/raw/{RAW_FILENAME}.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=MAX_TRAIN_SAMPLES,
        help=(
            "Balanced sample size before splitting. "
            f"Defaults to MAX_TRAIN_SAMPLES ({MAX_TRAIN_SAMPLES})."
        ),
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=MIN_WORD_COUNT,
        help=f"Drop texts shorter than this many words. Defaults to {MIN_WORD_COUNT}.",
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


def _match_column(columns: list[str], candidates: tuple[str, ...], role: str) -> str:
    lookup = {column.strip().lower(): column for column in columns}
    for name in candidates:
        if name in lookup:
            return lookup[name]
    raise ValueError(
        f"Could not find a {role} column. Expected one of {list(candidates)}. "
        f"Found columns: {columns}."
    )


def _normalize_labels(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    labels = pd.Series(pd.NA, index=series.index, dtype="Int64")
    parsed = numeric.notna()
    labels.loc[parsed] = numeric.loc[parsed].round().astype(int)

    unparsed = ~parsed & series.notna()
    if unparsed.any():
        mapped = series.loc[unparsed].astype(str).str.strip().str.lower().map(_LABEL_MAP)
        labels.loc[unparsed] = mapped

    unknown_mask = labels.isna() & series.notna()
    if unknown_mask.any():
        unknown = sorted(set(series.loc[unknown_mask].astype(str)))
        raise ValueError(
            f"Unrecognised label values: {unknown}. Expected 0/1 or human/AI."
        )

    present = set(labels.dropna().astype(int).unique())
    unexpected = sorted(present - {0, 1})
    if unexpected:
        raise ValueError(f"Expected labels 0 (human) and 1 (AI); found {unexpected}.")
    return labels


def _select_text_and_label(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("The dataset is empty.")

    columns = [str(column) for column in df.columns]
    text_col = _match_column(columns, _TEXT_CANDIDATES, "text")
    label_col = _match_column(columns, _LABEL_CANDIDATES, "label")
    if text_col == label_col:
        raise ValueError(f"Text and label resolved to the same column {text_col!r}.")
    logger.info("Using text column %r and label column %r. Label column will be stored as 'label'.", text_col, label_col)

    selected = pd.DataFrame(
        {
            "text": df[text_col],
            "label": _normalize_labels(df[label_col]),
        }
    )
    before = len(selected)
    selected = selected.dropna(subset=["text", "label"]).copy()
    selected["text"] = selected["text"].astype(str).str.strip()
    selected = selected.loc[selected["text"].ne("")].copy()
    selected["label"] = selected["label"].astype(int)
    selected = selected.reset_index(drop=True)
    logger.info("Dropped %d rows with null or blank text/label.", before - len(selected))
    return selected


def _format_balance(df: pd.DataFrame) -> str:
    counts = df["label"].value_counts()
    total = int(len(df))
    parts: list[str] = []
    for label, name in ((0, "human"), (1, "ai")):
        count = int(counts.get(label, 0))
        share = 100.0 * count / total if total else 0.0
        parts.append(f"{name}={count} ({share:.1f}%)")
    return ", ".join(parts)


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    main()
