"""Gradio demo for the AI-text detector.

Run from the project directory:

    python app.py
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import gradio as gr
import pandas as pd

from src.config import FIGURES_DIR, METRICS_DIR
from src.predict import (
    AI_LABEL,
    HUMAN_LABEL,
    predict,
    short_text_warning,
    tfidf_highlights,
    word_count,
)

logger = logging.getLogger(__name__)

DISCLAIMER = "AI-text detectors are imperfect and should not be used as sole evidence of misconduct."

HUMAN_EXAMPLE_1 = (
    "I missed the 8:15 bus again because my little brother hid one of my shoes behind the couch. "
    "By the time I found it, the sky had opened and I was halfway to the stop with a grocery bag "
    "tearing at the corner. The driver still waited, which surprised me. I spent the ride wiping "
    "rain off my notes for history and trying to remember whether the quiz was on Tuesday or "
    "Wednesday. None of this is profound. It is just the kind of morning that makes the rest of "
    "the day feel slightly behind, and I still had a damp sleeve when I finally sat down."
)
HUMAN_EXAMPLE_2 = (
    "Dad let me stir the onions even though I kept knocking the spoon against the pot. The kitchen "
    "smelled like garlic and the window fogged up, so I drew a lopsided cat in the steam and he "
    "pretended not to see it. We argued about whether the rice needed more salt. He was right, "
    "but I only admitted it after I tasted a second spoonful. Later the neighbor's dog barked "
    "through the whole meal and nobody got up to close the door because the food was too hot "
    "and nobody wanted to lose their seat."
)
AI_EXAMPLE_1 = (
    "Education plays a central role in fostering individual development and broader social progress. "
    "Furthermore, access to structured learning opportunities enables students to acquire analytical "
    "skills and adapt to changing demands. Moreover, collaborative environments facilitate the exchange "
    "of diverse perspectives, thereby strengthening problem-solving capacity. Therefore, institutions "
    "should adopt evidence-based methodologies and allocate resources in a transparent manner. "
    "In conclusion, a comprehensive framework that joins academic rigor with practical application "
    "can contribute to sustained intellectual and economic advancement."
)
AI_EXAMPLE_2 = (
    "Climate adaptation requires a coordinated strategy across infrastructure, policy, and community "
    "engagement. Additionally, data-driven monitoring systems can identify vulnerabilities and support "
    "timely intervention. Consequently, stakeholders should integrate scientific assessment with "
    "long-term planning in order to reduce systemic risk. Moreover, equitable implementation ensures "
    "that mitigation efforts remain both effective and socially responsible. Ultimately, a resilient "
    "framework depends on continuous evaluation, interdisciplinary collaboration, and the consistent "
    "application of established best practices."
)

EXAMPLES: tuple[tuple[str, str], ...] = (
    ("Human", HUMAN_EXAMPLE_1),
    ("Human", HUMAN_EXAMPLE_2),
    ("AI-generated", AI_EXAMPLE_1),
    ("AI-generated", AI_EXAMPLE_2),
)

MODEL_CHOICES: dict[str, tuple[str, ...]] = {
    "DistilBERT": ("distilbert",),
    "TF-IDF": ("tfidf",),
    "Both": ("distilbert", "tfidf"),
}

COMPARISON_FIGURES: tuple[tuple[str, str], ...] = (
    ("metric_comparison.png", "Classification metrics, speed, and model size"),
    ("roc_curves.png", "ROC curves"),
    ("calibration_curves.png", "Calibration curves"),
    ("cm_baseline.png", "TF-IDF confusion matrix"),
    ("cm_transformer.png", "DistilBERT confusion matrix"),
    ("cross_domain.png", "In-domain versus cross-domain scores"),
    ("error_boxplots.png", "Style features of correct and incorrect texts"),
    ("accuracy_by_length.png", "Accuracy by text length"),
)

CSS = """
.disclaimer {
  border: 1px solid #c2410c;
  background: #fff7ed;
  padding: 12px 14px;
  border-radius: 8px;
}
.disclaimer p {
  margin: 0;
  font-weight: 600;
}
"""


def classification_view(text: str, choice: str) -> dict[str, Any]:
    """Score the text and describe what the demo should show.

    A missing model is reported in ``warning`` and does not hide a result from
    the other model when the dropdown is set to Both.
    """
    raw = "" if text is None else str(text)
    if not raw.strip():
        return {"warning": "Enter some text to classify.", "distilbert": None, "tfidf": None, "highlights": None}
    if choice not in MODEL_CHOICES:
        raise ValueError(f"Unknown model choice {choice!r}.")
    notes: list[str] = []
    length_warning = short_text_warning(raw)
    if length_warning:
        notes.append(length_warning)
    selected = MODEL_CHOICES[choice]
    distilbert_value = None
    tfidf_value = None
    highlights = None
    if "distilbert" in selected:
        distilbert_value, message = _safe_label("DistilBERT", raw, "distilbert")
        if message:
            notes.append(message)
    if "tfidf" in selected:
        tfidf_value, message = _safe_label("TF-IDF", raw, "tfidf")
        if message:
            notes.append(message)
        elif tfidf_value is not None:
            highlights = tfidf_highlights(raw)
            if not any(label for _span, label in highlights):
                notes.append(
                    "None of the TF-IDF vocabulary appeared in this text, "
                    "so there are no contributing words to highlight."
                )
    logger.info(
        "Classified %d words with %s.",
        word_count(raw),
        choice,
    )
    return {
        "warning": "\n\n".join(notes),
        "distilbert": distilbert_value,
        "tfidf": tfidf_value,
        "highlights": highlights,
    }


def comparison_markdown() -> str:
    """Saved held-out comparison, or a note when that file is still missing."""
    markdown_path = METRICS_DIR / "model_comparison.md"
    if markdown_path.is_file():
        body = markdown_path.read_text(encoding="utf-8").strip()
        return body + "\n\nFigures below are the last files written under `outputs/figures`."
    csv_path = METRICS_DIR / "model_comparison.csv"
    if csv_path.is_file():
        frame = pd.read_csv(csv_path)
        return _markdown_table(frame) + "\n\nFigures below are the last files written under `outputs/figures`."
    return (
        "The comparison table is not in `outputs/metrics` yet. "
        "Train both models, then run `python -m src.evaluate`."
    )


def build_demo() -> gr.Blocks:
    """Assemble the detect tab and the saved-results tab."""
    with gr.Blocks(title="AI-text detector") as demo:
        gr.Markdown("# Detecting AI-generated text")
        gr.Markdown(DISCLAIMER, elem_classes=["disclaimer"])
        with gr.Tabs():
            with gr.Tab("Detect"):
                _build_detect_tab()
            with gr.Tab("Comparison"):
                _build_comparison_tab()
    return demo


def main() -> None:
    """Start the local demo."""
    _configure_logging()
    logger.info("Starting the Gradio demo.")
    demo = build_demo()
    demo.launch(theme=gr.themes.Soft(), css=CSS)


def _build_detect_tab() -> None:
    text_box = gr.Textbox(
        label="Text",
        lines=12,
        placeholder="Paste a passage to classify.",
        info="DistilBERT reads the first 256 tokens. TF-IDF uses the full text.",
    )
    model_choice = gr.Dropdown(
        choices=list(MODEL_CHOICES),
        value="DistilBERT",
        label="Model",
    )
    submit = gr.Button("Submit", variant="primary")
    warning = gr.Markdown()
    with gr.Row():
        distilbert_label = gr.Label(label="DistilBERT", num_top_classes=2)
        tfidf_label = gr.Label(label="TF-IDF", num_top_classes=2, visible=False)
    highlights = gr.HighlightedText(
        label="Top contributing words (TF-IDF)",
        combine_adjacent=True,
        show_legend=True,
        color_map={"Human": "#4C78A8", "AI": "#E45756"},
        visible=False,
    )
    gr.Markdown(
        "Examples 1 and 2 are human paragraphs. Examples 3 and 4 are written in a generic generated style."
    )
    gr.Examples(
        examples=[[sample] for _kind, sample in EXAMPLES],
        inputs=text_box,
        label="Examples",
    )
    outputs = [warning, distilbert_label, tfidf_label, highlights]
    submit.click(_on_submit, inputs=[text_box, model_choice], outputs=outputs)
    text_box.submit(_on_submit, inputs=[text_box, model_choice], outputs=outputs)


def _build_comparison_tab() -> None:
    gr.Markdown(comparison_markdown())
    shown = _render_figures(COMPARISON_FIGURES)
    known = {name for name, _caption in COMPARISON_FIGURES}
    extras = []
    if FIGURES_DIR.is_dir():
        extras = sorted(path for path in FIGURES_DIR.glob("*.png") if path.name not in known)
    for path in extras:
        gr.Image(value=str(path), label=path.stem.replace("_", " "), buttons=["download"])
        shown = True
    if not shown:
        gr.Markdown(
            "No figures are in `outputs/figures` yet. "
            "Run `python -m src.evaluate` to save the comparison plots."
        )


def _render_figures(figures: tuple[tuple[str, str], ...]) -> bool:
    shown = False
    if not FIGURES_DIR.is_dir():
        return False
    for filename, caption in figures:
        path = FIGURES_DIR / filename
        if not path.is_file():
            continue
        gr.Image(value=str(path), label=caption, buttons=["download"])
        shown = True
    return shown


def _on_submit(text: str, choice: str) -> tuple[Any, Any, Any, Any]:
    view = classification_view(text, choice)
    return (
        view["warning"],
        gr.update(value=view["distilbert"], visible=view["distilbert"] is not None),
        gr.update(value=view["tfidf"], visible=view["tfidf"] is not None),
        gr.update(value=view["highlights"], visible=view["highlights"] is not None),
    )


def _safe_label(display_name: str, text: str, model: str) -> tuple[dict[str, Any] | None, str]:
    try:
        result = predict(text, model)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError, OSError) as exc:
        return None, f"{display_name}: {exc}"
    if result["label"] not in {HUMAN_LABEL, AI_LABEL}:
        return None, f"{display_name}: unexpected label {result['label']!r}."
    return _label_value(result), ""


def _label_value(result: dict[str, Any]) -> dict[str, float]:
    """Map a prediction to the class-to-confidence dict ``gr.Label`` displays as bars."""
    human = float(result["probabilities"]["human"])
    ai = float(result["probabilities"]["ai"])
    if result["label"] == AI_LABEL:
        return {AI_LABEL: ai, HUMAN_LABEL: human}
    return {HUMAN_LABEL: human, AI_LABEL: ai}


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row[column]) for column in frame.columns) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, rule, *body])


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


if __name__ == "__main__":
    main()
