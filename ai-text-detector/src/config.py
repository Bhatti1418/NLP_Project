"""Central configuration for the AI-generated text detector.

Paths are derived from this file with pathlib so no script hardcodes an
absolute location. Import these constants rather than rebuilding paths.
"""

from pathlib import Path

SEED: int = 42

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
METRICS_DIR: Path = OUTPUTS_DIR / "metrics"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

MAX_TRAIN_SAMPLES: int = 20_000
TEST_SIZE: float = 0.15
VAL_SIZE: float = 0.15
MAX_LEN: int = 256
MODEL_NAME: str = "distilbert-base-uncased"
BATCH_SIZE: int = 16
EPOCHS: int = 2
LR: float = 2e-5
