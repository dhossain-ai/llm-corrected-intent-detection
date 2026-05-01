"""Project configuration values."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURES_DIR = OUTPUTS_DIR / "figures"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

MODELS_DIR = PROJECT_ROOT / "models"

CLINC_DATASET_NAME = "clinc_oos"
CLINC_PREFERRED_CONFIG = "plus"
