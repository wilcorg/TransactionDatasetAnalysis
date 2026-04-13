from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATASET_DIR: Final[Path] = PROJECT_ROOT / "dataset" / "cleaned"
RAW_DATASET_DIR: Final[Path] = PROJECT_ROOT / "dataset" / "raw"
