import os
from pathlib import Path
from typing import Final

from kagglehub import dataset_download

from transaction_analysis.paths import FRAUD_DATASET_DIR

DATASETS_TO_DOWNLOAD: Final[dict[str, Path]] = {
    "computingvictor/transactions-fraud-datasets": FRAUD_DATASET_DIR / "raw",
}


def download_competition_datasets(datasets: dict[str, Path] = DATASETS_TO_DOWNLOAD) -> None:
    for dataset_name, dataset_path in datasets.items():
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path, exist_ok=True)
            dataset_download(handle=dataset_name, output_dir=str(dataset_path))


def run(force: bool = False) -> None:
    if not force and all(dataset_path.exists() for dataset_path in DATASETS_TO_DOWNLOAD.values()):
        print("Datasets already downloaded, skipping. To force re-run, set force=True")
    else:
        download_competition_datasets(datasets=DATASETS_TO_DOWNLOAD)
