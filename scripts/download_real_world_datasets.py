"""
Acquisition script for Open-Access Real-World Clinical Datasets.
Downloads genuine human clinical cohorts into untracked data/ directory:
1. OASIS-2 Longitudinal Dementia MRI Cohort (Washington University)
2. UCI Parkinson's Telemonitoring Cohort (Oxford University / Tsanas et al.)
3. UCI Parkinson's Diagnostic Cohort (Little et al.)
"""

from __future__ import annotations

import logging
from pathlib import Path
import urllib.request
import pandas as pd

logger = logging.getLogger("aimedres.data_downloader")

DATA_SOURCES = {
    "oasis_longitudinal": {
        "url": "https://raw.githubusercontent.com/deepak525/Dementia-Classification-Compare-Classifiers/master/oasis_longitudinal.csv",
        "filename": "oasis_longitudinal.csv",
        "description": "OASIS-2 Longitudinal Brain MRI and Dementia Cohort (373 visits, 150 subjects)",
    },
    "parkinsons_telemonitoring": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data",
        "filename": "parkinsons_telemonitoring.csv",
        "description": "UCI Parkinson's Telemonitoring Study (5,875 records, 42 patients, motor UPDRS)",
    },
    "parkinsons_diagnostic": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data",
        "filename": "parkinsons_diagnostic.csv",
        "description": "UCI Parkinson's Diagnostic Voice Study (195 records, 31 subjects)",
    },
}


def download_real_world_datasets(output_dir: Path = Path("data/real_world")) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = {}

    for key, info in DATA_SOURCES.items():
        dest = output_dir / info["filename"]
        if dest.exists() and dest.stat().st_size > 0:
            logger.info(f"Using existing file: {dest}")
        else:
            logger.info(f"Downloading {info['description']} from {info['url']}...")
            try:
                urllib.request.urlretrieve(info["url"], dest)
                logger.info(f"Downloaded successfully: {dest} ({dest.stat().st_size} bytes)")
            except Exception as e:
                logger.error(f"Failed to download {key}: {e}")
                continue

        # Quick validation
        try:
            df = pd.read_csv(dest)
            logger.info(f"Validated {info['filename']}: shape={df.shape}")
            downloaded[key] = dest
        except Exception as e:
            logger.error(f"Error reading downloaded CSV {dest}: {e}")

    return downloaded


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    res = download_real_world_datasets()
    print("\n--- Download Summary ---")
    for name, path in res.items():
        print(f"[OK] {name}: {path.resolve()}")
