"""
Download TCGA-BRCA and TCGA-LGG data via GDC API.

Usage:
    python scripts/download_tcga.py --cohort BRCA --out data/raw/
    python scripts/download_tcga.py --cohort LGG  --out data/raw/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import requests

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT  = "https://api.gdc.cancer.gov/data"

COHORT_PROJECT = {
    "BRCA": "TCGA-BRCA",
    "LGG":  "TCGA-LGG",
}


def build_filter(project_id: str) -> dict:
    return {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Gene Expression Quantification"]}},
            {"op": "in", "content": {"field": "data_format", "value": ["TSV"]}},
            {"op": "in", "content": {"field": "experimental_strategy", "value": ["RNA-Seq"]}},
        ],
    }


def fetch_file_manifest(project_id: str, max_files: int = 1200) -> list:
    payload = {
        "filters": json.dumps(build_filter(project_id)),
        "fields":  "file_id,file_name,cases.case_id,cases.submitter_id",
        "format":  "JSON",
        "size":    max_files,
    }
    r = requests.get(GDC_FILES_ENDPOINT, params=payload, timeout=60)
    r.raise_for_status()
    return r.json()["data"]["hits"]


def download_files(file_ids: list, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_size = 100
    for i in range(0, len(file_ids), batch_size):
        batch = file_ids[i: i + batch_size]
        payload = {"ids": batch}
        r = requests.post(GDC_DATA_ENDPOINT, json=payload, timeout=300, stream=True)
        r.raise_for_status()
        out_path = out_dir / f"batch_{i//batch_size}.tar.gz"
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
        print(f"  Downloaded batch {i//batch_size} → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["BRCA", "LGG"], required=True)
    parser.add_argument("--out", default="data/raw/")
    args = parser.parse_args()

    project_id = COHORT_PROJECT[args.cohort]
    print(f"Fetching manifest for {project_id}...")
    hits = fetch_file_manifest(project_id)
    file_ids = [h["file_id"] for h in hits]
    print(f"  {len(file_ids)} files found")
    download_files(file_ids, Path(args.out) / args.cohort.lower())
    print("Done.")


if __name__ == "__main__":
    main()
