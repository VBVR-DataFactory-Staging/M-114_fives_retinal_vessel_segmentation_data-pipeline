"""Core download module — fetch raw data from external sources.

Provides generic download utilities (HuggingFace, S3 via boto3). The
actual dataset-specific download logic lives in ``src.download``; this
module always delegates to it via :func:`run_download`.
"""

from pathlib import Path
from typing import Iterator, Optional


# ============================================================================
#  HuggingFace downloader
# ============================================================================

class HuggingFaceDownloader:
    """Download datasets from HuggingFace Hub into the ``raw/`` directory."""

    def __init__(self, repo_id: str, split: str = "test", raw_dir: Path = Path("raw")):
        self.repo_id = repo_id
        self.split = split
        self.raw_dir = Path(raw_dir)

    def download(self, limit: Optional[int] = None) -> Iterator[dict]:
        from datasets import load_dataset

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {self.repo_id} (split: {self.split}) -> {self.raw_dir}/")
        dataset = load_dataset(
            self.repo_id,
            split=self.split,
            cache_dir=str(self.raw_dir / ".cache"),
        )

        if limit is not None:
            dataset = dataset.select(range(min(limit, len(dataset))))

        print(f"Streaming {len(dataset)} samples...")

        for item in dataset:
            yield item


# ============================================================================
#  S3 download (via boto3, EC2 IAM role)
# ============================================================================

def download_from_s3(
    bucket_name: str,
    s3_prefix: str,
    local_dir: Path,
) -> int:
    """Download dataset from S3 (private bucket OK via EC2 IAM role)."""
    import boto3

    s3 = boto3.client("s3")
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Listing s3://{bucket_name}/{s3_prefix} ...")

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)

    files = []
    for page in pages:
        for obj in page.get("Contents", []):
            files.append(obj["Key"])

    print(f"Found {len(files)} files to download...")

    downloaded = 0
    for key in files:
        rel = key.replace(s3_prefix, "", 1).lstrip("/")
        if not rel:
            continue
        local_path = local_dir / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket_name, key, str(local_path))
        downloaded += 1
        if downloaded % 10 == 0:
            print(f"  Downloaded {downloaded}/{len(files)} files...")

    print(f"Download complete: {downloaded} files")
    return downloaded


# ============================================================================
#  Orchestration — delegates to src.download
# ============================================================================

def run_download(config) -> Iterator[dict]:
    """Standard download entry point — calls src.download.create_downloader."""
    from src.download import create_downloader

    downloader = create_downloader(config)
    yield from downloader.download(limit=config.num_samples)
