"""S3 raw-data downloader for M-114 (FIVES).

The FIVES dataset is distributed by Figshare as a single 1.76GB .rar archive.
We mirror that archive at ``s3://med-vr-datasets/M-114/fives/34969398``.

This downloader:
  1. ``aws s3 cp`` the .rar to ``raw/_archive/fives.rar`` (idempotent).
  2. Extracts the archive using ``unar`` (apt: ``unar``; non-free ``unrar``
     is intentionally avoided).
  3. Yields one dict per (image, mask) pair, covering test split first
     then train split, with the FIVES disease class encoded in the
     filename suffix (A=AMD, D=DR, G=Glaucoma, N=Normal).
"""
from __future__ import annotations
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterator, Optional


CLASS_LETTER_TO_DIAGNOSIS = {
    "A": ("Age-related Macular Degeneration", "AMD", True),
    "D": ("Diabetic Retinopathy", "DR", True),
    "G": ("Glaucoma", "G", True),
    "N": ("Normal", "N", False),
}

FIVES_ROOT_DIRNAME = "FIVES A Fundus Image Dataset for AI-based Vessel Segmentation"


class TaskDownloader:
    """Fetches the FIVES rar from S3, unpacks it, then yields image+mask pairs."""

    def __init__(self, config):
        self.config = config
        self.raw_dir = Path(config.raw_dir)
        self.archive_dir = self.raw_dir / "_archive"
        self.extract_dir = self.raw_dir / "_extracted"

    # -- Stage 1: download .rar from S3 --------------------------------------
    def _download_archive(self) -> Path:
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        local_rar = self.archive_dir / "fives.rar"
        if local_rar.exists() and local_rar.stat().st_size > 1_000_000_000:
            print(f"  archive already present at {local_rar} ({local_rar.stat().st_size} bytes)")
            return local_rar

        s3_uri = f"s3://{self.config.s3_bucket}/{self.config.rar_key}"
        print(f"  downloading {s3_uri} -> {local_rar}")
        # Use aws-cli for speed (handles multipart). Fall back to boto3 if missing.
        if shutil.which("aws"):
            subprocess.run(
                ["aws", "s3", "cp", s3_uri, str(local_rar), "--only-show-errors"],
                check=True,
            )
        else:
            import boto3
            boto3.client("s3").download_file(
                self.config.s3_bucket, self.config.rar_key, str(local_rar)
            )
        return local_rar

    # -- Stage 2: unpack .rar -------------------------------------------------
    def _extract_archive(self, local_rar: Path) -> Path:
        fives_root = self.extract_dir / FIVES_ROOT_DIRNAME
        if fives_root.exists() and (fives_root / "test" / "Original").is_dir():
            print(f"  archive already extracted at {fives_root}")
            return fives_root

        if not shutil.which("unar"):
            raise RuntimeError(
                "`unar` not found on PATH. Install with: apt-get install -y unar"
            )

        self.extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"  extracting {local_rar} -> {self.extract_dir}")
        # `unar -q -o <dir>` writes the archive's own top dir under <dir>.
        subprocess.run(
            ["unar", "-q", "-f", "-o", str(self.extract_dir), str(local_rar)],
            check=True,
        )
        if not fives_root.exists():
            raise RuntimeError(
                f"Extraction succeeded but expected dir not found: {fives_root}"
            )
        return fives_root

    # -- Stage 3: ensure raw is ready ----------------------------------------
    def ensure_raw(self) -> Path:
        local_rar = self._download_archive()
        return self._extract_archive(local_rar)

    # -- Stage 4: iterate samples --------------------------------------------
    def iter_samples(self, limit: Optional[int] = None) -> Iterator[dict]:
        fives_root = self.ensure_raw()
        yielded = 0

        for split in self.config.splits:
            split_dir = fives_root / split
            img_dir = split_dir / "Original"
            mask_dir = split_dir / "Ground truth"
            if not img_dir.is_dir():
                print(f"  skipping missing split dir: {img_dir}")
                continue

            for img_path in sorted(img_dir.iterdir(), key=_natural_key):
                if not img_path.suffix.lower() == ".png":
                    continue
                mask_path = mask_dir / img_path.name
                if not mask_path.exists():
                    continue

                stem = img_path.stem  # e.g. "100_D"
                class_letter = stem.split("_")[-1] if "_" in stem else "N"
                diagnosis, code, diseased = CLASS_LETTER_TO_DIAGNOSIS.get(
                    class_letter.upper(), ("Unknown", "U", False)
                )

                yield {
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "image_id": f"{split}_{stem}",
                    "split": split,
                    "diagnosis": diagnosis,
                    "diagnosis_code": code,
                    "is_diseased": diseased,
                }
                yielded += 1
                if limit is not None and yielded >= limit:
                    return

    # Backwards-compat name expected by core.run_download
    def download(self, limit: Optional[int] = None) -> Iterator[dict]:
        yield from self.iter_samples(limit=limit)


def _natural_key(path: Path):
    """Sort 1, 2, 10 instead of 1, 10, 2."""
    stem = path.stem
    head = stem.split("_")[0]
    try:
        return (int(head), stem)
    except ValueError:
        return (10**9, stem)


def create_downloader(config) -> TaskDownloader:
    return TaskDownloader(config)
