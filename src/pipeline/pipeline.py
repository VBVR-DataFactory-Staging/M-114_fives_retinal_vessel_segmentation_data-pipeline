"""Task pipeline for M-114 FIVES retinal vessel segmentation.

For each FIVES (image, vessel_mask) pair, produces the standard
seven-file VBVR sample::

    data/questions/fives_retinal_vessel_segmentation_task/<task_id>/
        first_frame.png       — raw fundus
        final_frame.png       — fundus + red vessel overlay
        prompt.txt            — segmentation instruction
        first_video.mp4       — slight camera shake on the raw fundus
        last_video.mp4        — slight camera shake on the annotated overlay
        ground_truth.mp4      — vessel-reveal animation
        metadata.json         — provenance + parameters
"""
from __future__ import annotations
import hashlib
import shutil
from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import numpy as np

from core.pipeline import BasePipeline, OutputWriter, SampleProcessor, TaskSample
from src.download.downloader import create_downloader
from src.pipeline import transforms
from src.pipeline.config import TaskConfig


_TMP_DIR = Path("_tmp_videos")


class TaskPipeline(BasePipeline):
    """Generate FIVES vessel-segmentation tasks one image at a time."""

    def __init__(self, config: Optional[TaskConfig] = None):
        super().__init__(config or TaskConfig())
        self.task_config: TaskConfig = self.config  # narrow the type
        self.downloader = create_downloader(self.task_config)

    # ── 1) download — yield one dict per (image, mask) pair ────────────
    def download(self) -> Iterator[dict]:
        yield from self.downloader.iter_samples(limit=self.task_config.num_samples)

    # ── 2) process — convert one raw dict into a TaskSample ────────────
    def process_sample(self, raw_sample: dict, idx: int) -> Optional[TaskSample]:
        img_path: Path = raw_sample["image_path"]
        mask_path: Path = raw_sample["mask_path"]
        image_id: str = raw_sample["image_id"]
        diagnosis: str = raw_sample["diagnosis"]
        diagnosis_code: str = raw_sample["diagnosis_code"]
        is_diseased: bool = raw_sample["is_diseased"]

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            print(f"  [skip] cannot read {image_id}")
            return None

        # 2048×2048 is too big for fast video encoding; downsample to 1024.
        img, mask = _downsample(img, mask, target=1024)
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

        overlay = transforms.create_overlay(img, mask)
        first_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        final_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        task_id = f"m114_{idx:05d}_{image_id}"
        _TMP_DIR.mkdir(parents=True, exist_ok=True)

        n = self.task_config.num_video_frames
        fps = self.task_config.fps

        first_video = _TMP_DIR / f"{task_id}_first.mp4"
        last_video = _TMP_DIR / f"{task_id}_last.mp4"
        gt_video = _TMP_DIR / f"{task_id}_gt.mp4"

        transforms.make_video(
            transforms.fundus_motion_frames(img, num_frames=n),
            first_video, fps=fps,
        )
        transforms.make_video(
            transforms.annotated_motion_frames(img, mask, num_frames=n),
            last_video, fps=fps,
        )
        transforms.make_video(
            transforms.vessel_reveal_frames(img, mask, num_frames=n),
            gt_video, fps=fps,
        )

        mask_area = int(np.count_nonzero(mask))
        total_px = mask.shape[0] * mask.shape[1]
        param_hash = hashlib.sha256(
            f"{image_id}|{diagnosis_code}|fives".encode()
        ).hexdigest()[:16]

        metadata = {
            "task_id": f"fives_retinal_vessel_segmentation_{idx:08d}",
            "generator": "M-114_fives_retinal_vessel_segmentation_data-pipeline",
            "source_dataset": "FIVES",
            "source_sample_id": image_id,
            "parameters": {
                "split": raw_sample["split"],
                "diagnosis": diagnosis,
                "diagnosis_code": diagnosis_code,
                "is_diseased": is_diseased,
                "image_size": {"width": img.shape[1], "height": img.shape[0]},
                "mask_area_pixels": mask_area,
                "mask_area_ratio": round(mask_area / total_px, 6),
                "fps": fps,
                "num_frames": n,
                "duration_seconds": round(n / fps, 2),
            },
            "ground_truth": {
                "label": "diseased" if is_diseased else "normal",
                "diagnosis": diagnosis,
                "task_type": "C1_segmentation",
                "target": "retinal_vessel_tree",
            },
            "multimodal_source": {"imaging": "real"},
            "param_hash": param_hash,
            "generation": {
                "seed": 42 + idx,
                "generator_version": "1.0.0",
            },
        }

        return SampleProcessor.build_sample(
            task_id=task_id,
            domain=self.task_config.domain,
            first_image=first_rgb,
            prompt=self.task_config.task_prompt,
            final_image=final_rgb,
            first_video=str(first_video),
            last_video=str(last_video),
            ground_truth_video=str(gt_video),
            metadata=metadata,
        )

    # ── orchestrator: download → process → write, with cleanup ────────
    def run(self) -> List[TaskSample]:
        writer = OutputWriter(self.config.output_dir)
        samples: List[TaskSample] = []
        try:
            for idx, raw in enumerate(self.download()):
                sample = self.process_sample(raw, idx)
                if sample is None:
                    print(f"  Skipped sample {idx}")
                    continue
                writer.write_sample(sample)
                samples.append(sample)
                if (idx + 1) % 5 == 0:
                    print(f"  Processed {idx + 1} samples...")
            print(
                f"Done! Wrote {len(samples)} samples -> "
                f"{self.config.output_dir}/{self.task_config.domain}_task/"
            )
        finally:
            if _TMP_DIR.exists():
                shutil.rmtree(_TMP_DIR, ignore_errors=True)
        return samples


def _downsample(img: np.ndarray, mask: np.ndarray, target: int = 1024):
    """Resize the longer edge to *target* preserving aspect ratio."""
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= target:
        return img, mask
    scale = target / long_edge
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img2 = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask2 = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return img2, mask2
