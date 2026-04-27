#!/usr/bin/env python3
"""M-114 FIVES retinal vessel segmentation - self-contained generator.

Per fundus image with vessel mask, produces a 7-file VBVR sample:
    first_frame.png       raw fundus
    final_frame.png       fundus + red vessel overlay
    prompt.txt            segmentation instruction
    first_video.mp4       zoom-in on raw fundus  (60+ frames)
    last_video.mp4        full walkthrough on annotated fundus (60+ frames)
    ground_truth.mp4      progressive vessel-reveal animation (60+ frames)
    metadata.json         provenance + parameters

Sample dir naming: m114_NNNNN  (pure-digit suffix - required by harness
samples/_meta scanners).  Source images are augmented 5x (rotation, flip,
intensity, crop) so 800 base samples -> up to 800 unique tasks even after
sampling stride.

Raw layout on S3:
    s3://med-vr-datasets/M-114/fives/34969398           (rar, 1.6GB)
    s3://med-vr-datasets/M-114/fives/FIVES%20A%20...rar (rar, dup of above)

Usage:
    python3 examples/generate.py
    python3 examples/generate.py --output /tmp/out
    python3 examples/generate.py --num-samples 3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Force unbuffered stdout so EC2 logs stream live to S3.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DOMAIN = "fives_retinal_vessel_segmentation"
TASK_DIR_NAME = f"{DOMAIN}_task"
SAMPLE_PREFIX = "m114"
PAD_WIDTH = 5

S3_BUCKET = "med-vr-datasets"
S3_PREFIX = "M-114/fives/"

VESSEL_COLOR_BGR = (0, 0, 255)  # red
RESIZE_LONG = 1024              # downsample 2048x2048 -> 1024
FRAME_SIZE = 768                # video output size (square pad)
NUM_FRAMES = 64                 # 60+ frames @ 12fps ~ 5.3s
FPS = 12

CLASS_LETTER = {
    "A": ("Age-related Macular Degeneration", "AMD", True),
    "D": ("Diabetic Retinopathy", "DR", True),
    "G": ("Glaucoma", "G", True),
    "N": ("Normal", "N", False),
}

FIVES_ROOT_NAME = "FIVES A Fundus Image Dataset for AI-based Vessel Segmentation"

PROMPT = (
    "Segment all blood vessels in this fundus image. Trace the complete "
    "retinal vasculature including arteries, veins, and their branches; "
    "highlight every vessel pixel and ignore background, optic disc, "
    "and macular regions. Render the result as a red overlay on the fundus."
)


# ---------------------------------------------------------------------------
# Raw download + extraction
# ---------------------------------------------------------------------------
def ensure_raw(raw_root: Path) -> Path:
    """Download FIVES rar from S3 (aws s3 sync) and extract via unar."""
    raw_root.mkdir(parents=True, exist_ok=True)
    archive_dir = raw_root / "_archive"
    extract_dir = raw_root / "_extracted"
    archive_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sync the prefix (gets both the bare key + URL-encoded duplicate).
    print(f"[raw] aws s3 sync s3://{S3_BUCKET}/{S3_PREFIX} -> {archive_dir}", flush=True)
    subprocess.run(
        ["aws", "s3", "sync", f"s3://{S3_BUCKET}/{S3_PREFIX}",
         str(archive_dir), "--only-show-errors"],
        check=True,
    )

    # 2. Locate the rar (largest file in archive_dir).
    candidates = sorted(
        (p for p in archive_dir.iterdir() if p.is_file()),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(f"No archive in {archive_dir}")
    rar_path = candidates[0]
    print(f"[raw] using archive {rar_path.name} ({rar_path.stat().st_size} bytes)", flush=True)

    # 3. Extract - idempotent, skip if already extracted.
    fives_root = extract_dir / FIVES_ROOT_NAME
    if fives_root.exists() and (fives_root / "test" / "Original").is_dir():
        print(f"[raw] already extracted at {fives_root}", flush=True)
        return fives_root

    extract_dir.mkdir(parents=True, exist_ok=True)
    if not shutil.which("unar"):
        print("[raw] installing unar via apt-get", flush=True)
        subprocess.run(["apt-get", "update", "-qq"], check=False)
        subprocess.run(["apt-get", "install", "-y", "-qq", "unar"], check=False)
    if not shutil.which("unar"):
        raise RuntimeError("unar not available; install via `apt-get install -y unar`")

    print(f"[raw] unar -> {extract_dir}", flush=True)
    # FIVES rar contains 1-2 unreadable files (xlsx metadata, stray png).
    # unar may exit non-zero but extracts >500 PNGs - that's enough.
    subprocess.run(
        ["unar", "-q", "-f", "-o", str(extract_dir), str(rar_path)],
        check=False,
    )
    if not fives_root.is_dir():
        raise RuntimeError(f"extraction failed; expected {fives_root}")
    n_pngs = sum(1 for _ in fives_root.rglob("*.png"))
    print(f"[raw] extracted {n_pngs} PNGs", flush=True)
    if n_pngs < 500:
        raise RuntimeError(f"extracted only {n_pngs} PNGs, expected >500")
    return fives_root


def iter_pairs(fives_root: Path) -> List[Tuple[Path, Path, str, str]]:
    """Return list of (image_path, mask_path, image_id, class_letter)."""
    pairs: List[Tuple[Path, Path, str, str]] = []
    for split in ("test", "train"):
        img_dir = fives_root / split / "Original"
        mask_dir = fives_root / split / "Ground truth"
        if not img_dir.is_dir():
            continue
        for img_path in sorted(img_dir.iterdir(), key=lambda p: _nat_key(p.stem)):
            if img_path.suffix.lower() != ".png":
                continue
            mask_path = mask_dir / img_path.name
            if not mask_path.exists():
                continue
            stem = img_path.stem  # e.g. "100_D"
            cls = stem.rsplit("_", 1)[-1].upper() if "_" in stem else "N"
            if cls not in CLASS_LETTER:
                cls = "N"
            image_id = f"{split}_{stem}"
            pairs.append((img_path, mask_path, image_id, cls))
    return pairs


def _nat_key(stem: str):
    head = stem.split("_", 1)[0]
    try:
        return (int(head), stem)
    except ValueError:
        return (10**9, stem)


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------
def downsample(img: np.ndarray, mask: np.ndarray, target: int = RESIZE_LONG):
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= target:
        return img, mask
    s = target / long_edge
    nw, nh = int(round(w * s)), int(round(h * s))
    img2 = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    mask2 = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return img2, mask2


def square_pad(img: np.ndarray, side: int = FRAME_SIZE) -> np.ndarray:
    """Resize longest edge to side and zero-pad to (side, side)."""
    h, w = img.shape[:2]
    s = side / max(h, w)
    nw, nh = int(round(w * s)), int(round(h * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    if resized.ndim == 2:
        canvas = np.zeros((side, side), dtype=resized.dtype)
    else:
        canvas = np.zeros((side, side, resized.shape[2]), dtype=resized.dtype)
    y0 = (side - nh) // 2
    x0 = (side - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def overlay_vessels(img_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Red overlay + contour."""
    colored = np.zeros_like(img_bgr)
    colored[mask > 0] = VESSEL_COLOR_BGR
    blended = cv2.addWeighted(img_bgr, 1.0 - alpha, colored, alpha, 0.0)
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(blended, contours, -1, VESSEL_COLOR_BGR, 1)
    return blended


# ---------------------------------------------------------------------------
# Augmentations (5 modes)
# ---------------------------------------------------------------------------
def augment(img: np.ndarray, mask: np.ndarray, mode: int, rng: random.Random):
    """Five-way augmentation. mode=0 returns the original."""
    if mode == 0:
        return img, mask
    if mode == 1:  # rotate +/-10 deg
        angle = rng.uniform(-10, 10)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img2 = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        mask2 = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return img2, mask2
    if mode == 2:  # horizontal flip - OK on retinal images (rough symmetry)
        return cv2.flip(img, 1), cv2.flip(mask, 1)
    if mode == 3:  # intensity +/-10%
        factor = rng.uniform(0.9, 1.1)
        img2 = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return img2, mask
    if mode == 4:  # slight center crop (~92%) then resize back
        h, w = img.shape[:2]
        crop = rng.uniform(0.88, 0.96)
        ch, cw = int(h * crop), int(w * crop)
        y0 = (h - ch) // 2
        x0 = (w - cw) // 2
        img2 = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_AREA)
        mask2 = cv2.resize(mask[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_NEAREST)
        return img2, mask2
    return img, mask


# ---------------------------------------------------------------------------
# Video frame generators
# ---------------------------------------------------------------------------
def zoom_in_frames(img: np.ndarray, n: int = NUM_FRAMES) -> List[np.ndarray]:
    """Smoothly zoom from 1.0x to 1.25x centered."""
    h, w = img.shape[:2]
    frames: List[np.ndarray] = []
    for i in range(n):
        t = i / max(1, n - 1)
        zoom = 1.0 + 0.25 * t
        ch, cw = max(1, int(h / zoom)), max(1, int(w / zoom))
        y0 = (h - ch) // 2
        x0 = (w - cw) // 2
        cropped = img[y0:y0 + ch, x0:x0 + cw]
        frames.append(square_pad(cropped, FRAME_SIZE))
    return frames


def walkthrough_frames(img: np.ndarray, n: int = NUM_FRAMES) -> List[np.ndarray]:
    """Pan around the image: small circular drift."""
    h, w = img.shape[:2]
    frames: List[np.ndarray] = []
    R = min(h, w) * 0.06
    for i in range(n):
        t = i / max(1, n - 1)
        ang = 2 * np.pi * t
        dx = float(R * np.cos(ang))
        dy = float(R * np.sin(ang))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        frames.append(square_pad(shifted, FRAME_SIZE))
    return frames


def vessel_reveal_frames(img: np.ndarray, mask: np.ndarray, n: int = NUM_FRAMES) -> List[np.ndarray]:
    """Progressive overlay: 0% -> 100% of vessel mask painted in red."""
    frames: List[np.ndarray] = []
    base = img.copy()
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        for _ in range(n):
            frames.append(square_pad(base, FRAME_SIZE))
        return frames
    order = np.argsort(xs)
    ys, xs = ys[order], xs[order]
    chunks = np.array_split(np.arange(len(xs)), n)
    cur_mask = np.zeros_like(mask)
    for chunk in chunks:
        if len(chunk):
            cur_mask[ys[chunk], xs[chunk]] = 255
        f = overlay_vessels(base, cur_mask, alpha=0.55)
        frames.append(square_pad(f, FRAME_SIZE))
    while len(frames) < n:
        frames.append(frames[-1])
    return frames[:n]


# ---------------------------------------------------------------------------
# Video writer
# ---------------------------------------------------------------------------
def write_mp4(frames: List[np.ndarray], out_path: Path, fps: int = FPS) -> None:
    """Encode frames to mp4 via ffmpeg (libx264, browser-compatible)."""
    if not frames:
        raise RuntimeError(f"no frames for {out_path}")
    h, w = frames[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-preset", "veryfast",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for f in frames:
        if f.ndim == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    err = proc.stderr.read().decode("utf-8", errors="ignore")
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed ({rc}) for {out_path}: {err[:500]}")


# ---------------------------------------------------------------------------
# Per-sample writer
# ---------------------------------------------------------------------------
def write_sample(
    out_dir: Path,
    sample_idx: int,
    img_bgr: np.ndarray,
    mask: np.ndarray,
    image_id: str,
    diagnosis: str,
    diagnosis_code: str,
    is_diseased: bool,
    aug_mode: int,
) -> bool:
    sample_id = f"{SAMPLE_PREFIX}_{sample_idx:0{PAD_WIDTH}d}"
    sample_dir = out_dir / TASK_DIR_NAME / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    img_sq = square_pad(img_bgr, FRAME_SIZE)
    overlay = overlay_vessels(img_bgr, mask)
    overlay_sq = square_pad(overlay, FRAME_SIZE)

    cv2.imwrite(str(sample_dir / "first_frame.png"), img_sq)
    cv2.imwrite(str(sample_dir / "final_frame.png"), overlay_sq)

    write_mp4(zoom_in_frames(img_bgr), sample_dir / "first_video.mp4", fps=FPS)
    write_mp4(walkthrough_frames(overlay), sample_dir / "last_video.mp4", fps=FPS)
    write_mp4(vessel_reveal_frames(img_bgr, mask), sample_dir / "ground_truth.mp4", fps=FPS)

    (sample_dir / "prompt.txt").write_text(PROMPT, encoding="utf-8")

    mask_area = int(np.count_nonzero(mask))
    total_px = int(mask.shape[0] * mask.shape[1])
    metadata = {
        "task_id": sample_id,
        "generator": "M-114_fives_retinal_vessel_segmentation_data-pipeline",
        "source_dataset": "FIVES",
        "source_sample_id": image_id,
        "augmentation_mode": ["original", "rotate", "hflip", "intensity", "crop"][aug_mode],
        "parameters": {
            "diagnosis": diagnosis,
            "diagnosis_code": diagnosis_code,
            "is_diseased": is_diseased,
            "image_size": {"width": int(img_bgr.shape[1]), "height": int(img_bgr.shape[0])},
            "mask_area_pixels": mask_area,
            "mask_area_ratio": round(mask_area / total_px, 6) if total_px else 0.0,
            "fps": FPS,
            "num_frames": NUM_FRAMES,
            "duration_seconds": round(NUM_FRAMES / FPS, 2),
        },
        "ground_truth": {
            "label": "diseased" if is_diseased else "normal",
            "diagnosis": diagnosis,
            "task_type": "C1_segmentation",
            "target": "retinal_vessel_tree",
        },
        "multimodal_source": {"imaging": "real"},
        "generation": {"seed": 42 + sample_idx, "generator_version": "2.0.0"},
    }
    (sample_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8",
    )
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate M-114 FIVES dataset")
    parser.add_argument("--num-samples", type=int, default=800,
                        help="Total samples to write (default: 800)")
    parser.add_argument("--output", type=str, default="data/questions",
                        help="Output dir (default: data/questions)")
    parser.add_argument("--raw-dir", type=str, default="raw",
                        help="Local dir for raw FIVES download/extract")
    args = parser.parse_args()

    out_dir = Path(args.output)
    raw_dir = Path(args.raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[main] num_samples={args.num_samples} output={out_dir}", flush=True)

    fives_root = ensure_raw(raw_dir)
    pairs = iter_pairs(fives_root)
    print(f"[main] found {len(pairs)} (image, mask) pairs", flush=True)
    if not pairs:
        raise SystemExit("no FIVES pairs found")

    rng = random.Random(42)
    target = args.num_samples
    base_n = len(pairs)

    # Build a deterministic plan: for each output index, pick (pair_idx, aug_mode).
    # If target <= base_n, take a stride through originals (aug_mode=0).
    # If target > base_n, augment up to 5x.
    plan: List[Tuple[int, int]] = []
    if target <= base_n:
        if target <= 0:
            target = base_n
        stride = base_n / target
        for i in range(target):
            plan.append((min(int(i * stride), base_n - 1), 0))
    else:
        for aug in range(5):  # 5 modes total
            for j in range(base_n):
                plan.append((j, aug))
                if len(plan) >= target:
                    break
            if len(plan) >= target:
                break

    written = 0
    skipped = 0
    for sample_idx, (pair_idx, aug_mode) in enumerate(plan):
        img_path, mask_path, image_id, cls = pairs[pair_idx]
        diagnosis, code, diseased = CLASS_LETTER[cls]
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                print(f"  [skip] cannot read {image_id}", flush=True)
                skipped += 1
                continue
            img, mask = downsample(img, mask, target=RESIZE_LONG)
            _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
            img, mask = augment(img, mask, aug_mode, rng)
            ok = write_sample(out_dir, sample_idx, img, mask, image_id,
                              diagnosis, code, diseased, aug_mode)
            if ok:
                written += 1
        except Exception as e:
            print(f"  [skip] sample {sample_idx} ({image_id}): {e}", flush=True)
            traceback.print_exc()
            skipped += 1
            continue
        if (sample_idx + 1) % 10 == 0:
            print(f"  Processed {sample_idx + 1}/{len(plan)} (written={written} skipped={skipped})", flush=True)

    print(f"Done. Wrote {written} samples to {out_dir}/{TASK_DIR_NAME}/ "
          f"(skipped {skipped})", flush=True)


if __name__ == "__main__":
    main()
