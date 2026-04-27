"""Transforms for the M-114 FIVES retinal vessel segmentation pipeline.

Provides:
  - vessel mask overlay (red on the fundus)
  - "vessel reveal" animation: gradually paint the binary mask onto the
    image so vessels appear progressively, simulating an annotation
    play-through; used for ``ground_truth.mp4``
  - subtle camera-shake on the un-annotated fundus, used for
    ``first_video.mp4`` / ``last_video.mp4`` so they are non-trivial videos
  - ffmpeg-based mp4 writer (libx264, browser-compatible)
"""
from __future__ import annotations
import math
import subprocess
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


VESSEL_COLOR_BGR = (0, 0, 255)  # red — matches the prompt wording.


def create_overlay(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = VESSEL_COLOR_BGR,
    alpha: float = 0.55,
) -> np.ndarray:
    """Blend a binary vessel mask onto the fundus image with a contour edge."""
    colored = np.zeros_like(img_bgr)
    colored[mask > 0] = color
    blended = cv2.addWeighted(img_bgr, 1.0 - alpha, colored, alpha, 0.0)

    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(blended, contours, -1, color, 1)
    return blended


def vessel_reveal_frames(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    num_frames: int = 30,
    color: Tuple[int, int, int] = VESSEL_COLOR_BGR,
) -> List[np.ndarray]:
    """Animation that gradually paints in the vessel mask.

    Frame t shows the fundus image overlaid with the subset of vessel
    pixels whose horizontal coordinate is below a sweeping threshold
    ``progress * width``. The final 20% of frames hold the fully revealed
    overlay so the last frame matches ``final_frame.png``.
    """
    frames: List[np.ndarray] = []
    h, w = img_bgr.shape[:2]
    binary = (mask > 0).astype(np.uint8)
    hold_frames = max(1, num_frames // 5)
    sweep_frames = max(1, num_frames - hold_frames)
    ramp = np.tile(np.arange(w, dtype=np.float32), (h, 1)) / max(w - 1, 1)

    for i in range(num_frames):
        if i < sweep_frames:
            progress = (i + 1) / sweep_frames
            partial = (ramp <= progress).astype(np.uint8) * binary
        else:
            partial = binary
        frame = create_overlay(img_bgr, partial * 255, color=color, alpha=0.55)
        frames.append(frame)
    return frames


def fundus_motion_frames(
    img_bgr: np.ndarray,
    num_frames: int = 30,
) -> List[np.ndarray]:
    """Subtle camera-shake on the un-annotated fundus (for first_video.mp4)."""
    frames: List[np.ndarray] = []
    h, w = img_bgr.shape[:2]
    for i in range(num_frames):
        dx = int(4 * math.sin(2 * math.pi * i / num_frames))
        dy = int(3 * math.cos(2 * math.pi * i / num_frames))
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(img_bgr, m, (w, h), borderMode=cv2.BORDER_REFLECT)
        frames.append(shifted)
    return frames


def annotated_motion_frames(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    num_frames: int = 30,
) -> List[np.ndarray]:
    """Same camera-shake but on the annotated overlay (for last_video.mp4)."""
    overlay = create_overlay(img_bgr, mask, alpha=0.55)
    return fundus_motion_frames(overlay, num_frames=num_frames)


def make_video(frames: List[np.ndarray], out_path, fps: int = 6) -> None:
    """Write BGR frames to MP4 via ffmpeg (libx264, browser-friendly).

    Replaces cv2.VideoWriter('avc1') which silently fails on Linux.
    """
    if not frames:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = frames[0].shape[:2]
    w2 = w - (w % 2)
    h2 = h - (h % 2)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-vf", f"scale={w2}:{h2}",
        str(out_path),
    ]

    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h))
        if f.ndim == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        p.stdin.write(f.tobytes())
    p.stdin.close()
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed (rc={rc}) writing {out_path}")
