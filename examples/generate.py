#!/usr/bin/env python3
"""Dataset generation entry point for M-114 (FIVES retinal vessel segmentation).

Usage:
    python examples/generate.py
    python examples/generate.py --num-samples 3
    python examples/generate.py --output data/my_output
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import TaskPipeline, TaskConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the M-114 FIVES retinal vessel segmentation dataset",
    )
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Max samples to generate (default: 200; use 0 for all 801).")
    parser.add_argument("--output", type=str, default="data/questions",
                        help="Output directory (default: data/questions).")
    args = parser.parse_args()

    print("Generating M-114 FIVES retinal vessel segmentation dataset...")
    num_samples = None if args.num_samples in (0, -1) else args.num_samples
    config = TaskConfig(
        num_samples=num_samples,
        output_dir=Path(args.output),
    )
    pipeline = TaskPipeline(config)
    samples = pipeline.run()
    print(f"Wrote {len(samples)} samples to {args.output}/")


if __name__ == "__main__":
    main()
