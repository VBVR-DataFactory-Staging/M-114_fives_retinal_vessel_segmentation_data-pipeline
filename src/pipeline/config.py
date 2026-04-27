"""Pipeline configuration for M-114 (FIVES retinal vessel segmentation)."""
from pathlib import Path
from pydantic import Field
from core.pipeline import PipelineConfig


class TaskConfig(PipelineConfig):
    """Configuration for the M-114 FIVES retinal vessel segmentation pipeline."""

    domain: str = Field(default="fives_retinal_vessel_segmentation")

    s3_bucket: str = Field(
        default="med-vr-datasets",
        description="S3 bucket containing the raw FIVES rar archive",
    )
    s3_prefix: str = Field(
        default="M-114/fives/",
        description="S3 key prefix for the FIVES rar archive",
    )
    rar_key: str = Field(
        default="M-114/fives/34969398",
        description=(
            "S3 key of the FIVES rar archive (Figshare distributes the dataset "
            "as a single .rar). Both 34969398 and the URL-encoded twin map to "
            "the same 1.6GB archive."
        ),
    )
    fps: int = Field(
        default=6,
        description="Frames per second for generated videos (6 fps x 30 frames = 5s).",
    )
    num_video_frames: int = Field(
        default=30,
        description="Number of frames in each generated video clip.",
    )
    raw_dir: Path = Field(
        default=Path("raw"),
        description="Local directory for the extracted FIVES dataset",
    )
    splits: tuple = Field(
        default=("test", "train"),
        description=(
            "Which FIVES splits to iterate. Test (200 imgs) is preferred first; "
            "train (601 imgs) covers the rest."
        ),
    )
    task_prompt: str = Field(
        default=(
            "Segment all blood vessels in this fundus image. Trace the complete "
            "retinal vasculature including arteries, veins, and their branches; "
            "highlight every vessel pixel and ignore background, optic disc, "
            "and macular regions."
        ),
        description="Instruction shown to the reasoning model.",
    )
