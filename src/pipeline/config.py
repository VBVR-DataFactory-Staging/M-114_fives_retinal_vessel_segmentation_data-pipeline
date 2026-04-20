"""Pipeline configuration for M-042 (ddr_lesion_segmentation)."""
from pathlib import Path
from pydantic import Field
from core.pipeline import PipelineConfig


class TaskConfig(PipelineConfig):
    """Configuration for M-042 pipeline.

    Inherited from PipelineConfig:
        num_samples: Optional[int]  # Max samples (None = all)
        domain: str
        output_dir: Path
        split: str
    """

    domain: str = Field(default="fives_retinal_vessel_segmentation")

    s3_bucket: str = Field(
        default="med-vr-datasets",
        description="S3 bucket containing the raw M-042 data",
    )
    s3_prefix: str = Field(
        default="M-114_FIVES/raw/",
        description="S3 key prefix for the dataset raw data",
    )
    fps: int = Field(
        default=3,
        description="Frames per second for the generated videos",
    )
    raw_dir: Path = Field(
        default=Path("raw"),
        description="Local directory for downloaded raw data",
    )
    task_prompt: str = Field(
        default="This color fundus photograph (FIVES). Segment the retinal vessel network (red).",
        description="The task instruction shown to the reasoning model.",
    )
