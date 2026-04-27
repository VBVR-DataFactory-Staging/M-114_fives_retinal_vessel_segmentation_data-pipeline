# M-114 — FIVES Retinal Vessel Segmentation

Generates VBVR (Very Big Video Reasoning) tasks from the FIVES public
fundus-image dataset (801 high-resolution colour fundus photos with
expert-annotated binary vessel masks).

## Task

Prompt shown to the model:

> Segment all blood vessels in this fundus image. Trace the complete
> retinal vasculature including arteries, veins, and their branches;
> highlight every vessel pixel and ignore background, optic disc, and
> macular regions.

Each sample produces a 5-second vessel-reveal animation as
`ground_truth.mp4` (30 frames at 6 fps), with the un-annotated fundus
acting as `first_frame.png` / `first_video.mp4` and the red overlay as
`final_frame.png` / `last_video.mp4`.

## Source

FIVES — *A Fundus Image Dataset for AI-based Vessel Segmentation*
(Figshare DOI 10.6084/m9.figshare.19688169). Mirrored to:

```
s3://med-vr-datasets/M-114/fives/34969398    # 1.76 GB .rar
```

The pipeline downloads the .rar from S3 on the EC2 worker, extracts it
with `unar` (apt: `unar`; the non-free `unrar` is intentionally avoided),
then iterates the test split (200 imgs) followed by the train split
(601 imgs).

## Output Layout

```
data/questions/fives_retinal_vessel_segmentation_task/
├── m114_00000_test_100_D/
│   ├── first_frame.png
│   ├── final_frame.png
│   ├── prompt.txt
│   ├── first_video.mp4
│   ├── last_video.mp4
│   ├── ground_truth.mp4
│   └── metadata.json
├── m114_00001_test_101_G/
└── ...
```

The `_<class>` suffix encodes the FIVES disease label:
A=AMD, D=DR, G=Glaucoma, N=Normal.

## Quick Start

```bash
pip install -r requirements.txt
sudo apt-get install -y unar ffmpeg          # Linux deps

# 3-sample smoke test
python examples/generate.py --num-samples 3

# Full run (all 801 samples)
python examples/generate.py
```

## Configuration

`src/pipeline/config.py` (`TaskConfig`):

| Field | Default | Description |
|---|---|---|
| `domain` | `"fives_retinal_vessel_segmentation"` | Task domain string used in output paths. |
| `s3_bucket` | `"med-vr-datasets"` | S3 bucket containing the raw rar. |
| `s3_prefix` | `"M-114/fives/"` | S3 prefix for the FIVES rar. |
| `rar_key` | `"M-114/fives/34969398"` | Exact S3 key of the rar archive. |
| `fps` | `6` | Output video FPS. |
| `num_video_frames` | `30` | Frames per clip (5 s at 6 fps). |
| `splits` | `("test","train")` | FIVES splits to iterate. |
| `num_samples` | `None` | Max samples (None = all 801). |

## Repository Structure

```
M-114_fives_retinal_vessel_segmentation_data-pipeline/
├── core/                ← shared pipeline framework
├── eval/                ← shared evaluation utilities
├── src/
│   ├── download/
│   │   └── downloader.py   ← S3 rar fetch + unar extract
│   └── pipeline/
│       ├── config.py        ← task config
│       ├── pipeline.py      ← TaskPipeline (image+mask → 7-file sample)
│       └── transforms.py    ← overlay + vessel-reveal animation
├── examples/
│   └── generate.py
├── requirements.txt
├── README.md
└── LICENSE
```
