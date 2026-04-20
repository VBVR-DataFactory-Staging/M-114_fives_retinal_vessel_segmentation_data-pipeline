# M-114 scaffold TODO

Scaffolded from template: `M-042_ddr_lesion_segmentation_data-pipeline` (2026-04-20)

## Status
- [x] config.py updated (domain=fives_retinal_vessel_segmentation, s3_prefix=M-114_FIVES/raw/, fps=3)
- [ ] core/download.py: update URL / Kaggle slug / HF repo_id
- [ ] src/download/downloader.py: adapt to dataset file layout
- [ ] src/pipeline/_phase2/*.py: adapt raw → frames logic (inherited from M-042_ddr_lesion_segmentation_data-pipeline, likely needs rework)
- [ ] examples/generate.py: verify end-to-end on 3 samples

## Task prompt
This color fundus photograph (FIVES). Segment the retinal vessel network (red).

Fleet runs likely FAIL on first attempt for dataset parsing; iterate based on fleet logs at s3://vbvr-final-data/_logs/.
