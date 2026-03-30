# Python workspace

## Main entrypoints

- `scripts/run_report_suite.py`: runs the FashionGen experiment suite from one YAML config
- `scripts/prepare_report_assets.py`: converts retained results into LaTeX-ready macros and figures
- `train_joint_clip.py`: core training loop

## Experiment config

Edit `configs/fashiongen_experiments.yaml` and fill in:

- `data.train_dir`
- `data.valid_dir`
- `data.train_glob`
- `data.valid_glob`
- `model.cache_dir`
- `checkpoints.retrieval`
- `checkpoints.joint`

Dataset source:

- team shard folder: [Google Drive](https://drive.google.com/drive/folders/1u0ky29PSAIh1Hu02VuA9ufy8fBDt0spn?usp=drive_link)

Expected shard patterns:

- training: `clip_dataset_train_aug.*.tar`
- validation: `clip_dataset_valid.*.tar`

Then run:

```bash
python scripts/run_report_suite.py --config configs/fashiongen_experiments.yaml --tasks compare evaluate tsne ann
```
