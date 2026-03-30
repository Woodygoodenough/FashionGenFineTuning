# Fashion Multimodal Retrieval

This repository contains code and retained experiment outputs for a FashionGen-based multimodal retrieval project. It focuses on reproducible training, evaluation, and analysis:

- controlled FashionGen training runs for retrieval-only vs joint loss,
- zero-shot vs fine-tuned evaluation,
- ANN latency/accuracy benchmarking,
- embedding-space analysis.

## Repository layout

- `python/`: training, evaluation, ANN, and t-SNE code
- `artifacts/results/`: compact experiment outputs kept for reproducibility
- `docs/`: reproduction notes

## Dataset

The experiments assume access to a preprocessed Fashion-Gen shard export owned by the project team.

- Paper: [https://arxiv.org/abs/1806.08317](https://arxiv.org/abs/1806.08317)
- Challenge page: [https://sites.google.com/view/cvcreative/fashion-gen](https://sites.google.com/view/cvcreative/fashion-gen)
- Team shard folder: [Google Drive](https://drive.google.com/drive/folders/1u0ky29PSAIh1Hu02VuA9ufy8fBDt0spn?usp=drive_link)

This repository does not download or redistribute Fashion-Gen automatically. The workflow assumes you already have the WebDataset tar shards available locally. The shard names used in our runs are:

- training: `clip_dataset_train_aug.*.tar`
- validation: `clip_dataset_valid.*.tar`

Point the config at the directories that contain those files and adjust `train_glob` / `valid_glob` if your local naming differs.

## Quick start

### Python

```bash
cd python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the experiments

1. Download or sync the shard files from the team Drive folder into local directories for train and validation data.
2. Edit `python/configs/fashiongen_experiments.yaml` with your dataset paths, shard globs, cache directory, and checkpoints.
3. Run one task at a time or the full suite:

```bash
make compare
make evaluate
make tsne
make ann
make experiment-suite
make writeup-assets
make writeup-pdf
```

The same workflow can be submitted to any Slurm cluster with `python/hpc/submit_report_suite.sbatch.sh` by setting `PROJECT_ROOT`, `CONFIG_PATH`, and `PYTHON_BIN`.

## Retained results

The repository keeps a compact set of outputs under `artifacts/results/` so evaluation tables and figures can be regenerated without preserving every exploratory run.
