# Reproduction Notes

## Dataset reference

The experiments use a project-managed Fashion-Gen shard export.

- Paper: [https://arxiv.org/abs/1806.08317](https://arxiv.org/abs/1806.08317)
- Challenge page: [https://sites.google.com/view/cvcreative/fashion-gen](https://sites.google.com/view/cvcreative/fashion-gen)
- Team shard folder: [Google Drive](https://drive.google.com/drive/folders/1u0ky29PSAIh1Hu02VuA9ufy8fBDt0spn?usp=drive_link)

Expected shard patterns:

- training shards: `clip_dataset_train_aug.*.tar`
- validation shards: `clip_dataset_valid.*.tar`

The repository does not fetch the dataset automatically. Download or sync the shards yourself, then set `data.train_dir`, `data.valid_dir`, `data.train_glob`, and `data.valid_glob` in the config.

## Local workflow

1. Create a Python environment in `python/` and install `requirements.txt`.
2. Download or sync the shard files from the team Drive folder.
3. Fill in `python/configs/fashiongen_experiments.yaml`.
4. Run the experiment suite with `make compare`, `make evaluate`, `make tsne`, and `make ann`.
5. Build writeup assets with `make writeup-assets` or `make writeup-pdf` if needed.

## Slurm workflow

The repository includes a generic wrapper at `python/hpc/submit_report_suite.sbatch.sh`.

Required environment variables:

- `PROJECT_ROOT`: repository root
- `CONFIG_PATH`: path to `fashiongen_experiments.yaml`
- `PYTHON_BIN`: Python interpreter inside your environment
- `TASKS`: optional task list, for example `compare evaluate tsne ann`

Example:

```bash
export PROJECT_ROOT=$PWD
export CONFIG_PATH=$PWD/python/configs/fashiongen_experiments.yaml
export PYTHON_BIN=$PWD/python/.venv/bin/python
export TASKS="compare evaluate tsne ann"
sbatch python/hpc/submit_report_suite.sbatch.sh
```
