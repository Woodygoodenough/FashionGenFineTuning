#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the FashionGen experiment suite.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--tasks",
        nargs="+",
        choices=["compare", "evaluate", "tsne", "ann", "demo"],
        default=["compare", "evaluate", "tsne", "ann"],
    )
    return p.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path_value: str | Path, base: Path) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else (base / p).resolve()


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    cfg_path = args.config.resolve()
    cfg = load_config(cfg_path)
    cfg_dir = cfg_path.parent

    model = cfg["model"]
    data = cfg["data"]
    ckpt = cfg["checkpoints"]
    runtime = cfg["runtime"]
    compare = cfg["compare"]
    ann = cfg["ann"]
    tsne = cfg["tsne"]
    out_root = resolve(cfg["outputs"]["root"], cfg_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    python_bin = str(runtime.get("python_bin", sys.executable))
    train_script = ROOT / "train_joint_clip.py"

    if "compare" in args.tasks:
        run([
            python_bin,
            str(ROOT / "scripts" / "compare_fashiongen_shared_batch128.py"),
            "--python-bin", python_bin,
            "--train-script", str(train_script),
            "--shards-dir", str(resolve(data["train_dir"], cfg_dir)),
            "--valid-shards-dir", str(resolve(data["valid_dir"], cfg_dir)),
            "--model-cache-dir", str(resolve(model["cache_dir"], cfg_dir)),
            "--run-root", str(out_root / "compare"),
            "--seed", str(runtime["seed"]),
            "--batch-size", str(runtime["batch_size"]),
            "--epochs", str(compare["epochs"]),
            "--max-steps", str(compare["max_steps"]),
            "--eval-every", str(compare["eval_every"]),
            "--eval-max-batches", str(compare["eval_max_batches"]),
            "--num-workers", str(runtime["num_workers"]),
            "--lr", str(compare["lr"]),
            "--weight-decay", str(compare["weight_decay"]),
            "--joint-cls-weight", str(compare["joint_cls_weight"]),
        ])

    if "evaluate" in args.tasks:
        run([
            python_bin,
            str(ROOT / "scripts" / "evaluate_fashiongen_base_vs_finetuned.py"),
            "--valid-shards-dir", str(resolve(data["valid_dir"], cfg_dir)),
            "--valid-shards-glob", data["valid_glob"],
            "--model-name", model["name"],
            "--pretrained", model["pretrained"],
            "--model-cache-dir", str(resolve(model["cache_dir"], cfg_dir)),
            "--finetuned-checkpoint", str(resolve(ckpt["retrieval"], cfg_dir)),
            "--batch-size", str(runtime["batch_size"]),
            "--num-workers", str(runtime["num_workers"]),
            "--eval-max-batches", str(compare["eval_max_batches"]),
            "--out-dir", str(out_root / "evaluate"),
        ])

    if "tsne" in args.tasks:
        run([
            python_bin,
            str(ROOT / "scripts" / "analyze_fashiongen_tsne_triplet.py"),
            "--shards-dir", str(resolve(data["train_dir"], cfg_dir)),
            "--manifest", str(resolve(tsne["manifest"], cfg_dir)),
            "--model-name", model["name"],
            "--pretrained", model["pretrained"],
            "--model-cache-dir", str(resolve(model["cache_dir"], cfg_dir)),
            "--retrieval-checkpoint", str(resolve(ckpt["retrieval"], cfg_dir)),
            "--joint-checkpoint", str(resolve(ckpt["joint"], cfg_dir)),
            "--batch-size", str(runtime["batch_size"]),
            "--out-dir", str(out_root / "tsne"),
        ])

    if "ann" in args.tasks:
        run([
            python_bin,
            str(ROOT / "scripts" / "benchmark_fashiongen_ann_compare.py"),
            "--shards-dir", str(resolve(data["train_dir"], cfg_dir)),
            "--shards-glob", data["train_glob"],
            "--model-name", model["name"],
            "--pretrained", model["pretrained"],
            "--model-cache-dir", str(resolve(model["cache_dir"], cfg_dir)),
            "--finetuned-checkpoint", str(resolve(ckpt["retrieval"], cfg_dir)),
            "--batch-size", str(runtime["batch_size"]),
            "--num-workers", str(runtime["num_workers"]),
            "--sample-count", str(ann["sample_count"]),
            "--sample-size", str(ann["sample_size"]),
            "--query-count", str(ann["query_count"]),
            "--k", str(ann["k"]),
            "--hnsw-m", str(ann["hnsw_m"]),
            "--ef-construction", str(ann["ef_construction"]),
            "--ef-search", str(ann["ef_search"]),
            "--rerank-k", str(ann["rerank_k"]),
            "--seed", str(runtime["seed"]),
            "--out-dir", str(out_root / "ann"),
        ])

    if "demo" in args.tasks:
        run([
            python_bin,
            str(ROOT / "scripts" / "build_prototype_data.py"),
            "--checkpoint-path", str(resolve(ckpt["retrieval"], cfg_dir)),
        ])


if __name__ == "__main__":
    main()
