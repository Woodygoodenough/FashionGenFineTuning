#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a FashionGen joint-loss lambda sweep against a fixed retrieval baseline.")
    p.add_argument("--python-bin", type=Path, required=True)
    p.add_argument("--train-script", type=Path, required=True)
    p.add_argument("--baseline-comparison-json", type=Path, required=True)
    p.add_argument("--shards-dir", type=Path, required=True)
    p.add_argument("--valid-shards-dir", type=Path, required=True)
    p.add_argument("--model-cache-dir", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--weights", type=str, default="0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10")
    p.add_argument("--keep-top-k", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--min-epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--early-stop-min-delta", type=float, default=0.0)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-max-batches", type=int, default=125)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def setup_logger(run_root: Path) -> logging.Logger:
    run_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fg_lambda_sweep")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    fh = logging.FileHandler(run_root / "sweep.log")
    sh.setFormatter(fmt)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.info("Writing sweep log to %s", run_root / "sweep.log")
    return logger


def load_baseline(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    baseline = payload["retrieval_only"]
    baseline["source_comparison"] = path.name
    return baseline


def find_best_eval(run_dir: Path) -> dict[str, Any]:
    eval_files = sorted(run_dir.glob("eval_*.jsonl"))
    if not eval_files:
        raise FileNotFoundError(f"No eval JSONL found in {run_dir}")
    best = None
    best_score = float("-inf")
    chosen = None
    for path in eval_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            score = float(rec.get("score", float("-inf")))
            if score > best_score:
                best_score = score
                best = rec
                chosen = path
    if best is None or chosen is None:
        raise RuntimeError(f"No eval records in {run_dir}")
    return {"eval_file": chosen.name, "best_eval": best}


def find_checkpoint(run_dir: Path) -> str | None:
    ckpts = sorted(run_dir.glob("joint_clip_step_*.pt"))
    return ckpts[-1].name if ckpts else None


def variant_name(weight: float) -> str:
    return f"joint_w{weight:.2f}".replace(".", "p")


def run_variant(args: argparse.Namespace, logger: logging.Logger, weight: float) -> dict[str, Any]:
    name = variant_name(weight)
    run_dir = args.run_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.python_bin),
        str(args.train_script),
        "--shards-dir", str(args.shards_dir),
        "--shards-glob", "clip_dataset_train_aug.*.tar",
        "--valid-shards-dir", str(args.valid_shards_dir),
        "--valid-shards-glob", "clip_dataset_valid.*.tar",
        "--model-name", "ViT-B-16",
        "--pretrained", "laion2b_s34b_b88k",
        "--model-cache-dir", str(args.model_cache_dir),
        "--loss-type", "contrastive",
        "--align-weight", "1.0",
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--epochs", str(args.epochs),
        "--min-epochs", str(args.min_epochs),
        "--patience", str(args.patience),
        "--early-stop-min-delta", str(args.early_stop_min_delta),
        "--seed", str(args.seed),
        "--eval-every", str(args.eval_every),
        "--eval-max-batches", str(args.eval_max_batches),
        "--log-dir", str(run_dir),
        "--log-every", str(args.log_every),
        "--cls-head-type", "mlp",
        "--cls-feature", "img_raw",
        "--cls-grad", "shared",
        "--cls-loss-type", "ce_ls",
        "--cls-weight", str(weight),
        "--cls-weight-schedule", "warmup",
        "--cls-weight-warmup-steps", "500",
        "--cls-hidden-dim", "1024",
        "--cls-dropout", "0.1",
        "--cls-label-smoothing", "0.05",
        "--two-stage-eval", "off",
        "--save-checkpoint",
    ]
    if args.max_steps is not None:
        cmd.extend(["--max-steps", str(args.max_steps)])
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"variant_name": name, "cls_weight": weight, "cmd": cmd}, indent=2),
        encoding="utf-8",
    )
    logger.info("$ %s", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    (run_dir / "launcher_stdout.log").write_text(proc.stdout, encoding="utf-8")
    result = find_best_eval(run_dir)
    result["run_dir"] = run_dir.name
    result["checkpoint"] = find_checkpoint(run_dir)
    result["cls_weight"] = weight
    best = result["best_eval"]
    logger.info(
        "%s best score=%.4f avg_r1=%.4f avg_r5=%.4f avg_r10=%.4f step=%s",
        name,
        float(best["score"]),
        float(best["avg_r1"]),
        float(best["avg_r5"]),
        float(best["avg_r10"]),
        best.get("step"),
    )
    return result


def prune_checkpoints(results: list[dict[str, Any]], run_root: Path, keep_top_k: int) -> list[str]:
    ranked = sorted(results, key=lambda r: float(r["best_eval"]["score"]), reverse=True)
    keep = {entry["run_dir"] for entry in ranked[:keep_top_k]}
    kept_paths: list[str] = []
    for entry in results:
        ckpt = entry.get("checkpoint")
        if not ckpt:
            continue
        ckpt_path = run_root / entry["run_dir"] / ckpt
        if entry["run_dir"] in keep:
            kept_paths.append(str(ckpt_path))
        elif ckpt_path.exists():
            ckpt_path.unlink()
    return kept_paths


def main() -> int:
    args = parse_args()
    weights = [float(x) for x in args.weights.split(",") if x.strip()]
    if args.smoke:
        weights = weights[:2]
    label = "smoke" if args.smoke else "lambda_sweep"
    args.run_root = args.run_root / f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = setup_logger(args.run_root)

    baseline = load_baseline(args.baseline_comparison_json)
    base = baseline["best_eval"]
    logger.info(
        "Baseline score=%.4f avg_r1=%.4f avg_r5=%.4f avg_r10=%.4f",
        float(base["score"]),
        float(base["avg_r1"]),
        float(base["avg_r5"]),
        float(base["avg_r10"]),
    )

    results = [run_variant(args, logger, weight) for weight in weights]
    kept_checkpoints = prune_checkpoints(results, args.run_root, args.keep_top_k)
    kept_set = {Path(p).name for p in kept_checkpoints}
    ranked = sorted(
        [
            {
                "cls_weight": item["cls_weight"],
                "score": float(item["best_eval"]["score"]),
                "avg_r1": float(item["best_eval"]["avg_r1"]),
                "avg_r5": float(item["best_eval"]["avg_r5"]),
                "avg_r10": float(item["best_eval"]["avg_r10"]),
                "delta_score": float(item["best_eval"]["score"] - base["score"]),
                "delta_avg_r1": float(item["best_eval"]["avg_r1"] - base["avg_r1"]),
                "delta_avg_r5": float(item["best_eval"]["avg_r5"] - base["avg_r5"]),
                "delta_avg_r10": float(item["best_eval"]["avg_r10"] - base["avg_r10"]),
                "checkpoint": item.get("checkpoint") if item.get("checkpoint") in kept_set else None,
                "run_dir": item["run_dir"],
                "best_step": item["best_eval"].get("step"),
                "best_epoch": item["best_eval"].get("epoch"),
            }
            for item in results
        ],
        key=lambda r: r["score"],
        reverse=True,
    )

    summary = {
        "config": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "min_epochs": args.min_epochs,
            "patience": args.patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "max_steps": args.max_steps,
            "eval_every": args.eval_every,
            "eval_max_batches": args.eval_max_batches,
            "weights": weights,
            "keep_top_k": args.keep_top_k,
            "baseline_comparison_json": args.baseline_comparison_json.name,
        },
        "baseline": baseline,
        "ranked_results": ranked,
        "kept_checkpoints": kept_checkpoints,
    }
    (args.run_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        f"baseline score={base['score']:.4f} avg_r1={base['avg_r1']:.4f} avg_r5={base['avg_r5']:.4f} avg_r10={base['avg_r10']:.4f}",
        "rank cls_weight score delta_score avg_r1 avg_r5 avg_r10 checkpoint",
    ]
    for idx, row in enumerate(ranked, start=1):
        lines.append(
            f"{idx:02d} {row['cls_weight']:.2f} {row['score']:.4f} {row['delta_score']:+.4f} "
            f"{row['avg_r1']:.4f} {row['avg_r5']:.4f} {row['avg_r10']:.4f} "
            f"{row['checkpoint'] or '-'}"
        )
    (args.run_root / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote summary to %s", args.run_root / "summary.json")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
