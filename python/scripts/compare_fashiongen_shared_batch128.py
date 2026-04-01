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
    p = argparse.ArgumentParser(description='Run controlled FashionGen retrieval-only vs joint comparison at batch size 128.')
    p.add_argument('--python-bin', type=Path, required=True)
    p.add_argument('--train-script', type=Path, required=True)
    p.add_argument('--shards-dir', type=Path, required=True)
    p.add_argument('--valid-shards-dir', type=Path, required=True)
    p.add_argument('--model-cache-dir', type=Path, required=True)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--min-epochs', type=int, default=5)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--early-stop-min-delta', type=float, default=0.0)
    p.add_argument('--max-steps', type=int, default=None)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--eval-max-batches', type=int, default=125)
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--num-workers', type=int, default=1)
    p.add_argument('--lr', type=float, default=5e-6)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--joint-cls-weight', type=float, default=0.05)
    p.add_argument('--smoke', action='store_true')
    return p.parse_args()


def setup_logger(run_root: Path) -> logging.Logger:
    run_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('fg128_compare')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    log_path = run_root / 'compare.log'
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    fh = logging.FileHandler(log_path)
    sh.setFormatter(fmt)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.info('Writing comparison log to %s', log_path)
    return logger


def find_best_eval(run_dir: Path) -> dict[str, Any]:
    eval_files = sorted(run_dir.glob('eval_*.jsonl'))
    if not eval_files:
        raise FileNotFoundError(f'No eval JSONL found in {run_dir}')
    best = None
    best_score = float('-inf')
    chosen = None
    for path in eval_files:
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            score = float(rec.get('score', float('-inf')))
            if score > best_score:
                best_score = score
                best = rec
                chosen = path
    if best is None or chosen is None:
        raise RuntimeError(f'No eval records in {eval_files}')
    return {'eval_file': chosen.name, 'best_eval': best}


def find_checkpoint(run_dir: Path) -> str | None:
    ckpts = sorted(run_dir.glob('joint_clip_step_*.pt'))
    return ckpts[-1].name if ckpts else None


def run_variant(args: argparse.Namespace, logger: logging.Logger, variant_name: str, cls_weight: float) -> dict[str, Any]:
    variant_dir = args.run_root / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.python_bin),
        str(args.train_script),
        '--shards-dir', str(args.shards_dir),
        '--shards-glob', 'clip_dataset_train_aug.*.tar',
        '--valid-shards-dir', str(args.valid_shards_dir),
        '--valid-shards-glob', 'clip_dataset_valid.*.tar',
        '--model-name', 'ViT-B-16',
        '--pretrained', 'laion2b_s34b_b88k',
        '--model-cache-dir', str(args.model_cache_dir),
        '--loss-type', 'contrastive',
        '--align-weight', '1.0',
        '--batch-size', str(args.batch_size),
        '--num-workers', str(args.num_workers),
        '--lr', str(args.lr),
        '--weight-decay', str(args.weight_decay),
        '--epochs', str(args.epochs),
        '--min-epochs', str(args.min_epochs),
        '--patience', str(args.patience),
        '--early-stop-min-delta', str(args.early_stop_min_delta),
        '--seed', str(args.seed),
        '--eval-every', str(args.eval_every),
        '--eval-max-batches', str(args.eval_max_batches),
        '--log-dir', str(variant_dir),
        '--log-every', str(args.log_every),
        '--cls-head-type', 'mlp',
        '--cls-feature', 'img_raw',
        '--cls-grad', 'shared',
        '--cls-loss-type', 'ce_ls',
        '--cls-weight', str(cls_weight),
        '--cls-weight-schedule', 'warmup',
        '--cls-weight-warmup-steps', '500',
        '--cls-hidden-dim', '1024',
        '--cls-dropout', '0.1',
        '--cls-label-smoothing', '0.05',
        '--two-stage-eval', 'off',
        '--save-checkpoint',
    ]
    if args.max_steps is not None:
        cmd.extend(['--max-steps', str(args.max_steps)])
    (variant_dir / 'run_manifest.json').write_text(json.dumps({'variant_name': variant_name, 'cls_weight': cls_weight, 'cmd': cmd}, indent=2), encoding='utf-8')
    logger.info('$ %s', ' '.join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    (variant_dir / 'launcher_stdout.log').write_text(proc.stdout, encoding='utf-8')
    result = find_best_eval(variant_dir)
    best = result['best_eval']
    logger.info('%s best score=%.4f avg_r1=%.4f avg_r5=%.4f avg_r10=%.4f step=%s', variant_name, float(best['score']), float(best['avg_r1']), float(best['avg_r5']), float(best['avg_r10']), best.get('step'))
    result['run_dir'] = variant_dir.name
    result['checkpoint'] = find_checkpoint(variant_dir)
    return result


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    label = 'smoke' if args.smoke else 'full'
    args.run_root = args.run_root / f'{label}_{timestamp}'
    logger = setup_logger(args.run_root)
    retrieval = run_variant(args, logger, 'retrieval_only', 0.0)
    joint = run_variant(args, logger, 'joint_shared', args.joint_cls_weight)

    base = retrieval['best_eval']
    joint_best = joint['best_eval']
    comparison = {
        'config': {
            'seed': args.seed,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'min_epochs': args.min_epochs,
            'patience': args.patience,
            'early_stop_min_delta': args.early_stop_min_delta,
            'max_steps': args.max_steps,
            'eval_every': args.eval_every,
            'eval_max_batches': args.eval_max_batches,
            'joint_cls_weight': args.joint_cls_weight,
        },
        'retrieval_only': retrieval,
        'joint_shared': joint,
        'delta': {
            'avg_r1': float(joint_best['avg_r1'] - base['avg_r1']),
            'avg_r5': float(joint_best['avg_r5'] - base['avg_r5']),
            'avg_r10': float(joint_best['avg_r10'] - base['avg_r10']),
            'score': float(joint_best['score'] - base['score']),
        },
    }
    (args.run_root / 'comparison.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
    lines = [
        f"batch_size={args.batch_size} eval_max_batches={args.eval_max_batches} epochs={args.epochs} max_steps={args.max_steps}",
        f"retrieval_only score={base['score']:.4f} avg_r1={base['avg_r1']:.4f} avg_r5={base['avg_r5']:.4f} avg_r10={base['avg_r10']:.4f}",
        f"joint_shared cls_weight={args.joint_cls_weight:.2f} score={joint_best['score']:.4f} avg_r1={joint_best['avg_r1']:.4f} avg_r5={joint_best['avg_r5']:.4f} avg_r10={joint_best['avg_r10']:.4f}",
        f"delta score={comparison['delta']['score']:+.4f} avg_r1={comparison['delta']['avg_r1']:+.4f} avg_r5={comparison['delta']['avg_r5']:+.4f} avg_r10={comparison['delta']['avg_r10']:+.4f}",
    ]
    (args.run_root / 'comparison.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    logger.info('Wrote comparison to %s', args.run_root / 'comparison.json')
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
