#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import open_clip
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_joint_clip import JointCLIPModel, create_loader, resolve_shards, run_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate FashionGen retrieval on a fixed validation subset for both the public "
            "zero-shot OpenCLIP checkpoint and the local fine-tuned checkpoint. "
            "This yields a clean before/after comparison on the same split."
        )
    )
    parser.add_argument("--valid-shards-dir", type=Path, required=True)
    parser.add_argument("--valid-shards-glob", default="clip_dataset_valid.*.tar")
    parser.add_argument("--valid-max-shards", type=int, default=None)
    parser.add_argument("--model-name", default="ViT-B-16")
    parser.add_argument("--pretrained", default="laion2b_s34b_b88k")
    parser.add_argument("--model-cache-dir", type=Path, required=True)
    parser.add_argument("--finetuned-checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--eval-max-batches", type=int, default=125)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def evaluate_variant(
    *,
    variant_name: str,
    checkpoint: Path | None,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    clip_model, preprocess, _ = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained,
        cache_dir=str(args.model_cache_dir),
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)

    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        clip_model.load_state_dict(state_dict)

    emb_dim = int(getattr(clip_model.text_projection, "shape", [0, 0])[1])
    model = JointCLIPModel(
        clip_model=clip_model,
        emb_dim=emb_dim,
        num_classes=1,
        cls_head_type="linear",
        cls_hidden_dim=emb_dim,
        cls_dropout=0.0,
        cls_feature="img_norm",
    ).to(device)

    valid_shards = resolve_shards(args.valid_shards_dir, args.valid_shards_glob, args.valid_max_shards)
    loader = create_loader(
        shards=valid_shards,
        image_key="jpg",
        caption_keys=["txt", "txt0.txt"],
        meta_key="meta.json",
        preprocess=preprocess,
        category_to_id={},
        caption_mode="primary",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shardshuffle=False,
    )

    metrics = run_validation(
        model=model,
        tokenizer=tokenizer,
        loader=loader,
        device=device,
        loss_type="contrastive",
        logit_bias=None,
        max_batches=args.eval_max_batches,
        cls_loss_type="ce",
        cls_label_smoothing=0.0,
        cls_focal_gamma=2.0,
        class_weights=None,
        two_stage_mode="off",
        two_stage_topk_categories=1,
        two_stage_alpha=0.0,
        two_stage_text_temp=1.0,
        category_text_prototypes=None,
    )
    metrics["variant"] = variant_name
    return metrics


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_metrics = evaluate_variant(
        variant_name="zero_shot",
        checkpoint=None,
        args=args,
        device=device,
    )
    ft_metrics = evaluate_variant(
        variant_name="finetuned",
        checkpoint=args.finetuned_checkpoint,
        args=args,
        device=device,
    )

    comparison = {
        "config": {
            "model_name": args.model_name,
            "pretrained": args.pretrained,
            "eval_max_batches": args.eval_max_batches,
            "batch_size": args.batch_size,
        },
        "zero_shot": base_metrics,
        "finetuned": ft_metrics,
        "delta": {
            "avg_r1": float(ft_metrics["avg_r1"] - base_metrics["avg_r1"]),
            "avg_r5": float(ft_metrics["avg_r5"] - base_metrics["avg_r5"]),
            "avg_r10": float(ft_metrics["avg_r10"] - base_metrics["avg_r10"]),
        },
    }

    out_json = args.out_dir / "fashiongen_base_vs_finetuned.json"
    out_txt = args.out_dir / "fashiongen_base_vs_finetuned.txt"
    out_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    out_txt.write_text(
        "\n".join(
            [
                "FashionGen base vs fine-tuned",
                f"zero-shot avg_r1={base_metrics['avg_r1']:.4f} avg_r5={base_metrics['avg_r5']:.4f} avg_r10={base_metrics['avg_r10']:.4f}",
                f"finetuned avg_r1={ft_metrics['avg_r1']:.4f} avg_r5={ft_metrics['avg_r5']:.4f} avg_r10={ft_metrics['avg_r10']:.4f}",
                f"delta avg_r1={comparison['delta']['avg_r1']:+.4f} avg_r5={comparison['delta']['avg_r5']:+.4f} avg_r10={comparison['delta']['avg_r10']:+.4f}",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
