#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import open_clip
import torch
import torch.nn.functional as F
import webdataset as wds
from PIL import Image
from torch import nn
from torch.cuda.amp import GradScaler, autocast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CLIP with alignment loss + vision-only classification loss."
    )
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--shards-glob", type=str, default="clip_dataset_train_aug.*.tar")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--valid-shards-dir", type=Path, default=None)
    parser.add_argument("--valid-shards-glob", type=str, default="clip_dataset_valid.*.tar")
    parser.add_argument("--valid-max-shards", type=int, default=None)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Run validation every N steps. 0 disables periodic eval.",
    )
    parser.add_argument("--eval-max-batches", type=int, default=20)
    parser.add_argument(
        "--two-stage-eval",
        choices=["off", "soft", "hard"],
        default="off",
        help="Optional two-stage retrieval reranking during validation only.",
    )
    parser.add_argument(
        "--two-stage-topk-categories",
        type=int,
        default=3,
        help="Top-k predicted categories used for two-stage gating.",
    )
    parser.add_argument(
        "--two-stage-alpha",
        type=float,
        default=0.15,
        help="Soft two-stage reranking strength.",
    )
    parser.add_argument(
        "--two-stage-text-temp",
        type=float,
        default=20.0,
        help="Temperature for text-side category probabilities from category prompts.",
    )

    parser.add_argument("--model-name", type=str, default="ViT-B-16")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b88k")
    parser.add_argument("--model-cache-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, default=None)

    parser.add_argument("--image-key", type=str, default="jpg")
    parser.add_argument("--caption-keys", nargs="+", default=["txt0.txt", "txt1.txt", "txt2.txt"])
    parser.add_argument("--meta-key", type=str, default="meta.json")
    parser.add_argument("--valid-image-key", type=str, default="jpg")
    parser.add_argument("--valid-caption-keys", nargs="+", default=["txt", "txt0.txt"])
    parser.add_argument("--valid-meta-key", type=str, default="meta.json")
    parser.add_argument("--caption-mode", choices=["primary", "random"], default="random")

    parser.add_argument("--loss-type", choices=["contrastive", "siglip"], default="contrastive")
    parser.add_argument("--align-weight", type=float, default=1.0)
    parser.add_argument("--cls-weight", type=float, default=0.2)
    parser.add_argument("--cls-weight-schedule", choices=["constant", "warmup"], default="constant")
    parser.add_argument("--cls-weight-warmup-steps", type=int, default=1000)

    parser.add_argument("--cls-head-type", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--cls-hidden-dim", type=int, default=1024)
    parser.add_argument("--cls-dropout", type=float, default=0.1)
    parser.add_argument("--cls-feature", choices=["img_norm", "img_raw"], default="img_norm")
    parser.add_argument("--cls-grad", choices=["shared", "detached"], default="shared")
    parser.add_argument("--cls-loss-type", choices=["ce", "ce_ls", "focal"], default="ce")
    parser.add_argument("--cls-label-smoothing", type=float, default=0.05)
    parser.add_argument("--cls-focal-gamma", type=float, default=2.0)
    parser.add_argument("--cls-class-weighting", choices=["none", "inv_freq", "sqrt_inv_freq"], default="none")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-checkpoint", action="store_true")
    return parser.parse_args()


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("joint_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"joint_train_{ts}.log"

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.info("Writing logs to %s", log_file)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_shards(shards_dir: Path, shards_glob: str, max_shards: int | None) -> List[Path]:
    shards = sorted(shards_dir.glob(shards_glob))
    if max_shards is not None:
        shards = shards[:max_shards]
    if not shards:
        raise FileNotFoundError(f"No shards found in {shards_dir} matching '{shards_glob}'")
    return shards


def decode_text(raw: bytes | str | None) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="ignore")
    return str(raw)


def decode_meta(raw: bytes | str | dict | None) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    try:
        return json.loads(raw)
    except Exception:
        return {}


def build_category_vocab(shards: Sequence[Path], logger: logging.Logger) -> Dict[str, int]:
    categories: set[str] = set()
    for shard in shards:
        with tarfile.open(shard, "r") as tf:
            for member in tf.getmembers():
                if not member.name.endswith(".meta.json"):
                    continue
                f = tf.extractfile(member)
                if f is None:
                    continue
                meta = decode_meta(f.read())
                cat = str(meta.get("category", "UNKNOWN")).strip() or "UNKNOWN"
                categories.add(cat)
    sorted_cats = sorted(categories)
    vocab = {cat: idx for idx, cat in enumerate(sorted_cats)}
    logger.info("Category vocab size: %d", len(vocab))
    logger.info("First categories: %s", sorted_cats[:10])
    return vocab


def build_class_weights(
    *,
    shards: Sequence[Path],
    category_to_id: Dict[str, int],
    mode: str,
    logger: logging.Logger,
    device: torch.device,
) -> torch.Tensor | None:
    if mode == "none":
        return None

    counts = torch.ones(len(category_to_id), dtype=torch.float64)
    for shard in shards:
        with tarfile.open(shard, "r") as tf:
            for member in tf.getmembers():
                if not member.name.endswith(".meta.json"):
                    continue
                f = tf.extractfile(member)
                if f is None:
                    continue
                meta = decode_meta(f.read())
                category = str(meta.get("category", "UNKNOWN")).strip() or "UNKNOWN"
                idx = category_to_id.get(category)
                if idx is not None:
                    counts[idx] += 1.0

    inv = 1.0 / counts
    if mode == "sqrt_inv_freq":
        inv = torch.sqrt(inv)

    weights = (inv / inv.mean()).to(torch.float32)
    logger.info(
        "Class weighting enabled (%s). weight range=[%.4f, %.4f]",
        mode,
        float(weights.min().item()),
        float(weights.max().item()),
    )
    return weights.to(device)


def create_loader(
    shards: Sequence[Path],
    image_key: str,
    caption_keys: Sequence[str],
    meta_key: str,
    preprocess,
    category_to_id: Dict[str, int],
    caption_mode: str,
    batch_size: int,
    num_workers: int,
    shardshuffle: int | bool,
):
    shard_pattern = "::".join(str(p) for p in shards)

    def process_sample(sample: dict):
        image = sample.get(image_key)
        if image is None or not isinstance(image, Image.Image):
            return None

        captions = []
        for key in caption_keys:
            txt = decode_text(sample.get(key)).strip()
            if txt:
                captions.append(txt)
        if not captions:
            captions = [""]

        if caption_mode == "primary":
            text = captions[0]
        else:
            text = random.choice(captions)

        meta = decode_meta(sample.get(meta_key))
        if "category" not in meta:
            cat_id = -1
        else:
            category = str(meta.get("category", "UNKNOWN")).strip() or "UNKNOWN"
            cat_id = category_to_id.get(category, -1)

        return preprocess(image), text, cat_id

    dataset = (
        wds.WebDataset(shard_pattern, shardshuffle=shardshuffle, handler=wds.handlers.warn_and_continue)
        .decode(wds.autodecode.imagehandler("pil"), handler=wds.autodecode.basichandlers)
        .map(process_sample)
        .select(lambda x: x is not None)
    )

    return wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
    )


class JointCLIPModel(nn.Module):
    def __init__(
        self,
        clip_model: nn.Module,
        emb_dim: int,
        num_classes: int,
        cls_head_type: str,
        cls_hidden_dim: int,
        cls_dropout: float,
        cls_feature: str,
        cls_grad: str,
    ):
        super().__init__()
        self.clip_model = clip_model
        self.cls_feature = cls_feature
        self.cls_grad = cls_grad

        if cls_head_type == "linear":
            self.cls_head = nn.Sequential(
                nn.LayerNorm(emb_dim),
                nn.Linear(emb_dim, num_classes),
            )
        elif cls_head_type == "mlp":
            self.cls_head = nn.Sequential(
                nn.LayerNorm(emb_dim),
                nn.Linear(emb_dim, cls_hidden_dim),
                nn.GELU(),
                nn.Dropout(cls_dropout),
                nn.Linear(cls_hidden_dim, num_classes),
            )
        else:
            raise ValueError(f"Unsupported cls_head_type: {cls_head_type}")

    def forward(self, image_batch: torch.Tensor, text_tokens: torch.Tensor):
        img_emb = self.clip_model.encode_image(image_batch)
        txt_emb = self.clip_model.encode_text(text_tokens)

        cls_input = F.normalize(img_emb, dim=-1) if self.cls_feature == "img_norm" else img_emb
        if self.cls_grad == "detached":
            cls_input = cls_input.detach()

        cls_logits = self.cls_head(cls_input)
        return img_emb, txt_emb, cls_logits


def compute_alignment_loss(
    loss_type: str,
    img_emb: torch.Tensor,
    txt_emb: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor | None,
) -> torch.Tensor:
    img = F.normalize(img_emb, dim=-1)
    txt = F.normalize(txt_emb, dim=-1)
    logits = (img @ txt.T) * logit_scale.exp()

    if loss_type == "contrastive":
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

    if loss_type == "siglip":
        if logit_bias is None:
            raise ValueError("siglip loss requires logit_bias")
        logits = logits + logit_bias
        targets = torch.eye(logits.size(0), device=logits.device)
        return F.binary_cross_entropy_with_logits(logits, targets)

    raise ValueError(f"Unsupported loss_type: {loss_type}")


def compute_cls_loss(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    cls_loss_type: str,
    cls_label_smoothing: float,
    cls_focal_gamma: float,
    class_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, int]:
    valid_mask = targets >= 0
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        return logits.new_zeros(()), 0

    valid_logits = logits[valid_mask]
    valid_targets = targets[valid_mask]

    if cls_loss_type == "ce":
        return F.cross_entropy(valid_logits, valid_targets, weight=class_weights), valid_count

    if cls_loss_type == "ce_ls":
        return (
            F.cross_entropy(
                valid_logits,
                valid_targets,
                weight=class_weights,
                label_smoothing=cls_label_smoothing,
            ),
            valid_count,
        )

    if cls_loss_type == "focal":
        ce = F.cross_entropy(valid_logits, valid_targets, weight=class_weights, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** cls_focal_gamma) * ce
        return focal.mean(), valid_count

    raise ValueError(f"Unsupported cls_loss_type: {cls_loss_type}")


def compute_cls_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float | None, int]:
    valid_mask = targets >= 0
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        return None, 0
    pred = torch.argmax(logits[valid_mask], dim=-1)
    acc = float((pred == targets[valid_mask]).float().mean().item())
    return acc, valid_count


def recall_at_k(sim: torch.Tensor, k: int) -> tuple[float, float]:
    n = sim.shape[0]
    if n <= 0:
        return 0.0, 0.0
    k = min(k, n)
    t2i_topk = torch.argsort(sim, dim=0)[-k:].T
    i2t_topk = torch.argsort(sim, dim=1)[:, -k:]
    idx = torch.arange(n)
    t2i = (t2i_topk == idx[:, None]).any(dim=1).float().mean().item()
    i2t = (i2t_topk == idx[:, None]).any(dim=1).float().mean().item()
    return t2i, i2t


def aggregate_retrieval_metrics(sim: torch.Tensor) -> dict:
    r1_t2i, r1_i2t = recall_at_k(sim, 1)
    r5_t2i, r5_i2t = recall_at_k(sim, 5)
    r10_t2i, r10_i2t = recall_at_k(sim, 10)
    return {
        "recall@1_t2i": float(r1_t2i),
        "recall@1_i2t": float(r1_i2t),
        "recall@5_t2i": float(r5_t2i),
        "recall@5_i2t": float(r5_i2t),
        "recall@10_t2i": float(r10_t2i),
        "recall@10_i2t": float(r10_i2t),
        "avg_r1": float((r1_t2i + r1_i2t) / 2.0),
        "avg_r5": float((r5_t2i + r5_i2t) / 2.0),
        "avg_r10": float((r10_t2i + r10_i2t) / 2.0),
    }


@torch.no_grad()
def build_category_text_prototypes(
    *,
    clip_model: nn.Module,
    tokenizer,
    category_to_id: Dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    ordered = sorted(category_to_id.items(), key=lambda kv: kv[1])
    prompts = [f"a product photo of {name.lower().replace('_', ' ')}" for name, _ in ordered]
    tokens = tokenizer(prompts).to(device)
    txt = clip_model.encode_text(tokens)
    return F.normalize(txt, dim=-1)


def build_topk_overlap_mask(
    *,
    img_probs: torch.Tensor,
    txt_probs: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    n, num_classes = img_probs.shape
    k = min(max(1, topk), num_classes)
    img_topk = torch.topk(img_probs, k=k, dim=1).indices
    txt_topk = torch.topk(txt_probs, k=k, dim=1).indices

    img_bool = torch.zeros((n, num_classes), dtype=torch.bool, device=img_probs.device)
    txt_bool = torch.zeros((n, num_classes), dtype=torch.bool, device=txt_probs.device)
    img_bool.scatter_(1, img_topk, True)
    txt_bool.scatter_(1, txt_topk, True)

    overlap = torch.logical_and(img_bool[:, None, :], txt_bool[None, :, :]).any(dim=-1)
    # Guard against degenerate all-false rows/cols to avoid invalid ranking.
    no_row = ~overlap.any(dim=1)
    no_col = ~overlap.any(dim=0)
    if no_row.any():
        overlap[no_row, :] = True
    if no_col.any():
        overlap[:, no_col] = True
    return overlap


def two_stage_similarity(
    *,
    base_sim: torch.Tensor,
    cls_logits_img: torch.Tensor,
    txt_emb: torch.Tensor,
    category_text_prototypes: torch.Tensor,
    mode: str,
    topk_categories: int,
    alpha: float,
    text_temp: float,
) -> tuple[torch.Tensor, dict]:
    if mode == "off":
        return base_sim, {"mode": "off"}

    img_probs = torch.softmax(cls_logits_img, dim=-1)
    txt_cat_logits = (txt_emb @ category_text_prototypes.T) * text_temp
    txt_probs = torch.softmax(txt_cat_logits, dim=-1)

    if mode == "soft":
        affinity = img_probs @ txt_probs.T
        reranked = base_sim + alpha * torch.log(torch.clamp(affinity, min=1e-8))
        diag = {
            "mode": "soft",
            "alpha": float(alpha),
            "topk_categories": int(min(max(1, topk_categories), img_probs.shape[1])),
            "text_temp": float(text_temp),
            "mean_affinity": float(affinity.mean().item()),
        }
        return reranked, diag

    if mode == "hard":
        overlap = build_topk_overlap_mask(
            img_probs=img_probs,
            txt_probs=txt_probs,
            topk=topk_categories,
        )
        reranked = base_sim.masked_fill(~overlap, -1e4)
        diag = {
            "mode": "hard",
            "alpha": float(alpha),
            "topk_categories": int(min(max(1, topk_categories), img_probs.shape[1])),
            "text_temp": float(text_temp),
            "allowed_fraction": float(overlap.float().mean().item()),
        }
        return reranked, diag

    raise ValueError(f"Unsupported two-stage mode: {mode}")


def cls_weight_multiplier(step: int, schedule: str, warmup_steps: int) -> float:
    if schedule == "constant":
        return 1.0
    if schedule == "warmup":
        if warmup_steps <= 0:
            return 1.0
        return float(min(1.0, step / warmup_steps))
    raise ValueError(f"Unsupported cls weight schedule: {schedule}")


@torch.no_grad()
def run_validation(
    *,
    model: JointCLIPModel,
    tokenizer,
    loader,
    device: torch.device,
    loss_type: str,
    logit_bias: torch.Tensor | None,
    max_batches: int,
    cls_loss_type: str,
    cls_label_smoothing: float,
    cls_focal_gamma: float,
    class_weights: torch.Tensor | None,
    two_stage_mode: str,
    two_stage_topk_categories: int,
    two_stage_alpha: float,
    two_stage_text_temp: float,
    category_text_prototypes: torch.Tensor | None,
) -> dict:
    model.eval()
    img_embeds: list[torch.Tensor] = []
    txt_embeds: list[torch.Tensor] = []
    align_losses: list[float] = []
    cls_losses: list[float] = []
    cls_logits_list: list[torch.Tensor] = []
    cls_correct = 0.0
    cls_total = 0
    seen = 0

    for batch_idx, (images, texts, cls_targets) in enumerate(loader):
        if batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        cls_targets = cls_targets.to(device, non_blocking=True)
        text_tokens = tokenizer(list(texts)).to(device)

        img, txt, cls_logits = model(images, text_tokens)
        align_loss = compute_alignment_loss(
            loss_type,
            img,
            txt,
            model.clip_model.logit_scale,
            logit_bias,
        )
        cls_loss, cls_valid = compute_cls_loss(
            logits=cls_logits,
            targets=cls_targets,
            cls_loss_type=cls_loss_type,
            cls_label_smoothing=cls_label_smoothing,
            cls_focal_gamma=cls_focal_gamma,
            class_weights=class_weights,
        )

        align_losses.append(float(align_loss.item()))
        if cls_valid > 0:
            cls_losses.append(float(cls_loss.item()))

        valid_mask = cls_targets >= 0
        if valid_mask.any():
            pred = torch.argmax(cls_logits[valid_mask], dim=-1)
            cls_correct += float((pred == cls_targets[valid_mask]).float().sum().item())
            cls_total += int(valid_mask.sum().item())

        img_embeds.append(F.normalize(img, dim=-1).cpu())
        txt_embeds.append(F.normalize(txt, dim=-1).cpu())
        cls_logits_list.append(cls_logits.cpu())
        seen += images.size(0)

    if not img_embeds:
        model.train()
        return {
            "num_samples": 0,
            "align_loss": None,
            "cls_loss": None,
            "cls_acc": None,
            "cls_valid_count": 0,
            "recall@1_t2i": None,
            "recall@1_i2t": None,
            "recall@5_t2i": None,
            "recall@5_i2t": None,
            "recall@10_t2i": None,
            "recall@10_i2t": None,
            "avg_r1": None,
            "avg_r5": None,
            "avg_r10": None,
        }

    img_mat = torch.cat(img_embeds, dim=0)
    txt_mat = torch.cat(txt_embeds, dim=0)
    cls_logits_mat = torch.cat(cls_logits_list, dim=0)
    sim = img_mat @ txt_mat.T

    record = {
        "num_samples": seen,
        "align_loss": float(sum(align_losses) / max(len(align_losses), 1)),
        "cls_loss": float(sum(cls_losses) / max(len(cls_losses), 1)) if cls_losses else None,
        "cls_acc": float(cls_correct / cls_total) if cls_total > 0 else None,
        "cls_valid_count": int(cls_total),
    }
    base_metrics = aggregate_retrieval_metrics(sim)
    record.update(base_metrics)
    record["two_stage_mode"] = two_stage_mode
    record["base_avg_r1"] = base_metrics["avg_r1"]
    record["base_avg_r5"] = base_metrics["avg_r5"]
    record["base_avg_r10"] = base_metrics["avg_r10"]

    if two_stage_mode != "off":
        if category_text_prototypes is None:
            raise RuntimeError("two-stage eval enabled but category_text_prototypes is missing")
        ts_sim, ts_diag = two_stage_similarity(
            base_sim=sim,
            cls_logits_img=cls_logits_mat,
            txt_emb=txt_mat,
            category_text_prototypes=category_text_prototypes,
            mode=two_stage_mode,
            topk_categories=two_stage_topk_categories,
            alpha=two_stage_alpha,
            text_temp=two_stage_text_temp,
        )
        ts_metrics = aggregate_retrieval_metrics(ts_sim)
        record.update({f"two_stage_{k}": v for k, v in ts_metrics.items()})
        record["two_stage_score"] = (
            0.30 * float(ts_metrics["avg_r1"])
            + 0.40 * float(ts_metrics["avg_r5"])
            + 0.30 * float(ts_metrics["avg_r10"])
        )
        record["two_stage_delta_avg_r1"] = float(ts_metrics["avg_r1"] - base_metrics["avg_r1"])
        record["two_stage_delta_avg_r5"] = float(ts_metrics["avg_r5"] - base_metrics["avg_r5"])
        record["two_stage_delta_avg_r10"] = float(ts_metrics["avg_r10"] - base_metrics["avg_r10"])
        record["two_stage_delta_score"] = float(
            record["two_stage_score"]
            - (0.30 * base_metrics["avg_r1"] + 0.40 * base_metrics["avg_r5"] + 0.30 * base_metrics["avg_r10"])
        )
        for k, v in ts_diag.items():
            record[f"two_stage_{k}"] = v

    model.train()
    return record


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_dir)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    set_seed(args.seed)

    if args.model_name != "ViT-B-16" or args.pretrained != "laion2b_s34b_b88k":
        logger.warning(
            "This experiment suite is tuned for ViT-B-16/laion2b_s34b_b88k; got model=%s pretrained=%s",
            args.model_name,
            args.pretrained,
        )

    shards = resolve_shards(args.shards_dir, args.shards_glob, args.max_shards)
    logger.info("Using %d shard(s). First shard: %s", len(shards), shards[0])

    category_to_id = build_category_vocab(shards, logger)
    if "UNKNOWN" not in category_to_id:
        category_to_id["UNKNOWN"] = len(category_to_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    clip_model, preprocess, _ = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained,
        cache_dir=str(args.model_cache_dir),
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)

    if args.init_checkpoint is not None:
        ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        clip_model.load_state_dict(state_dict)
        logger.info("Loaded init checkpoint: %s", args.init_checkpoint)

    emb_dim = int(getattr(clip_model.text_projection, "shape", [0, 0])[1])
    if emb_dim <= 0:
        raise RuntimeError("Could not infer embedding dimension from CLIP model")

    model = JointCLIPModel(
        clip_model=clip_model,
        emb_dim=emb_dim,
        num_classes=len(category_to_id),
        cls_head_type=args.cls_head_type,
        cls_hidden_dim=args.cls_hidden_dim,
        cls_dropout=args.cls_dropout,
        cls_feature=args.cls_feature,
        cls_grad=args.cls_grad,
    ).to(device)

    logit_bias = None
    if args.loss_type == "siglip":
        logit_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))

    class_weights = build_class_weights(
        shards=shards,
        category_to_id=category_to_id,
        mode=args.cls_class_weighting,
        logger=logger,
        device=device,
    )

    trainable_params = list(model.parameters())
    if logit_bias is not None:
        trainable_params.append(logit_bias)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    loader = create_loader(
        shards=shards,
        image_key=args.image_key,
        caption_keys=args.caption_keys,
        meta_key=args.meta_key,
        preprocess=preprocess,
        category_to_id=category_to_id,
        caption_mode=args.caption_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shardshuffle=100,
    )

    valid_loader = None
    category_text_prototypes = None
    if args.eval_every > 0:
        if args.valid_shards_dir is None:
            logger.warning("eval-every > 0 but --valid-shards-dir is not set; validation disabled.")
        else:
            valid_shards = resolve_shards(
                args.valid_shards_dir, args.valid_shards_glob, args.valid_max_shards
            )
            logger.info(
                "Using %d valid shard(s). First valid shard: %s",
                len(valid_shards),
                valid_shards[0],
            )
            valid_loader = create_loader(
                shards=valid_shards,
                image_key=args.valid_image_key,
                caption_keys=args.valid_caption_keys,
                meta_key=args.valid_meta_key,
                preprocess=preprocess,
                category_to_id=category_to_id,
                caption_mode="primary",
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shardshuffle=False,
            )
            if args.two_stage_eval != "off":
                with torch.no_grad():
                    category_text_prototypes = build_category_text_prototypes(
                        clip_model=model.clip_model,
                        tokenizer=tokenizer,
                        category_to_id=category_to_id,
                        device=device,
                    ).cpu()
                logger.info(
                    "Two-stage eval enabled (mode=%s, topk=%d, alpha=%.3f, text_temp=%.2f). "
                    "Gating uses predicted categories only.",
                    args.two_stage_eval,
                    args.two_stage_topk_categories,
                    args.two_stage_alpha,
                    args.two_stage_text_temp,
                )

    metrics_path = args.log_dir / f"metrics_{run_ts}.jsonl"
    eval_path = args.log_dir / f"eval_{run_ts}.jsonl"
    step = 0
    model.train()

    logger.info(
        (
            "Training start | loss=%s align_w=%.3f cls_w=%.3f schedule=%s "
            "batch=%d max_steps=%d cls_head=%s cls_feature=%s cls_grad=%s cls_loss=%s"
        ),
        args.loss_type,
        args.align_weight,
        args.cls_weight,
        args.cls_weight_schedule,
        args.batch_size,
        args.max_steps,
        args.cls_head_type,
        args.cls_feature,
        args.cls_grad,
        args.cls_loss_type,
    )

    for epoch in range(args.epochs):
        logger.info("Epoch %d/%d", epoch + 1, args.epochs)
        for images, texts, cls_targets in loader:
            if step >= args.max_steps:
                break
            step += 1

            images = images.to(device, non_blocking=True)
            cls_targets = cls_targets.to(device, non_blocking=True)
            text_tokens = tokenizer(list(texts)).to(device)

            cls_w_eff = args.cls_weight * cls_weight_multiplier(
                step=step,
                schedule=args.cls_weight_schedule,
                warmup_steps=args.cls_weight_warmup_steps,
            )

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                img_emb, txt_emb, cls_logits = model(images, text_tokens)

                align_loss = compute_alignment_loss(
                    args.loss_type,
                    img_emb,
                    txt_emb,
                    model.clip_model.logit_scale,
                    logit_bias,
                )
                cls_loss, cls_valid = compute_cls_loss(
                    logits=cls_logits,
                    targets=cls_targets,
                    cls_loss_type=args.cls_loss_type,
                    cls_label_smoothing=args.cls_label_smoothing,
                    cls_focal_gamma=args.cls_focal_gamma,
                    class_weights=class_weights,
                )
                total_loss = args.align_weight * align_loss + cls_w_eff * cls_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                cls_acc, cls_valid_count = compute_cls_accuracy(cls_logits, cls_targets)

            record = {
                "step": step,
                "epoch": epoch + 1,
                "total_loss": float(total_loss.item()),
                "align_loss": float(align_loss.item()),
                "cls_loss": float(cls_loss.item()) if cls_valid > 0 else None,
                "cls_acc": cls_acc,
                "cls_valid_count": cls_valid_count,
                "cls_weight_effective": float(cls_w_eff),
                "logit_scale": float(model.clip_model.logit_scale.exp().item()),
                "loss_type": args.loss_type,
                "cls_head_type": args.cls_head_type,
                "cls_feature": args.cls_feature,
                "cls_grad": args.cls_grad,
                "cls_loss_type": args.cls_loss_type,
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            if step % args.log_every == 0:
                logger.info(
                    "step=%d total=%.4f align=%.4f cls=%s acc=%s cls_w=%.4f",
                    step,
                    record["total_loss"],
                    record["align_loss"],
                    f"{record['cls_loss']:.4f}" if record["cls_loss"] is not None else "n/a",
                    f"{record['cls_acc']:.3f}" if record["cls_acc"] is not None else "n/a",
                    record["cls_weight_effective"],
                )

            if valid_loader is not None and args.eval_every > 0 and step % args.eval_every == 0:
                eval_record = run_validation(
                    model=model,
                    tokenizer=tokenizer,
                    loader=valid_loader,
                    device=device,
                    loss_type=args.loss_type,
                    logit_bias=logit_bias,
                    max_batches=args.eval_max_batches,
                    cls_loss_type=args.cls_loss_type,
                    cls_label_smoothing=args.cls_label_smoothing,
                    cls_focal_gamma=args.cls_focal_gamma,
                    class_weights=class_weights,
                    two_stage_mode=args.two_stage_eval,
                    two_stage_topk_categories=args.two_stage_topk_categories,
                    two_stage_alpha=args.two_stage_alpha,
                    two_stage_text_temp=args.two_stage_text_temp,
                    category_text_prototypes=category_text_prototypes,
                )
                eval_record["step"] = step
                eval_record["epoch"] = epoch + 1
                eval_record["cls_weight_effective"] = float(cls_w_eff)
                eval_record["score"] = (
                    0.30 * float(eval_record["avg_r1"])
                    + 0.40 * float(eval_record["avg_r5"])
                    + 0.30 * float(eval_record["avg_r10"])
                )
                with eval_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(eval_record) + "\n")
                logger.info(
                    "eval@step=%d n=%s avg_r1=%.4f avg_r5=%.4f avg_r10=%.4f score=%.4f cls_acc=%s",
                    step,
                    eval_record.get("num_samples"),
                    eval_record.get("avg_r1", 0.0),
                    eval_record.get("avg_r5", 0.0),
                    eval_record.get("avg_r10", 0.0),
                    eval_record.get("score", 0.0),
                    eval_record.get("cls_acc"),
                )
                if args.two_stage_eval != "off":
                    logger.info(
                        (
                            "two-stage@step=%d mode=%s avg_r1=%.4f avg_r5=%.4f "
                            "avg_r10=%.4f score=%.4f delta_score=%+.4f"
                        ),
                        step,
                        eval_record.get("two_stage_mode"),
                        eval_record.get("two_stage_avg_r1", 0.0),
                        eval_record.get("two_stage_avg_r5", 0.0),
                        eval_record.get("two_stage_avg_r10", 0.0),
                        eval_record.get("two_stage_score", 0.0),
                        eval_record.get("two_stage_delta_score", 0.0),
                    )

        if step >= args.max_steps:
            break

    logger.info("Training done at step %d", step)
    logger.info("Metrics written to %s", metrics_path)
    if valid_loader is not None and args.eval_every > 0:
        logger.info("Validation metrics written to %s", eval_path)

    if args.save_checkpoint:
        ckpt_path = args.log_dir / f"joint_clip_step_{step}.pt"
        payload = {
            "model_state_dict": model.clip_model.state_dict(),
            "cls_head_state_dict": model.cls_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "category_to_id": category_to_id,
            "args": vars(args),
        }
        if logit_bias is not None:
            payload["logit_bias"] = logit_bias.detach().cpu()
        torch.save(payload, ckpt_path)
        logger.info("Saved checkpoint to %s", ckpt_path)


if __name__ == "__main__":
    main()
