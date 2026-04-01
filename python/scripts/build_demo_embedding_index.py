#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="ViT-B-16")
    parser.add_argument("--pretrained", default="laion2b_s34b_b88k")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(args: argparse.Namespace, device: torch.device):
    model, preprocess, _ = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:5]}")
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(args.model_name)
    return model, preprocess, tokenizer, missing


def resolve_image_path(images_root: Path, image_field: str) -> Path:
    return images_root / Path(image_field).name


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(args.catalog.read_text())
    items = payload["items"][: args.limit or None]

    device = pick_device()
    model, preprocess, tokenizer, missing = load_model(args, device)
    print(f"device={device} items={len(items)} missing_keys={len(missing)}")

    embedding_rows: list[np.ndarray] = []
    kept_items = []
    batch_tensors = []
    batch_items = []

    with torch.inference_mode():
        for item in items:
            image_path = resolve_image_path(args.images_root, item["image"])
            if not image_path.exists():
                continue
            image = Image.open(image_path).convert("RGB")
            batch_tensors.append(preprocess(image))
            batch_items.append(item)

            if len(batch_tensors) >= args.batch_size:
                batch = torch.stack(batch_tensors).to(device)
                feats = model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                embedding_rows.append(feats.detach().cpu().numpy().astype(np.float32))
                kept_items.extend(batch_items)
                batch_tensors = []
                batch_items = []

        if batch_tensors:
            batch = torch.stack(batch_tensors).to(device)
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embedding_rows.append(feats.detach().cpu().numpy().astype(np.float32))
            kept_items.extend(batch_items)

    embeddings = np.concatenate(embedding_rows, axis=0)
    np.save(args.output_dir / "image_embeddings.npy", embeddings)
    (args.output_dir / "catalog.json").write_text(json.dumps({"items": kept_items}))
    config = {
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "checkpoint": args.checkpoint.name,
        "count": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]),
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
