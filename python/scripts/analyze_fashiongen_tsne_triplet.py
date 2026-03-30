#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a three-way balanced FashionGen t-SNE comparison on a fixed sample manifest "
            "for zero-shot, retrieval-only, and joint checkpoints."
        )
    )
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-name", default="ViT-B-16")
    parser.add_argument("--pretrained", default="laion2b_s34b_b88k")
    parser.add_argument("--model-cache-dir", type=Path, required=True)
    parser.add_argument("--retrieval-checkpoint", type=Path, required=True)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    return x / denom


def knn_same_category_ratio(emb: np.ndarray, labels: np.ndarray, k: int) -> float:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(emb)
    indices = nn.kneighbors(return_distance=False)[:, 1:]
    return float(np.mean([(labels[row] == labels[nbrs]).mean() for row, nbrs in enumerate(indices)]))


def cluster_metrics(emb: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "knn_same_category_at5": knn_same_category_ratio(emb, labels, k=5),
        "knn_same_category_at10": knn_same_category_ratio(emb, labels, k=10),
        "silhouette": float(silhouette_score(emb, labels, metric="cosine")),
        "calinski_harabasz": float(calinski_harabasz_score(emb, labels)),
        "davies_bouldin": float(davies_bouldin_score(emb, labels)),
    }


def short_ref(path: Path, depth: int = 2) -> str:
    parts = path.parts[-depth:]
    return "/".join(parts) if parts else path.name


def load_manifest_samples(manifest_path: Path, shards_dir: Path) -> tuple[list[dict], list[str]]:
    manifest = json.loads(manifest_path.read_text())
    by_shard: dict[str, list[dict]] = {}
    for sample in manifest:
        by_shard.setdefault(sample["shard"], []).append(sample)

    loaded: list[dict] = []
    for shard_name, shard_samples in by_shard.items():
        shard_path = shards_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard: {shard_path}")
        wanted = {sample["member_stem"]: sample for sample in shard_samples}
        with tarfile.open(shard_path, "r") as tf:
            members = {member.name: member for member in tf.getmembers()}
            for stem, sample in wanted.items():
                jpg_name = f"{stem}.jpg"
                if jpg_name not in members:
                    raise FileNotFoundError(f"Missing image {jpg_name} in {shard_path}")
                fh = tf.extractfile(members[jpg_name])
                if fh is None:
                    raise RuntimeError(f"Failed to read {jpg_name} from {shard_path}")
                image = Image.open(io.BytesIO(fh.read())).convert("RGB")
                loaded.append(
                    {
                        "sample_id": sample["sample_id"],
                        "category": sample["category"],
                        "caption": sample["caption"],
                        "shard": sample["shard"],
                        "member_stem": sample["member_stem"],
                        "image": image,
                    }
                )

    loaded.sort(key=lambda x: x["sample_id"])
    category_order = []
    for sample in loaded:
        if sample["category"] not in category_order:
            category_order.append(sample["category"])
    return loaded, category_order


def load_model(model_name: str, pretrained: str, cache_dir: Path, checkpoint: Path | None, device: torch.device):
    clip_model, preprocess, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        cache_dir=str(cache_dir),
    )
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        clip_model.load_state_dict(state_dict)
    clip_model = clip_model.to(device).eval()
    return clip_model, preprocess


@torch.no_grad()
def encode_images(
    samples: list[dict],
    *,
    model_name: str,
    pretrained: str,
    cache_dir: Path,
    checkpoint: Path | None,
    batch_size: int,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = load_model(model_name, pretrained, cache_dir, checkpoint, device)
    tensors = [preprocess(sample["image"]) for sample in samples]
    batches = []
    for start in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[start : start + batch_size]).to(device, non_blocking=True)
        emb = model.encode_image(batch)
        batches.append(torch.nn.functional.normalize(emb, dim=-1).cpu().numpy().astype(np.float32))
    return np.concatenate(batches, axis=0)


def plot_tsne(ax, emb: np.ndarray, categories: np.ndarray, category_order: list[str], title: str) -> None:
    coords = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=35,
        random_state=42,
    ).fit_transform(emb)
    colors = plt.cm.tab10(np.linspace(0, 1, len(category_order)))
    for color, category in zip(colors, category_order):
        mask = categories == category
        ax.scatter(coords[mask, 0], coords[mask, 1], s=9, alpha=0.75, label=category, color=color)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    samples, category_order = load_manifest_samples(args.manifest, args.shards_dir)
    categories = np.array([sample["category"] for sample in samples])

    zero_shot = encode_images(
        samples,
        model_name=args.model_name,
        pretrained=args.pretrained,
        cache_dir=args.model_cache_dir,
        checkpoint=None,
        batch_size=args.batch_size,
    )
    retrieval = encode_images(
        samples,
        model_name=args.model_name,
        pretrained=args.pretrained,
        cache_dir=args.model_cache_dir,
        checkpoint=args.retrieval_checkpoint,
        batch_size=args.batch_size,
    )
    joint = encode_images(
        samples,
        model_name=args.model_name,
        pretrained=args.pretrained,
        cache_dir=args.model_cache_dir,
        checkpoint=args.joint_checkpoint,
        batch_size=args.batch_size,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    plot_tsne(axes[0], zero_shot, categories, category_order, "Zero-shot")
    plot_tsne(axes[1], retrieval, categories, category_order, "Retrieval-only")
    plot_tsne(axes[2], joint, categories, category_order, "Joint")
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.savefig(args.out_dir / "fashiongen_tsne_triplet.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "sample_count": int(len(samples)),
        "categories": category_order,
        "manifest": short_ref(args.manifest),
        "retrieval_checkpoint": short_ref(args.retrieval_checkpoint),
        "joint_checkpoint": short_ref(args.joint_checkpoint),
        "zero_shot": cluster_metrics(zero_shot, categories),
        "retrieval_only": cluster_metrics(retrieval, categories),
        "joint": cluster_metrics(joint, categories),
    }
    (args.out_dir / "fashiongen_tsne_triplet_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
