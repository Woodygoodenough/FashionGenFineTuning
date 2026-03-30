#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import open_clip
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import umap

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data_aug"
CHECKPOINT_PATH = ROOT / "checkpoints" / "ck_ViT-B-16.pt"
OUT_DIR = ROOT / "artifacts" / "demo" / "data"
THUMB_DIR = ROOT / "artifacts" / "demo" / "static" / "thumbnails"

SPACE_KEYS = ("image", "text", "joint")
MODEL_KEYS = ("base", "finetuned")

CATEGORY_RULES = {
    "dress": ["dress", "gown"],
    "tops": ["shirt", "top", "blouse", "sweater", "hoodie", "tee", "t-shirt"],
    "bottoms": ["jean", "pants", "trouser", "shorts", "skirt"],
    "outerwear": ["jacket", "coat", "parka", "blazer", "vest"],
    "footwear": ["sneaker", "shoe", "boot", "loafer", "sandal", "heel"],
    "bags": ["bag", "backpack", "tote", "satchel", "pouch", "wallet"],
    "accessories": ["hat", "cap", "belt", "scarf", "glove", "watch", "bracelet", "ring"],
}


def normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    return x / denom


def infer_category(caption: str) -> str:
    text = caption.lower()
    for category, needles in CATEGORY_RULES.items():
        if any(n in text for n in needles):
            return category
    return "other"


def chunked(seq: List, size: int):
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]


def load_samples(data_dir: Path, limit: int) -> List[Dict]:
    samples: List[Dict] = []
    shards = sorted(data_dir.glob("clip_dataset_train_aug.*.tar"))
    if not shards:
        shards = sorted(data_dir.glob("clip_dataset_train.*.tar"))

    for shard in shards:
        if len(samples) >= limit:
            break

        with tarfile.open(shard, "r") as tf:
            names = tf.getnames()
            members = set(names)
            jpg_names = [n for n in names if n.endswith(".jpg")]
            for jpg_name in jpg_names:
                if len(samples) >= limit:
                    break
                txt_name = jpg_name[:-4] + ".txt"
                txt0_name = jpg_name[:-4] + ".txt0.txt"
                meta_name = jpg_name[:-4] + ".meta.json"

                if txt0_name in members:
                    main_txt_name = txt0_name
                elif txt_name in members:
                    main_txt_name = txt_name
                else:
                    continue

                jpg_file = tf.extractfile(jpg_name)
                txt_file = tf.extractfile(main_txt_name)
                if jpg_file is None or txt_file is None:
                    continue

                image_bytes = jpg_file.read()
                caption = txt_file.read().decode("utf-8", errors="ignore").strip()
                meta = {}
                if meta_name in members:
                    meta_file = tf.extractfile(meta_name)
                    if meta_file is not None:
                        try:
                            meta = json.loads(meta_file.read())
                        except Exception:
                            meta = {}
                try:
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                except Exception:
                    continue

                sample_id = f"{shard.stem}_{Path(jpg_name).stem}"
                samples.append(
                    {
                        "id": sample_id,
                        "image": image,
                        "caption": caption,
                        "category": meta.get("category"),
                        "product_id": meta.get("product_id"),
                        "image_name": meta.get("image_name"),
                    }
                )

    return samples


def build_embeddings(
    samples: List[Dict],
    checkpoint_path: Path,
    batch_size: int,
    model_name: str,
    pretrained: str,
) -> Dict[str, np.ndarray]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model, preprocess, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    base_model = base_model.to(device).eval()

    ft_model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    ft_model.load_state_dict(state_dict)
    ft_model = ft_model.to(device).eval()

    tokenizer = open_clip.get_tokenizer(model_name)

    img_tensors = [preprocess(s["image"]) for s in samples]
    texts = [s["caption"] for s in samples]

    base_img, base_txt, ft_img, ft_txt = [], [], [], []

    with torch.no_grad():
        for img_batch in chunked(img_tensors, batch_size):
            batch = torch.stack(img_batch).to(device)
            base_img.append(base_model.encode_image(batch).cpu().numpy())
            ft_img.append(ft_model.encode_image(batch).cpu().numpy())

        for text_batch in chunked(texts, batch_size):
            tokens = tokenizer(text_batch).to(device)
            base_txt.append(base_model.encode_text(tokens).cpu().numpy())
            ft_txt.append(ft_model.encode_text(tokens).cpu().numpy())

    base_image = normalize_rows(np.concatenate(base_img, axis=0).astype(np.float32))
    base_text = normalize_rows(np.concatenate(base_txt, axis=0).astype(np.float32))
    ft_image = normalize_rows(np.concatenate(ft_img, axis=0).astype(np.float32))
    ft_text = normalize_rows(np.concatenate(ft_txt, axis=0).astype(np.float32))

    base_joint = normalize_rows(((base_image + base_text) / 2.0).astype(np.float32))
    ft_joint = normalize_rows(((ft_image + ft_text) / 2.0).astype(np.float32))

    return {
        "base_image": base_image,
        "base_text": base_text,
        "base_joint": base_joint,
        "finetuned_image": ft_image,
        "finetuned_text": ft_text,
        "finetuned_joint": ft_joint,
    }


def compute_umap_and_clusters(
    embeds: Dict[str, np.ndarray],
    captions: List[str],
    categories: List[str],
    n_clusters: int,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, List[Dict]]]:
    projections: Dict[str, np.ndarray] = {}
    labels: Dict[str, np.ndarray] = {}
    cluster_summaries: Dict[str, List[Dict]] = {}

    vectorizer = CountVectorizer(stop_words="english", max_features=1000)
    x_words = vectorizer.fit_transform(captions)
    vocab = np.array(vectorizer.get_feature_names_out())

    for model in MODEL_KEYS:
        for space in SPACE_KEYS:
            key = f"{model}_{space}"
            arr = embeds[key]

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(30, max(5, arr.shape[0] // 50)),
                min_dist=0.12,
                metric="cosine",
                random_state=42,
            )
            projections[key] = reducer.fit_transform(arr).astype(np.float32)

            km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            cluster_ids = km.fit_predict(arr)
            labels[key] = cluster_ids.astype(np.int16)

            summaries = []
            for cluster_id in range(n_clusters):
                idx = np.where(cluster_ids == cluster_id)[0]
                if idx.size == 0:
                    summaries.append(
                        {
                            "cluster_id": int(cluster_id),
                            "size": 0,
                            "purity": 0.0,
                            "dominant_category": "n/a",
                            "dominant_words": [],
                            "representative_indices": [],
                        }
                    )
                    continue

                cat_counter = Counter(categories[i] for i in idx)
                dom_cat, dom_count = cat_counter.most_common(1)[0]
                purity = float(dom_count / idx.size)

                word_scores = np.asarray(x_words[idx].sum(axis=0)).ravel()
                top_words = vocab[np.argsort(word_scores)[-6:]][::-1].tolist()

                centroid = km.cluster_centers_[cluster_id]
                dists = np.linalg.norm(arr[idx] - centroid, axis=1)
                rep = idx[np.argsort(dists)[:3]].tolist()

                summaries.append(
                    {
                        "cluster_id": int(cluster_id),
                        "size": int(idx.size),
                        "purity": purity,
                        "dominant_category": dom_cat,
                        "dominant_words": top_words,
                        "representative_indices": rep,
                    }
                )

            cluster_summaries[key] = summaries

    return projections, labels, cluster_summaries


def compute_error_bundle(
    items: List[Dict], embeds: Dict[str, np.ndarray], labels: Dict[str, np.ndarray], top_n: int
) -> Dict:
    img = embeds["finetuned_image"]
    txt = embeds["finetuned_text"]
    sim = img @ txt.T
    self_scores = np.diag(sim)

    np.fill_diagonal(sim, -1e9)
    best_neg_idx = np.argmax(sim, axis=1)
    best_neg_score = sim[np.arange(sim.shape[0]), best_neg_idx]

    ordering = np.argsort(self_scores)[:top_n]
    mismatches = []
    for idx in ordering:
        item = items[idx]
        mismatches.append(
            {
                "id": item["id"],
                "caption": item["caption"],
                "category": item["category"],
                "thumbnail": item["thumbnail"],
                "self_similarity": float(self_scores[idx]),
                "hard_negative_id": items[int(best_neg_idx[idx])]["id"],
                "hard_negative_similarity": float(best_neg_score[idx]),
            }
        )

    img_clusters = labels["finetuned_image"]
    txt_clusters = labels["finetuned_text"]
    n_clusters = int(max(img_clusters.max(), txt_clusters.max()) + 1)

    heatmap = np.zeros((n_clusters, n_clusters), dtype=np.float32)
    counts = np.zeros((n_clusters, n_clusters), dtype=np.float32)

    for i in range(sim.shape[0]):
        c_i = int(img_clusters[i])
        for j in range(sim.shape[1]):
            c_j = int(txt_clusters[j])
            heatmap[c_i, c_j] += float((img[i] @ txt[j]))
            counts[c_i, c_j] += 1.0

    heatmap = np.divide(heatmap, np.clip(counts, 1.0, None)).tolist()

    return {
        "misaligned": mismatches,
        "cluster_heatmap": heatmap,
    }


def save_thumbnails(samples: List[Dict], thumb_dir: Path, base_url: str = "/static/thumbnails"):
    thumb_dir.mkdir(parents=True, exist_ok=True)
    for s in samples:
        thumb_path = thumb_dir / f"{s['id']}.jpg"
        image = s["image"].copy()
        image.thumbnail((160, 160))
        image.save(thumb_path, format="JPEG", quality=80)
        s["thumbnail"] = f"{base_url}/{s['id']}.jpg"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--clusters", type=int, default=12)
    parser.add_argument("--errors", type=int, default=120)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--model-name", type=str, default="ViT-B-16")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b88k")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="prototype",
        help="Prefix for cache files. Writes <prefix>_embeddings.npz and <prefix>_metadata.json",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--thumb-dir", type=Path, default=THUMB_DIR)
    parser.add_argument(
        "--thumbnail-base-url",
        type=str,
        default="/static/thumbnails",
        help="Base URL prefix stored in the metadata for thumbnails.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading samples...")
    samples = load_samples(args.data_dir, args.limit)
    if not samples:
        raise RuntimeError("No samples found in data shards.")

    save_thumbnails(samples, thumb_dir=args.thumb_dir, base_url=args.thumbnail_base_url)

    for sample in samples:
        if not sample.get("category"):
            sample["category"] = infer_category(sample["caption"])

    print(f"Loaded {len(samples)} samples. Building embeddings...")
    embeds = build_embeddings(
        samples,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        model_name=args.model_name,
        pretrained=args.pretrained,
    )

    print("Computing UMAP and clusters...")
    captions = [s["caption"] for s in samples]
    categories = [s["category"] for s in samples]
    projections, labels, cluster_summaries = compute_umap_and_clusters(
        embeds, captions, categories, args.clusters
    )

    print("Computing error analysis...")
    error_bundle = compute_error_bundle(samples, embeds, labels, args.errors)

    print("Saving cache files...")
    np.savez_compressed(
        args.out_dir / f"{args.output_prefix}_embeddings.npz",
        **embeds,
        **{f"umap_{k}": v for k, v in projections.items()},
        **{f"cluster_{k}": v for k, v in labels.items()},
    )

    items_payload = [
        {
            "id": s["id"],
            "caption": s["caption"],
            "category": s["category"],
            "thumbnail": s["thumbnail"],
            "product_id": s.get("product_id"),
            "image_name": s.get("image_name"),
        }
        for s in samples
    ]

    payload = {
        "meta": {
            "num_items": len(samples),
            "models": list(MODEL_KEYS),
            "spaces": list(SPACE_KEYS),
        },
        "items": items_payload,
        "cluster_summaries": cluster_summaries,
        "error_bundle": error_bundle,
    }

    with (args.out_dir / f"{args.output_prefix}_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f)

    print("Done.")


if __name__ == "__main__":
    main()
