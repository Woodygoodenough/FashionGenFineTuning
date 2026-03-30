#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import open_clip
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_joint_clip import JointCLIPModel, create_loader, resolve_shards

try:
    import faiss
except Exception as exc:
    raise SystemExit("faiss is required") from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare exact, pure ANN, and ANN+rerank on FashionGen retrieval.")
    p.add_argument("--shards-dir", type=Path, required=True)
    p.add_argument("--shards-glob", default="clip_dataset_train_aug.*.tar")
    p.add_argument("--model-name", default="ViT-B-16")
    p.add_argument("--pretrained", default="laion2b_s34b_b88k")
    p.add_argument("--model-cache-dir", type=Path, required=True)
    p.add_argument("--finetuned-checkpoint", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--sample-count", type=int, default=60000)
    p.add_argument("--sample-size", type=int, default=30000)
    p.add_argument("--query-count", type=int, default=200)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--hnsw-m", type=int, default=32)
    p.add_argument("--ef-construction", type=int, default=80)
    p.add_argument("--ef-search", type=int, default=32)
    p.add_argument("--rerank-k", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def load_model(args: argparse.Namespace, device: torch.device):
    clip_model, preprocess, _ = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained,
        cache_dir=str(args.model_cache_dir),
    )
    ckpt = torch.load(args.finetuned_checkpoint, map_location=device, weights_only=False)
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
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model_name)
    return model, tokenizer, preprocess


@torch.no_grad()
def encode_subset(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, preprocess = load_model(args, device)
    shards = resolve_shards(args.shards_dir, args.shards_glob, None)
    loader = create_loader(
        shards=shards,
        image_key="jpg",
        caption_keys=["txt0.txt", "txt1.txt", "txt2.txt"],
        meta_key="meta.json",
        preprocess=preprocess,
        category_to_id={},
        caption_mode="primary",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shardshuffle=False,
    )
    images_all, texts_all = [], []
    seen = 0
    for images, texts, _ in loader:
        images = images.to(device, non_blocking=True)
        tokens = tokenizer(list(texts)).to(device)
        img_emb, txt_emb, _ = model(images, tokens)
        images_all.append(torch.nn.functional.normalize(img_emb, dim=-1).cpu().numpy().astype(np.float32))
        texts_all.append(torch.nn.functional.normalize(txt_emb, dim=-1).cpu().numpy().astype(np.float32))
        seen += images.size(0)
        if seen >= args.sample_count:
            break
    image = np.concatenate(images_all, axis=0)[: args.sample_count]
    text = np.concatenate(texts_all, axis=0)[: args.sample_count]
    return image, text


def topk_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
    order = np.argpartition(scores, -k, axis=1)[:, -k:]
    row_ids = np.arange(scores.shape[0])[:, None]
    sorted_local = np.argsort(scores[row_ids, order], axis=1)[:, ::-1]
    return order[row_ids, sorted_local]


def exact_search(index_vectors: np.ndarray, queries: np.ndarray, k: int):
    start = time.perf_counter()
    scores = queries @ index_vectors.T
    order = topk_from_scores(scores, k)
    elapsed = time.perf_counter() - start
    return order, elapsed


def ann_search(index_vectors: np.ndarray, queries: np.ndarray, k: int, hnsw_m: int, ef_construction: int, ef_search: int, rerank_k: int):
    retrieve_k = max(k, rerank_k)
    index = faiss.IndexHNSWFlat(index_vectors.shape[1], hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(index_vectors)
    start = time.perf_counter()
    _, order = index.search(queries, retrieve_k)
    if rerank_k > k:
        reranked = []
        for query, candidate_ids in zip(queries, order):
            valid_ids = candidate_ids[candidate_ids >= 0]
            candidate_scores = index_vectors[valid_ids] @ query
            reranked.append(valid_ids[np.argsort(candidate_scores)[::-1][:k]].astype(np.int64))
        order = np.stack(reranked, axis=0)
    else:
        order = order[:, :k]
    elapsed = time.perf_counter() - start
    return order, elapsed


def overlap_at_k(exact_topk: np.ndarray, approx_topk: np.ndarray) -> float:
    vals = []
    for a, b in zip(exact_topk, approx_topk):
        vals.append(len(set(a.tolist()) & set(b.tolist())) / len(a))
    return float(np.mean(vals))


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    image, text = encode_subset(args)
    encoded_count = int(len(image))
    sample_size = min(args.sample_size, encoded_count)
    index_vectors = image[:sample_size]
    query_ids = rng.choice(sample_size, size=min(args.query_count, sample_size), replace=False)
    queries = text[query_ids]

    exact_topk, exact_elapsed = exact_search(index_vectors, queries, args.k)
    ann_topk, ann_elapsed = ann_search(index_vectors, queries, args.k, args.hnsw_m, args.ef_construction, args.ef_search, 0)
    rerank_topk, rerank_elapsed = ann_search(index_vectors, queries, args.k, args.hnsw_m, args.ef_construction, args.ef_search, args.rerank_k)

    payload = {
        "config": {
            "requested_sample_count": args.sample_count,
            "encoded_count": encoded_count,
            "sample_size": sample_size,
            "query_count": int(len(query_ids)),
            "k": args.k,
            "hnsw_m": args.hnsw_m,
            "ef_construction": args.ef_construction,
            "ef_search": args.ef_search,
            "rerank_k": args.rerank_k,
        },
        "exact": {
            "avg_ms": float(exact_elapsed * 1000.0 / len(query_ids)),
        },
        "pure_ann": {
            "avg_ms": float(ann_elapsed * 1000.0 / len(query_ids)),
            "speedup": float(exact_elapsed / ann_elapsed) if ann_elapsed > 0 else None,
            "overlap_at_k": overlap_at_k(exact_topk, ann_topk),
        },
        "ann_rerank": {
            "avg_ms": float(rerank_elapsed * 1000.0 / len(query_ids)),
            "speedup": float(exact_elapsed / rerank_elapsed) if rerank_elapsed > 0 else None,
            "overlap_at_k": overlap_at_k(exact_topk, rerank_topk),
        },
    }
    (args.out_dir / 'fashiongen_ann_compare.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
