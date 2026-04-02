from __future__ import annotations

import json
import os
import time
from pathlib import Path

import faiss
import numpy as np
import open_clip
import torch


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class EmbeddingSearchIndex:
    def __init__(self) -> None:
        base_dir = Path(os.environ.get("DEMO_INDEX_DIR", Path(__file__).resolve().parents[2] / "artifacts" / "generated" / "demo_index"))
        self.base_dir = base_dir
        self.catalog_path = Path(os.environ.get("DEMO_INDEX_CATALOG", base_dir / "catalog.json"))
        self.embedding_path = Path(os.environ.get("DEMO_INDEX_EMBEDDINGS", base_dir / "image_embeddings.npy"))
        self.index_path = Path(os.environ.get("DEMO_INDEX_ANN", base_dir / "image_ann.index"))
        self.checkpoint_path = Path(os.environ.get("DEMO_MODEL_CHECKPOINT", ""))
        self.model_name = os.environ.get("DEMO_MODEL_NAME", "ViT-B-16")
        self.pretrained = os.environ.get("DEMO_MODEL_PRETRAINED", "laion2b_s34b_b88k")
        self.ef_search = int(os.environ.get("DEMO_ANN_EF_SEARCH", "128"))
        self.device = pick_device()

        if (
            not self.catalog_path.exists()
            or not self.embedding_path.exists()
            or not self.index_path.exists()
            or not self.checkpoint_path.exists()
        ):
            raise FileNotFoundError("Embedding search artifacts are missing")

        payload = json.loads(self.catalog_path.read_text())
        self.items = payload["items"]
        self.image_embeddings = np.load(self.embedding_path).astype(np.float32)
        self.index = faiss.read_index(str(self.index_path))
        if hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = self.ef_search

        model, _, _ = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.eval().to(self.device)

        self.model = model
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.benchmark_queries = self._build_benchmark_queries()

    def _build_benchmark_queries(self) -> list[str]:
        seeds = [
            "red wool hat",
            "black leather handbag",
            "white cotton shirt",
            "blue denim jeans",
            "cream evening dress",
            "brown suede boots",
            "grey knit sweater",
            "navy tailored blazer",
            "silver chain necklace",
            "green pleated skirt",
            "beige trench coat",
            "running sneakers in white",
            "pink silk blouse",
            "black ankle boots",
            "structured tote bag",
            "oversized hoodie",
            "cropped jacket",
            "printed summer dress",
            "straight leg pants",
            "wool scarf",
        ]
        queries: list[str] = []
        while len(queries) < 100:
            queries.extend(seeds)
        return queries[:100]

    def _encode_query(self, query: str) -> np.ndarray:
        tokens = self.tokenizer([query])
        with torch.inference_mode():
            text_features = self.model.encode_text(tokens.to(self.device))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features[0].detach().cpu().numpy().astype(np.float32)

    def health(self) -> dict:
        return {
            "status": "ok",
            "items": len(self.items),
            "dim": int(self.image_embeddings.shape[1]),
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "checkpoint": self.checkpoint_path.name,
            "search_backend": "faiss_hnsw",
            "ef_search": self.ef_search,
        }

    def _format_items(self, indices: list[int], scores: list[float]) -> list[dict]:
        items = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            item = dict(self.items[idx])
            item["score"] = float(score)
            items.append(item)
        return items

    def search_exact(self, query: str, k: int = 10) -> dict:
        k = min(k, len(self.items))
        query_vec = self._encode_query(query)
        scores = self.image_embeddings @ query_vec
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return {
            "query": query,
            "items": self._format_items(top_idx.tolist(), scores[top_idx].tolist()),
        }

    def search_ann(self, query: str, k: int = 10) -> dict:
        k = min(k, len(self.items))
        query_vec = self._encode_query(query)
        scores, indices = self.index.search(query_vec.reshape(1, -1), k)
        return {
            "query": query,
            "items": self._format_items(indices[0].tolist(), scores[0].tolist()),
        }

    def search(self, query: str, k: int = 10) -> dict:
        return self.search_ann(query, k)

    def benchmark(self, query_count: int = 100, k: int = 10) -> dict:
        queries = self.benchmark_queries[:query_count]

        ann_started = time.perf_counter()
        for query in queries:
            self.search_ann(query, k)
        ann_total_ms = (time.perf_counter() - ann_started) * 1000.0

        exact_started = time.perf_counter()
        for query in queries:
            self.search_exact(query, k)
        exact_total_ms = (time.perf_counter() - exact_started) * 1000.0

        return {
            "query_count": len(queries),
            "k": k,
            "ann_total_ms": ann_total_ms,
            "ann_avg_ms": ann_total_ms / len(queries),
            "exact_total_ms": exact_total_ms,
            "exact_avg_ms": exact_total_ms / len(queries),
            "speedup": exact_total_ms / ann_total_ms if ann_total_ms > 0 else None,
        }
