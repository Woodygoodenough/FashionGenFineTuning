from __future__ import annotations

import json
import os
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

    def search(self, query: str, k: int = 10) -> dict:
        k = min(k, len(self.items))
        tokens = self.tokenizer([query])
        with torch.inference_mode():
            text_features = self.model.encode_text(tokens.to(self.device))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        query_vec = text_features[0].detach().cpu().numpy().astype(np.float32)
        scores, indices = self.index.search(query_vec.reshape(1, -1), k)

        items = []
        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx < 0:
                continue
            item = dict(self.items[idx])
            item["score"] = float(score)
            items.append(item)
        return {"query": query, "items": items}
