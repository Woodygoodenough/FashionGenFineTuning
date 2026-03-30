from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
try:
    import faiss
except Exception:  # pragma: no cover - the API should still work without FAISS installed.
    faiss = None

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "artifacts" / "demo"
DATA_DIR = ARTIFACT_DIR / "data"
STATIC_DIR = ARTIFACT_DIR / "static"
EMBED_FILE = DATA_DIR / "prototype_embeddings.npz"
META_FILE = DATA_DIR / "prototype_metadata.json"


class PrototypeStore:
    def __init__(self):
        if not EMBED_FILE.exists() or not META_FILE.exists():
            raise FileNotFoundError(
                "Prototype cache missing. Run python/scripts/build_prototype_data.py first."
            )

        with META_FILE.open("r", encoding="utf-8") as f:
            self.meta_payload = json.load(f)

        self.items = self.meta_payload["items"]
        self.item_index = {item["id"]: idx for idx, item in enumerate(self.items)}

        npz = np.load(EMBED_FILE)
        self.embeddings: Dict[str, np.ndarray] = {
            "base_image": npz["base_image"],
            "base_text": npz["base_text"],
            "base_joint": npz["base_joint"],
            "finetuned_image": npz["finetuned_image"],
            "finetuned_text": npz["finetuned_text"],
            "finetuned_joint": npz["finetuned_joint"],
        }
        self.umaps: Dict[str, np.ndarray] = {
            k.replace("umap_", ""): npz[k] for k in npz.files if k.startswith("umap_")
        }
        self.clusters: Dict[str, np.ndarray] = {
            k.replace("cluster_", ""): npz[k] for k in npz.files if k.startswith("cluster_")
        }

        self.cluster_summaries = self.meta_payload["cluster_summaries"]
        self.error_bundle = self.meta_payload["error_bundle"]
        self.hybrid_vectors: Dict[str, np.ndarray] = {
            model: np.concatenate(
                [self.embeddings[f"{model}_image"], self.embeddings[f"{model}_text"]], axis=1
            ).astype(np.float32)
            for model in ("base", "finetuned")
        }
        self.ann_indexes = self._build_ann_indexes()
        self.meta = {
            **self.meta_payload["meta"],
            "artifact_dir": str(ARTIFACT_DIR),
            "faiss_available": faiss is not None,
            "ann_ready": all(v is not None for v in self.ann_indexes.values()),
        }

    def _build_ann_indexes(self) -> Dict[str, object | None]:
        indexes: Dict[str, object | None] = {}
        for model, vectors in self.hybrid_vectors.items():
            if faiss is None:
                indexes[model] = None
                continue
            index = faiss.IndexHNSWFlat(vectors.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 80
            index.hnsw.efSearch = 64
            index.add(vectors)
            indexes[model] = index
        return indexes

    def _key(self, model: str, space: str) -> str:
        return f"{model}_{space}"

    def map_points(self, model: str, space: str, color_by: str) -> List[dict]:
        key = self._key(model, space)
        umap = self.umaps[key]
        cluster = self.clusters[key]
        emb = self.embeddings[key]

        points = []
        for i, item in enumerate(self.items):
            color_value = item.get(color_by, "n/a") if color_by != "cluster" else int(cluster[i])
            points.append(
                {
                    "id": item["id"],
                    "x": float(umap[i, 0]),
                    "y": float(umap[i, 1]),
                    "category": item["category"],
                    "cluster": int(cluster[i]),
                    "caption": item["caption"],
                    "thumbnail": item["thumbnail"],
                    "self_similarity": float(np.sum(self.embeddings[f"{model}_image"][i] * self.embeddings[f"{model}_text"][i])),
                    "color_value": color_value,
                    "norm": float(np.linalg.norm(emb[i])),
                }
            )
        return points

    def retrieval(
        self,
        item_id: str,
        model: str,
        w_image: float,
        w_text: float,
        k: int = 15,
        method: str = "exact",
    ):
        if item_id not in self.item_index:
            raise KeyError(item_id)
        idx = self.item_index[item_id]

        img = self.embeddings[f"{model}_image"]
        txt = self.embeddings[f"{model}_text"]

        query_img = img[idx]
        query_txt = txt[idx]

        sim_img = img @ query_img
        sim_txt = txt @ query_txt
        score = w_image * sim_img + w_text * sim_txt
        if method == "ann":
            index = self.ann_indexes.get(model)
            if index is None:
                raise RuntimeError("FAISS ANN index is unavailable. Install faiss-cpu or faiss-gpu.")
            query = np.concatenate([w_image * query_img, w_text * query_txt], axis=0).astype(np.float32)
            _, neighbors = index.search(query[None, :], k + 1)
            order = neighbors[0].tolist()
        else:
            order = np.argsort(score)[::-1].tolist()
        results = []
        for j in order[: k + 1]:
            if j == idx:
                continue
            item = self.items[j]
            results.append(
                {
                    "id": item["id"],
                    "caption": item["caption"],
                    "category": item["category"],
                    "thumbnail": item["thumbnail"],
                    "score": float(score[j]),
                    "image_similarity": float(sim_img[j]),
                    "text_similarity": float(sim_txt[j]),
                }
            )
            if len(results) >= k:
                break

        return {
            "query": self.items[idx],
            "neighbors": results,
            "weights": {"image": w_image, "text": w_text},
            "method": method,
        }

    def cluster_panel(self, model: str, space: str):
        key = self._key(model, space)
        summaries = self.cluster_summaries.get(key, [])
        enriched = []
        for row in summaries:
            reps = [self.items[i] for i in row["representative_indices"]]
            enriched.append({**row, "representatives": reps})
        return enriched

    def error_panel(self, limit: int):
        return {
            "misaligned": self.error_bundle["misaligned"][:limit],
            "cluster_heatmap": self.error_bundle["cluster_heatmap"],
        }
