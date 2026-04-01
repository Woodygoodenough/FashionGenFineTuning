from __future__ import annotations

import json
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG_FILE = ROOT / "nextjs" / "public" / "demo" / "catalog.json"


def _tokenize(text: str) -> list[str]:
    return [t for t in re.sub(r"[^a-z0-9\s]", " ", text.lower()).split() if len(t) > 1]


class DemoCatalog:
    def __init__(self):
        env_path = os.environ.get("DEMO_CATALOG_FILE")
        if env_path:
            catalog_file = Path(env_path)
        else:
            packaged = Path(__file__).resolve().parents[1] / "catalog.json"
            catalog_file = packaged if packaged.exists() else DEFAULT_CATALOG_FILE

        if not catalog_file.exists():
            raise FileNotFoundError(
                "Demo catalog missing. Run python/scripts/export_demo_catalog.py first."
            )
        payload = json.loads(catalog_file.read_text())
        self.items = payload["items"]

    def search(self, query: str, k: int = 10) -> dict:
        tokens = _tokenize(query)
        if not tokens:
            return {"query": query, "items": self.items[:k]}

        scored = []
        for item in self.items:
            haystack = f"{item['title']} {item['category']} {item['caption']}".lower()
            score = 0
            for token in tokens:
                if token in item["title"].lower():
                    score += 6
                if token in item["category"].lower():
                    score += 4
                if token in haystack:
                    score += 2
            if query.strip().lower() in haystack:
                score += 8
            scored.append((score, item))

        scored.sort(key=lambda row: row[0], reverse=True)
        return {"query": query, "items": [item for _, item in scored[:k]]}
