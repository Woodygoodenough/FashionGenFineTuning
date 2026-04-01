#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
META_FILE = ROOT / "artifacts" / "demo" / "data" / "prototype_metadata.json"
STATIC_DIR = ROOT / "artifacts" / "demo" / "static"
OUT_DIR = ROOT / "nextjs" / "public" / "demo"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=300)
    args = parser.parse_args()

    payload = json.loads(META_FILE.read_text())
    items = payload["items"][: args.limit]

    out_thumb_dir = OUT_DIR / "thumbnails"
    out_thumb_dir.mkdir(parents=True, exist_ok=True)

    catalog = []
    for item in items:
        thumb_rel = item["thumbnail"].replace("/static/", "")
        src = STATIC_DIR / thumb_rel
        dst = OUT_DIR / thumb_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)

        words = (item.get("caption") or "").split()
        title = " ".join(words[:8]).strip().rstrip(".")
        if not title:
            title = item.get("category") or item["id"]

        catalog.append(
            {
                "id": item["id"],
                "title": title,
                "category": item.get("category") or "Fashion",
                "caption": item.get("caption") or "",
                "image": f"/demo/{thumb_rel}",
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "catalog.json").write_text(json.dumps({"items": catalog}, indent=2))
    (OUT_DIR / "manifest.txt").write_text("\n".join(x["id"] for x in catalog) + "\n")


if __name__ == "__main__":
    main()
