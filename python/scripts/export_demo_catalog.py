#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHARDS_DIR = (ROOT.parent / "clip_finetuning" / "data_aug").resolve()
OUT_DIR = ROOT / "nextjs" / "public" / "demo"


def extract_catalog(shards_dir: Path, limit: int, thumb_size: int) -> dict:
    shards = sorted(shards_dir.glob("clip_dataset_train_aug.*.tar"))
    if not shards:
        shards = sorted(shards_dir.glob("clip_dataset_train.*.tar"))
    if not shards:
        raise FileNotFoundError(f"No FashionGen tar shards found in {shards_dir}")

    thumb_dir = OUT_DIR / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    items = []
    seen = set()

    for shard in shards:
        if len(items) >= limit:
            break
        with tarfile.open(shard, "r") as tf:
            members = set(tf.getnames())
            image_names = [name for name in members if name.endswith(".jpg")]
            for image_name in sorted(image_names):
                if len(items) >= limit:
                    break

                stem = image_name[:-4]
                txt_name = stem + ".txt0.txt"
                if txt_name not in members:
                    txt_name = stem + ".txt"
                if txt_name not in members:
                    continue

                meta_name = stem + ".meta.json"
                image_file = tf.extractfile(image_name)
                text_file = tf.extractfile(txt_name)
                if image_file is None or text_file is None:
                    continue

                item_id = f"{Path(shard.name).stem}_{Path(stem).name}"
                if item_id in seen:
                    continue

                caption = text_file.read().decode("utf-8", errors="ignore").strip()
                if not caption:
                    continue

                category = "Fashion"
                if meta_name in members:
                    meta_file = tf.extractfile(meta_name)
                    if meta_file is not None:
                        try:
                            meta = json.loads(meta_file.read())
                            category = meta.get("category") or category
                        except Exception:
                            pass

                thumb_name = f"{item_id}.jpg"
                thumb_path = thumb_dir / thumb_name
                if not thumb_path.exists():
                    image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
                    image.thumbnail((thumb_size, thumb_size))
                    image.save(thumb_path, format="JPEG", quality=85, optimize=True)

                words = caption.split()
                title = " ".join(words[:8]).strip().rstrip(".") or category
                items.append(
                    {
                        "id": item_id,
                        "title": title,
                        "category": category,
                        "caption": caption,
                        "image": f"/demo/thumbnails/{thumb_name}",
                    }
                )
                seen.add(item_id)

    return {"items": items}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--thumb-size", type=int, default=384)
    parser.add_argument("--shards-dir", type=Path, default=DEFAULT_SHARDS_DIR)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog = extract_catalog(args.shards_dir, args.limit, args.thumb_size)
    (OUT_DIR / "catalog.json").write_text(json.dumps(catalog, indent=2))
    (OUT_DIR / "manifest.txt").write_text("\n".join(item["id"] for item in catalog["items"]) + "\n")
    print(f"exported {len(catalog['items'])} demo items to {OUT_DIR}")


if __name__ == "__main__":
    main()
