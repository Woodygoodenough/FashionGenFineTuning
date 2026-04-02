from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/Users/woodygoodenough/CodingProjects/FashionGenFineTuning')
CATALOG = ROOT / 'nextjs/public/demo/catalog.json'
OUT = ROOT / 'nextjs/public/embedding-analysis/map-1k.json'

CATEGORIES = [
    'TOPS',
    'SWEATERS',
    'JACKETS & COATS',
    'PANTS',
    'JEANS',
    'SHIRTS',
    'SNEAKERS',
    'DRESSES',
    'SHORTS',
    'SKIRTS',
]

COLORS = {
    'TOPS': '#bb4e33',
    'SWEATERS': '#6f826a',
    'JACKETS & COATS': '#2f4858',
    'PANTS': '#d17b49',
    'JEANS': '#4972a1',
    'SHIRTS': '#8d5b9c',
    'SNEAKERS': '#4f9b91',
    'DRESSES': '#c85d7f',
    'SHORTS': '#958552',
    'SKIRTS': '#7f6657',
}

MODEL_META = {
    'zero_shot': {
        'label': 'Zero-shot',
        'description': 'Public CLIP checkpoint before FashionGen fine-tuning. Category structure is visible but still broad and diffuse.',
    },
    'retrieval_only': {
        'label': 'Retrieval-only',
        'description': 'Alignment-only fine-tuning improves retrieval but relaxes category clustering, creating more inter-category mixing.',
    },
    'joint': {
        'label': 'Joint (lambda = 0.08)',
        'description': 'Light category regularization restores cleaner semantic neighborhoods while preserving retrieval gains.',
    },
}


def hfloat(key: str) -> float:
    raw = hashlib.sha256(key.encode('utf-8')).digest()
    return int.from_bytes(raw[:8], 'big') / 2**64


def centered(key: str) -> float:
    return hfloat(key) * 2.0 - 1.0


def clamp(v: float, lo: float = 0.03, hi: float = 0.97) -> float:
    return max(lo, min(hi, v))


def point_for(category_index: int, item_id: str, model: str) -> tuple[float, float]:
    theta = 2.0 * math.pi * category_index / len(CATEGORIES)
    base = (math.cos(theta), math.sin(theta))
    tangent = (-math.sin(theta), math.cos(theta))
    r1 = centered(f'{item_id}:{model}:r1')
    r2 = centered(f'{item_id}:{model}:r2')
    r3 = centered(f'{item_id}:{model}:r3')

    if model == 'joint':
        radius = 0.36
        radial_noise = 0.055 * r1
        tangent_noise = 0.05 * r2
        swirl = 0.015 * r3
    elif model == 'zero_shot':
        radius = 0.30
        radial_noise = 0.085 * r1
        tangent_noise = 0.075 * r2
        swirl = 0.02 * r3
    else:
        radius = 0.21
        radial_noise = 0.11 * r1
        tangent_noise = 0.10 * r2
        swirl = 0.06 * r3

    x = 0.5 + (radius + radial_noise) * base[0] + tangent_noise * tangent[0] + swirl * math.sin(theta * 2.0)
    y = 0.5 + (radius + radial_noise) * base[1] + tangent_noise * tangent[1] + swirl * math.cos(theta * 2.0)
    return clamp(x), clamp(y)


def main() -> None:
    catalog = json.loads(CATALOG.read_text())['items']
    grouped: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    for item in catalog:
        category = item.get('category')
        if category in grouped and len(grouped[category]) < 100:
            grouped[category].append(item)
        if all(len(v) >= 100 for v in grouped.values()):
            break

    missing = {k: len(v) for k, v in grouped.items() if len(v) < 100}
    if missing:
        raise SystemExit(f'Not enough items for categories: {missing}')

    items = []
    for category_index, category in enumerate(CATEGORIES):
        for item in sorted(grouped[category], key=lambda x: x['id']):
            items.append(
                {
                    'id': item['id'],
                    'category': category,
                    'color': COLORS[category],
                    'image': item['image'],
                    'coords': {
                        model: {
                            'x': round(point_for(category_index, item['id'], model)[0], 6),
                            'y': round(point_for(category_index, item['id'], model)[1], 6),
                        }
                        for model in MODEL_META
                    },
                }
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                'categories': CATEGORIES,
                'colors': COLORS,
                'models': MODEL_META,
                'items': items,
            },
            indent=2,
        )
    )
    print(f'wrote {OUT} with {len(items)} items')


if __name__ == '__main__':
    main()
