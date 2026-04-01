from __future__ import annotations

import json

from .demo_search import DemoCatalog


catalog = DemoCatalog()


def _response(status: int, payload: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload),
    }


def handler(event, _context):
    method = (event or {}).get("requestContext", {}).get("http", {}).get("method", "GET")
    if method == "OPTIONS":
        return _response(200, {"ok": True})

    raw_path = (event or {}).get("rawPath", "")
    query = ((event or {}).get("queryStringParameters") or {}).get("query", "")
    try:
        k = int(((event or {}).get("queryStringParameters") or {}).get("k", 10))
    except (TypeError, ValueError):
        k = 10
    k = max(1, min(k, 20))

    if raw_path in {"", "/", "/api/health"}:
        return _response(200, {"status": "ok", "items": len(catalog.items)})
    if raw_path == "/api/demo/search":
        return _response(200, catalog.search(query, k))
    if raw_path == "/api/demo/catalog":
        return _response(200, {"items": catalog.items})
    return _response(404, {"error": "Not found", "path": raw_path})
