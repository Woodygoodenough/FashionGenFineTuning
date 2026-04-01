from __future__ import annotations

import os

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from .demo_search import DemoCatalog
from .embedding_search import EmbeddingSearchIndex


def build_app() -> FastAPI:
    search_mode = os.environ.get("DEMO_SEARCH_MODE", "lexical")
    if search_mode == "embedding":
        search_impl = EmbeddingSearchIndex()
    else:
        search_impl = DemoCatalog()

    app = FastAPI(title="FashionGen Demo API", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "https://demo.woodygoodenough.com",
        ],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health():
        if hasattr(search_impl, "health"):
            return search_impl.health()
        return {"status": "ok", "items": len(search_impl.items), "mode": "lexical"}

    @app.get("/api/demo/catalog")
    def catalog():
        return {"items": search_impl.items}

    @app.get("/api/demo/search")
    def search(query: str = Query(""), k: int = Query(10, ge=1, le=20)):
        return search_impl.search(query, k)

    return app


app = build_app()
