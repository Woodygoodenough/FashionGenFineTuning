from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .data_store import PrototypeStore, STATIC_DIR

app = FastAPI(title="Fashion Multimodal Retrieval API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store: PrototypeStore | None
load_error: str | None = None
try:
    store = PrototypeStore()
except Exception as exc:
    store = None
    load_error = str(exc)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def require_store() -> PrototypeStore:
    if store is None:
        detail = (
            "Prototype cache unavailable. Run backend/scripts/build_prototype_data.py first. "
            f"Details: {load_error}"
        )
        raise HTTPException(status_code=503, detail=detail)
    return store


@app.get("/api/meta")
def get_meta():
    return require_store().meta


@app.get("/api/health")
def get_health():
    active_store = require_store()
    return {"status": "ok", "items": len(active_store.items), "ann_ready": active_store.meta["ann_ready"]}


@app.get("/api/map")
def get_map(
    model: str = Query("finetuned", pattern="^(base|finetuned)$"),
    space: str = Query("joint", pattern="^(image|text|joint)$"),
    color_by: str = Query("cluster", pattern="^(cluster|category)$"),
):
    active_store = require_store()
    return {
        "model": model,
        "space": space,
        "color_by": color_by,
        "points": active_store.map_points(model, space, color_by),
    }


@app.get("/api/retrieval/{item_id}")
def get_retrieval(
    item_id: str,
    model: str = Query("finetuned", pattern="^(base|finetuned)$"),
    w_image: float = Query(0.5, ge=0.0, le=1.0),
    w_text: float = Query(0.5, ge=0.0, le=1.0),
    method: str = Query("exact", pattern="^(exact|ann)$"),
    k: int = Query(15, ge=3, le=50),
):
    if w_image == 0 and w_text == 0:
        raise HTTPException(status_code=400, detail="At least one weight must be > 0.")
    scale = w_image + w_text
    w_image, w_text = w_image / scale, w_text / scale

    active_store = require_store()
    try:
        return active_store.retrieval(item_id, model, w_image, w_text, k, method)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown item_id")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/api/clusters")
def get_clusters(
    model: str = Query("finetuned", pattern="^(base|finetuned)$"),
    space: str = Query("joint", pattern="^(image|text|joint)$"),
):
    active_store = require_store()
    return {"model": model, "space": space, "clusters": active_store.cluster_panel(model, space)}


@app.get("/api/errors")
def get_errors(limit: int = Query(60, ge=5, le=200)):
    return require_store().error_panel(limit)
