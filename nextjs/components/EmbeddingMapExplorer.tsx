"use client";

import { useEffect, useMemo, useState } from "react";

type ModelKey = "zero_shot" | "retrieval_only" | "joint";

type MapItem = {
  id: string;
  category: string;
  color: string;
  image: string;
  coords: Record<ModelKey, { x: number; y: number }>;
};

type MapPayload = {
  categories: string[];
  colors: Record<string, string>;
  models: Record<ModelKey, { label: string; description: string }>;
  items: MapItem[];
};

type HoverState = {
  item: MapItem;
  x: number;
  y: number;
} | null;

const MODEL_ORDER: ModelKey[] = ["zero_shot", "retrieval_only", "joint"];

export function EmbeddingMapExplorer() {
  const [payload, setPayload] = useState<MapPayload | null>(null);
  const [model, setModel] = useState<ModelKey>("zero_shot");
  const [activeCategories, setActiveCategories] = useState<string[]>([]);
  const [hovered, setHovered] = useState<HoverState>(null);

  useEffect(() => {
    let cancelled = false;
    fetch("/embedding-analysis/map-1k.json")
      .then((response) => response.json())
      .then((data: MapPayload) => {
        if (cancelled) return;
        setPayload(data);
        setActiveCategories(data.categories);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const visibleItems = useMemo(() => {
    if (!payload) return [];
    const active = new Set(activeCategories);
    return payload.items.filter((item) => active.has(item.category));
  }, [payload, activeCategories]);

  const counts = useMemo(() => {
    const out = new Map<string, number>();
    for (const item of visibleItems) {
      out.set(item.category, (out.get(item.category) ?? 0) + 1);
    }
    return out;
  }, [visibleItems]);

  const toggleCategory = (category: string) => {
    setActiveCategories((current) => {
      if (current.includes(category)) {
        if (current.length === 1) return current;
        return current.filter((value) => value !== category);
      }
      return [...current, category];
    });
  };

  if (!payload) {
    return <section className="embedding-shell">Loading embedding map…</section>;
  }

  return (
    <section className="embedding-shell">
      <div className="embedding-left-rail">
        <section className="embedding-card">
          <div className="embedding-card-head">
            <span className="stat-label">Model view</span>
            <strong>{payload.models[model].label}</strong>
          </div>
          <div className="embedding-tab-row">
            {MODEL_ORDER.map((key) => (
              <button
                key={key}
                type="button"
                className={`embedding-tab ${model === key ? "is-active" : ""}`}
                onClick={() => setModel(key)}
              >
                {payload.models[key].label}
              </button>
            ))}
          </div>
          <p className="embedding-copy">{payload.models[model].description}</p>
        </section>

        <section className="embedding-card">
          <div className="embedding-card-head">
            <span className="stat-label">Category filter</span>
            <strong>{activeCategories.length} active</strong>
          </div>
          <div className="embedding-filter-grid">
            {payload.categories.map((category) => {
              const active = activeCategories.includes(category);
              return (
                <button
                  key={category}
                  type="button"
                  className={`embedding-filter-chip ${active ? "is-active" : ""}`}
                  onClick={() => toggleCategory(category)}
                >
                  <span
                    className="embedding-chip-dot"
                    style={{ backgroundColor: payload.colors[category] }}
                  />
                  <span>{category}</span>
                  <span className="embedding-chip-count">{counts.get(category) ?? 0}</span>
                </button>
              );
            })}
          </div>
        </section>

        <section className="embedding-card embedding-stat-grid">
          <article>
            <span className="stat-label">Rendered points</span>
            <strong>{visibleItems.length}</strong>
          </article>
          <article>
            <span className="stat-label">Categories</span>
            <strong>{payload.categories.length}</strong>
          </article>
          <article>
            <span className="stat-label">Shared sample</span>
            <strong>1,000 items</strong>
          </article>
        </section>
      </div>

      <section className="embedding-canvas-card">
        <div className="embedding-canvas-head">
          <div>
            <span className="stat-label">Interactive map</span>
            <strong>Shared 1k sample across three models</strong>
          </div>
          <p className="embedding-copy">Hover any point to inspect the deployed image asset for that sample.</p>
        </div>

        <div className="embedding-canvas-wrap">
          <svg className="embedding-canvas" viewBox="0 0 880 700" role="img" aria-label="Embedding map">
            <g className="embedding-grid">
              {[0.2, 0.4, 0.6, 0.8].map((t) => (
                <g key={t}>
                  <line x1={t * 880} y1={0} x2={t * 880} y2={700} />
                  <line x1={0} y1={t * 700} x2={880} y2={t * 700} />
                </g>
              ))}
            </g>
            {visibleItems.map((item) => {
              const point = item.coords[model];
              const cx = point.x * 880;
              const cy = point.y * 700;
              const isHovered = hovered?.item.id === item.id;
              return (
                <circle
                  key={item.id}
                  cx={cx}
                  cy={cy}
                  r={isHovered ? 6.2 : 4.2}
                  fill={item.color}
                  fillOpacity={isHovered ? 0.96 : 0.82}
                  stroke={isHovered ? "#fffaf4" : "rgba(255,255,255,0.4)"}
                  strokeWidth={isHovered ? 2.2 : 1}
                  onMouseEnter={(event) =>
                    setHovered({
                      item,
                      x: event.clientX,
                      y: event.clientY,
                    })
                  }
                  onMouseMove={(event) =>
                    setHovered((current) =>
                      current?.item.id === item.id
                        ? { item, x: event.clientX, y: event.clientY }
                        : current
                    )
                  }
                  onMouseLeave={() => setHovered((current) => (current?.item.id === item.id ? null : current))}
                />
              );
            })}
          </svg>

          {hovered ? (
            <div
              className="embedding-tooltip"
              style={{ left: hovered.x + 18, top: hovered.y - 38 }}
            >
              <img src={hovered.item.image} alt={hovered.item.category} />
              <div className="embedding-tooltip-copy">
                <span
                  className="embedding-chip-dot"
                  style={{ backgroundColor: hovered.item.color }}
                />
                <strong>{hovered.item.category}</strong>
                <span>{hovered.item.id}</span>
              </div>
            </div>
          ) : null}
        </div>
      </section>
    </section>
  );
}
