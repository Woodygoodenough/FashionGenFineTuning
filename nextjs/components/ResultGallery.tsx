"use client";

import Image from "next/image";
import type { DemoItem } from "./demo-catalog";

type ResultGalleryProps = {
  query: string;
  topK: number;
  results: DemoItem[];
};

export function ResultGallery({ query, topK, results }: ResultGalleryProps) {
  const visible = results.slice(0, topK);

  return (
    <section className="gallery-shell">
      <div className="gallery-toolbar">
        <div className="gallery-stat">
          <span className="stat-label">Query</span>
          <strong>{query.trim() || "No query entered yet"}</strong>
        </div>
        <div className="gallery-stat">
          <span className="stat-label">Showing</span>
          <strong>{visible.length} result{visible.length === 1 ? "" : "s"}</strong>
        </div>
      </div>

      <div className="result-grid">
        {visible.map((item, index) => (
          <article className="result-card" key={item.id}>
            <div className="result-visual">
              <span className="result-rank">#{index + 1}</span>
              <Image
                className="result-image"
                src={item.image}
                alt={item.title}
                fill
                sizes="(max-width: 980px) 100vw, 40vw"
              />
            </div>
            <div className="result-copy">
              <div className="result-meta">
                <span>{item.category}</span>
                <span>{item.id}</span>
              </div>
              <h3>{item.title}</h3>
              <p>{item.caption}</p>
              <div className="result-footer">
                <span>FashionGen</span>
                <button type="button">Inspect</button>
              </div>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
