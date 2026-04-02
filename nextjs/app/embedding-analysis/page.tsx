import { TitleCard } from "../../components/TitleCard";

export default function EmbeddingAnalysisPage() {
  return (
    <main className="page-shell">
      <section className="placeholder-page">
        <div className="placeholder-left">
          <TitleCard
            eyebrow="Analysis"
            title="Embedding analysis"
            description="Reserved for interactive structure diagnostics over zero-shot, retrieval-only, and regularized checkpoints."
          />

          <section className="placeholder-card">
            <div className="placeholder-card-head">
              <span className="stat-label">Planned views</span>
              <strong>Interactive diagnostics</strong>
            </div>
            <ul className="placeholder-list">
              <li>2D embedding map with model switcher and category legend</li>
              <li>Cluster purity and neighborhood consistency summaries</li>
              <li>Sample-level inspection for local neighborhoods and failure cases</li>
            </ul>
          </section>

          <section className="placeholder-card">
            <div className="placeholder-card-head">
              <span className="stat-label">Data contract</span>
              <strong>Expected inputs</strong>
            </div>
            <div className="placeholder-keyvals">
              <div>
                <span className="stat-label">Embeddings</span>
                <strong>image, text, and joint vectors</strong>
              </div>
              <div>
                <span className="stat-label">Metadata</span>
                <strong>category, item id, prompt, split</strong>
              </div>
              <div>
                <span className="stat-label">Metrics</span>
                <strong>kNN purity, silhouette, DBI</strong>
              </div>
            </div>
          </section>
        </div>

        <div className="placeholder-right">
          <section className="placeholder-hero">
            <div className="placeholder-hero-grid">
              <article className="placeholder-panel">
                <span className="stat-label">Primary canvas</span>
                <strong>Embedding map</strong>
                <div className="placeholder-wireframe scatter-wireframe" />
              </article>
              <article className="placeholder-panel">
                <span className="stat-label">Summary rail</span>
                <strong>Metric cards</strong>
                <div className="placeholder-wireframe metric-wireframe" />
              </article>
              <article className="placeholder-panel">
                <span className="stat-label">Detail view</span>
                <strong>Neighborhood inspector</strong>
                <div className="placeholder-wireframe detail-wireframe" />
              </article>
            </div>
          </section>
        </div>
      </section>
    </main>
  );
}
