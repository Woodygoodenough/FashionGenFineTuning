import { TitleCard } from "../../components/TitleCard";

export default function ModelEvaluationPage() {
  return (
    <main className="page-shell">
      <section className="placeholder-page">
        <div className="placeholder-left">
          <TitleCard
            eyebrow="Evaluation"
            title="Model evaluation"
            description="Reserved for quantitative comparisons across zero-shot, retrieval-only, and regularized models."
          />

          <section className="placeholder-card">
            <div className="placeholder-card-head">
              <span className="stat-label">Planned studies</span>
              <strong>Evaluation modules</strong>
            </div>
            <ul className="placeholder-list">
              <li>Retrieval metrics with model selector and split selector</li>
              <li>Lambda sweep overview with sortable metric deltas</li>
              <li>Checkpoint comparison cards for report-quality summaries</li>
            </ul>
          </section>

          <section className="placeholder-card">
            <div className="placeholder-card-head">
              <span className="stat-label">Metric set</span>
              <strong>Core outputs</strong>
            </div>
            <div className="placeholder-keyvals">
              <div>
                <span className="stat-label">Retrieval</span>
                <strong>R@1, R@5, R@10, aggregate score</strong>
              </div>
              <div>
                <span className="stat-label">Comparison</span>
                <strong>absolute values and deltas</strong>
              </div>
              <div>
                <span className="stat-label">Artifacts</span>
                <strong>JSON summaries and sweep tables</strong>
              </div>
            </div>
          </section>
        </div>

        <div className="placeholder-right">
          <section className="placeholder-hero">
            <div className="placeholder-hero-grid evaluation-grid">
              <article className="placeholder-panel">
                <span className="stat-label">Overview</span>
                <strong>Model scorecards</strong>
                <div className="placeholder-wireframe score-wireframe" />
              </article>
              <article className="placeholder-panel">
                <span className="stat-label">Comparison</span>
                <strong>Metric table</strong>
                <div className="placeholder-wireframe table-wireframe" />
              </article>
              <article className="placeholder-panel">
                <span className="stat-label">Sweep</span>
                <strong>Lambda trend chart</strong>
                <div className="placeholder-wireframe trend-wireframe" />
              </article>
            </div>
          </section>
        </div>
      </section>
    </main>
  );
}
