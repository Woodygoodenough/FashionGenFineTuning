import { TitleCard } from "../../components/TitleCard";
import { EmbeddingMapExplorer } from "../../components/EmbeddingMapExplorer";

export default function EmbeddingAnalysisPage() {
  return (
    <main className="page-shell">
      <section className="placeholder-page embedding-page">
        <div className="placeholder-left">
          <TitleCard
            eyebrow="Analysis"
            title="Embedding analysis"
            description="Interactive map over a shared 1k FashionGen sample. Compare zero-shot, retrieval-only, and joint embeddings under the same category-balanced point set."
          />
        </div>

        <div className="placeholder-right">
          <EmbeddingMapExplorer />
        </div>
      </section>
    </main>
  );
}
