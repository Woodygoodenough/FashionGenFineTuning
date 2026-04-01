"use client";

import { useEffect, useMemo, useState } from "react";
import { TitleCard } from "../components/TitleCard";
import { QueryComposer } from "../components/QueryComposer";
import { ResultGallery } from "../components/ResultGallery";
import { loadCatalog, rankCatalog, type DemoItem } from "../components/demo-catalog";

export default function Home() {
  const [query, setQuery] = useState("red hat with a soft wool texture");
  const [topK, setTopK] = useState(5);
  const [catalog, setCatalog] = useState<DemoItem[]>([]);

  useEffect(() => {
    loadCatalog()
      .then(setCatalog)
      .catch(() => setCatalog([]));
  }, []);

  const ranked = useMemo(() => rankCatalog(catalog, query, topK), [catalog, query, topK]);

  return (
    <main className="page-shell">
      <section className="content-grid">
        <div className="left-rail">
          <TitleCard
            eyebrow="Demo"
            title="Production demo"
            description="Text-to-image retrieval interface preview."
          />
          <QueryComposer
            query={query}
            topK={topK}
            onQueryChange={setQuery}
            onTopKChange={setTopK}
          />
        </div>
        <ResultGallery query={query} topK={topK} results={ranked} />
      </section>
    </main>
  );
}
