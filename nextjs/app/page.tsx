"use client";

import { useEffect, useState } from "react";
import { TitleCard } from "../components/TitleCard";
import { QueryComposer } from "../components/QueryComposer";
import { ResultGallery } from "../components/ResultGallery";
import {
  loadCatalog,
  rankCatalog,
  searchCatalogApi,
  type DemoItem,
} from "../components/demo-catalog";

export default function Home() {
  const apiEnabled = Boolean(process.env.NEXT_PUBLIC_DEMO_API_BASE);
  const [query, setQuery] = useState("red hat with a soft wool texture");
  const [topK, setTopK] = useState(5);
  const [catalog, setCatalog] = useState<DemoItem[]>([]);
  const [results, setResults] = useState<DemoItem[]>([]);

  useEffect(() => {
    if (apiEnabled) {
      return;
    }
    loadCatalog().then(setCatalog).catch(() => setCatalog([]));
  }, [apiEnabled]);

  useEffect(() => {
    let cancelled = false;

    async function run() {
      try {
        const remote = apiEnabled ? await searchCatalogApi(query, topK) : null;
        if (!cancelled && remote) {
          setResults(remote);
          return;
        }
      } catch {}

      if (!cancelled) {
        setResults(rankCatalog(catalog, query, topK));
      }
    }

    run();
    return () => {
      cancelled = true;
    };
  }, [apiEnabled, catalog, query, topK]);

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
        <ResultGallery query={query} topK={topK} results={results} />
      </section>
    </main>
  );
}
