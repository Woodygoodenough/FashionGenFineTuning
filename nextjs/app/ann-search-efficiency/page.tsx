"use client";

import { useMemo, useState } from "react";
import { TitleCard } from "../../components/TitleCard";

type BenchmarkResult = {
  query_count: number;
  k: number;
  ann_total_ms: number;
  ann_avg_ms: number;
  exact_total_ms: number;
  exact_avg_ms: number;
  speedup: number | null;
};

const RECORDED_RUNS: BenchmarkResult[] = [
  {
    query_count: 100,
    k: 10,
    ann_total_ms: 7538.044934000936,
    ann_avg_ms: 75.38044934000936,
    exact_total_ms: 8459.913558999688,
    exact_avg_ms: 84.59913558999688,
    speedup: 1.1222954536713614,
  },
  {
    query_count: 100,
    k: 10,
    ann_total_ms: 7524.352619999263,
    ann_avg_ms: 75.24352619999263,
    exact_total_ms: 8477.06386199934,
    exact_avg_ms: 84.7706386199934,
    speedup: 1.1266170380515967,
  },
  {
    query_count: 100,
    k: 10,
    ann_total_ms: 7573.58387900058,
    ann_avg_ms: 75.7358387900058,
    exact_total_ms: 8577.041796999765,
    exact_avg_ms: 85.77041796999765,
    speedup: 1.1324944615430341,
  },
];

const MOCK_RESULT: BenchmarkResult = {
  query_count: 100,
  k: 10,
  ann_total_ms: 7545.327144333593,
  ann_avg_ms: 75.45327144333594,
  exact_total_ms: 8504.673072666264,
  exact_avg_ms: 85.04673072666264,
  speedup: 1.127135651088664,
};

const PIPELINE_STEPS = [
  {
    label: "Text Encoder",
    summary: "Query tokens become a CLIP text embedding.",
    detail:
      "The service computes a text embedding in real time for each query. With normalized embedding t, the retrieval stage compares it against normalized image vectors. Core score: s(q, i) = t^T v_i.",
  },
  {
    label: "Normalization",
    summary: "Cosine similarity becomes an inner-product lookup.",
    detail:
      "Both text and image embeddings are L2-normalized. That makes cosine similarity equal to the inner product, so FAISS can search with MaxIP directly: cos(t, v_i) = (t^T v_i) / (||t|| ||v_i||) = t^T v_i.",
  },
  {
    label: "HNSW Entry",
    summary: "Search starts from graph entry points.",
    detail:
      "HNSW stores image embeddings as a layered proximity graph. Query search starts from upper-layer entry nodes, then descends toward denser layers to locate promising neighborhoods quickly.",
  },
  {
    label: "Graph Traversal",
    summary: "ANN visits only promising neighbors.",
    detail:
      "Instead of scanning all N images, HNSW maintains a candidate frontier and expands nearest graph neighbors. This approximates arg max_i t^T v_i while keeping traversal sublinear in practice.",
  },
  {
    label: "Candidate Heap",
    summary: "Top candidates are kept in a bounded heap.",
    detail:
      "At the bottom layer, the search keeps a bounded candidate set of the strongest matches found so far. efSearch controls how aggressively the graph is explored before final ranking.",
  },
  {
    label: "Top-k Results",
    summary: "The final heap becomes the returned gallery.",
    detail:
      "The service returns the k highest-similarity items from the ANN candidate set. Exact search computes all scores {t^T v_i}_{i=1}^N, while ANN returns an approximation much faster.",
  },
] as const;

async function fetchBenchmark(): Promise<BenchmarkResult> {
  const base = process.env.NEXT_PUBLIC_DEMO_API_BASE;
  if (!base) {
    throw new Error("Benchmark API is not configured");
  }

  const url = new URL("/api/demo/ann-benchmark", base);
  url.searchParams.set("query_count", "100");
  url.searchParams.set("k", "10");

  const response = await fetch(url.toString(), { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Failed to run ANN benchmark");
  }
  return (await response.json()) as BenchmarkResult;
}

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

export default function AnnSearchEfficiencyPage() {
  const [mode, setMode] = useState<"mock" | "real">("mock");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<BenchmarkResult | null>(MOCK_RESULT);
  const [runCount, setRunCount] = useState(0);

  const totalBenchmarkMs = useMemo(() => {
    if (!result) return 0;
    return result.ann_total_ms + result.exact_total_ms;
  }, [result]);

  async function runBenchmark() {
    setRunning(true);
    try {
      let next: BenchmarkResult;
      if (mode === "mock") {
        await sleep(MOCK_RESULT.ann_total_ms + MOCK_RESULT.exact_total_ms);
        next = MOCK_RESULT;
      } else {
        next = await fetchBenchmark();
      }
      setResult(next);
      setRunCount((value) => value + 1);
    } finally {
      setRunning(false);
    }
  }

  return (
    <main className="page-shell">
      <section className="ann-page">
        <div className="ann-left">
          <TitleCard
            eyebrow="Simulation"
            title="ANN search efficiency"
            description="Exact search versus HNSW ANN over 100 text queries."
          />

          <section className="ann-control-card">
            <div className="ann-control-row">
              <div>
                <span className="stat-label">Benchmark scope</span>
                <strong>100 queries · top 10 results</strong>
              </div>
              <div className="ann-mode-switch">
                <button
                  type="button"
                  className={mode === "mock" ? "primary-button" : "ghost-button"}
                  onClick={() => setMode("mock")}
                >
                  Mock
                </button>
                <button
                  type="button"
                  className={mode === "real" ? "primary-button" : "ghost-button"}
                  onClick={() => setMode("real")}
                >
                  Real
                </button>
              </div>
            </div>

            <div className="ann-control-meta">
              <span className="stat-label">Execution mode</span>
              <strong>{mode === "mock" ? "Simulated from recorded runs" : "Live API benchmark"}</strong>
            </div>

            <div className="ann-control-meta">
              <span className="stat-label">Recorded baseline</span>
              <strong>{`${RECORDED_RUNS.length} live runs averaged`}</strong>
            </div>

            <button
              type="button"
              className="primary-button ann-run-button"
              onClick={runBenchmark}
              disabled={running}
            >
              {running ? "Running 100-query benchmark..." : "Run 100-query benchmark"}
            </button>

            <div className="ann-control-meta">
              <span className="stat-label">Simulated total wait</span>
              <strong>{mode === "mock" ? `${(MOCK_RESULT.ann_total_ms + MOCK_RESULT.exact_total_ms).toFixed(0)} ms` : "Live timing"}</strong>
            </div>
          </section>
        </div>

        <div className="ann-right">
          <section className="ann-results-card">
            <div className="gallery-toolbar">
              <div className="gallery-stat">
                <span className="stat-label">Latest run</span>
                <strong>{runCount === 0 ? "Not run in this session" : `Run #${runCount}`}</strong>
              </div>
              <div className="gallery-stat">
                <span className="stat-label">Total time</span>
                <strong>{result ? `${totalBenchmarkMs.toFixed(0)} ms` : "N/A"}</strong>
              </div>
            </div>

            <div className="ann-metric-grid">
              <article className="ann-metric-tile">
                <span className="stat-label">Exact search</span>
                <strong>{result ? `${result.exact_total_ms.toFixed(0)} ms` : "N/A"}</strong>
                <span>Average {result ? `${result.exact_avg_ms.toFixed(2)} ms/query` : "N/A"}</span>
              </article>
              <article className="ann-metric-tile">
                <span className="stat-label">ANN search</span>
                <strong>{result ? `${result.ann_total_ms.toFixed(0)} ms` : "N/A"}</strong>
                <span>Average {result ? `${result.ann_avg_ms.toFixed(2)} ms/query` : "N/A"}</span>
              </article>
              <article className="ann-metric-tile">
                <span className="stat-label">Observed speedup</span>
                <strong>{result && result.speedup ? `${result.speedup.toFixed(2)}x` : "N/A"}</strong>
                <span>ANN over exact for the same query workload</span>
              </article>
            </div>
          </section>

          <section className="ann-architecture-card">
            <div className="ann-architecture-head">
              <span className="stat-label">ANN pipeline</span>
              <strong>Hover each stage for the retrieval mechanics</strong>
            </div>
            <div className="ann-architecture-grid">
              {PIPELINE_STEPS.map((step, index) => (
                <div className="ann-step" key={step.label}>
                  <div className="ann-step-node">
                    <span className="ann-step-index">{index + 1}</span>
                    <strong>{step.label}</strong>
                    <p>{step.summary}</p>
                  </div>
                  <div className="ann-tooltip" role="tooltip">
                    <strong>{step.label}</strong>
                    <p>{step.detail}</p>
                  </div>
                </div>
              ))}
            </div>
          </section>
        </div>
      </section>
    </main>
  );
}
