"use client";

type QueryComposerProps = {
  query: string;
  topK: number;
  onQueryChange: (value: string) => void;
  onTopKChange: (value: number) => void;
};

const suggestedQueries = [
  "red hat with a soft wool texture",
  "structured black handbag with gold hardware",
  "cream dress for a summer evening",
];

export function QueryComposer({
  query,
  topK,
  onQueryChange,
  onTopKChange,
}: QueryComposerProps) {
  return (
    <section className="query-composer">
      <label className="input-label" htmlFor="query">
        Query
      </label>
      <textarea
        id="query"
        className="query-input"
        rows={4}
        value={query}
        onChange={(event) => onQueryChange(event.target.value)}
        placeholder="Describe the product you want to retrieve..."
      />

      <div className="composer-row">
        <div className="k-control">
          <div className="k-header">
            <label htmlFor="top-k">Top K</label>
            <span>{topK}</span>
          </div>
          <input
            id="top-k"
            type="range"
            min={1}
            max={10}
            step={1}
            value={topK}
            onChange={(event) => onTopKChange(Number(event.target.value))}
          />
          <div className="k-scale">
            <span>1</span>
            <span>10</span>
          </div>
        </div>

        <div className="suggestion-block">
          <span className="input-label">Suggested prompts</span>
          <div className="suggestion-list">
            {suggestedQueries.map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                className="suggestion-chip"
                onClick={() => onQueryChange(suggestion)}
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="composer-actions">
        <button type="button" className="primary-button">
          Preview Retrieval
        </button>
        <button
          type="button"
          className="ghost-button"
          onClick={() => {
            onQueryChange("");
            onTopKChange(5);
          }}
        >
          Reset
        </button>
      </div>
    </section>
  );
}
