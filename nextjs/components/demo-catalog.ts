export type DemoItem = {
  id: string;
  title: string;
  category: string;
  caption: string;
  image: string;
  score?: number;
};

type DemoCatalogPayload = {
  items: DemoItem[];
};

type DemoSearchPayload = {
  query: string;
  items: DemoItem[];
};

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((token) => token.length > 1);
}

export async function loadCatalog(): Promise<DemoItem[]> {
  const response = await fetch("/demo/catalog.json");
  if (!response.ok) {
    throw new Error("Failed to load demo catalog");
  }
  const payload = (await response.json()) as DemoCatalogPayload;
  return payload.items;
}

export async function searchCatalogApi(
  query: string,
  topK: number,
): Promise<DemoItem[] | null> {
  const base = process.env.NEXT_PUBLIC_DEMO_API_BASE;
  if (!base) {
    return null;
  }

  const url = new URL("/api/demo/search", base);
  url.searchParams.set("query", query);
  url.searchParams.set("k", String(topK));

  const response = await fetch(url.toString());
  if (!response.ok) {
    throw new Error("Failed to fetch demo search results");
  }
  const payload = (await response.json()) as DemoSearchPayload;
  return payload.items;
}

export function rankCatalog(items: DemoItem[], query: string, topK: number): DemoItem[] {
  const queryTokens = tokenize(query);
  if (queryTokens.length === 0) {
    return items.slice(0, topK);
  }

  const scored = items.map((item) => {
    const haystack = `${item.title} ${item.category} ${item.caption}`.toLowerCase();
    let score = 0;
    for (const token of queryTokens) {
      if (item.category.toLowerCase().includes(token)) score += 4;
      if (item.title.toLowerCase().includes(token)) score += 6;
      if (haystack.includes(token)) score += 2;
    }
    if (haystack.includes(query.trim().toLowerCase())) score += 8;
    return { item, score };
  });

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK).map((row) => ({
    ...row.item,
    score: row.score,
  }));
}
