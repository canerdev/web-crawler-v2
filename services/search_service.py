import re

from core.index_store import IndexEntry, IndexStore

_WORD_RE = re.compile(r"\b[a-zA-Z]{2,}\b")


class SearchService:
    def __init__(self, index_store: IndexStore):
        self._store = index_store

    def search(self, query: str, limit: int = 50, sort_by: str = "relevance") -> list[dict]:
        words = [w.lower() for w in _WORD_RE.findall(query)]
        if not words:
            return []

        entries = self._store.search(words)
        scored = _score_entries(entries, words)

        best: dict[str, tuple[float, dict]] = {}
        for score, entry in scored:
            if entry.url not in best or score > best[entry.url][0]:
                best[entry.url] = (
                    score,
                    {
                        "relevant_url": entry.url,
                        "origin_url": entry.origin,
                        "depth": entry.depth,
                        "frequency": entry.frequency,
                        "relevance_score": round(score, 2),
                    },
                )

        if sort_by == "relevance":
            ranked = sorted(best.values(), key=lambda x: x[0], reverse=True)
        elif sort_by == "depth":
            ranked = sorted(best.values(), key=lambda x: x[1]["depth"])
        else:
            ranked = sorted(best.values(), key=lambda x: x[0], reverse=True)

        return [item[1] for item in ranked[:limit]]


def _score_entries(
    entries: list[IndexEntry], query_words: list[str]
) -> list[tuple[float, IndexEntry]]:
    results: list[tuple[float, IndexEntry]] = []
    query_set = set(query_words)

    for entry in entries:
        score = entry.frequency * 10.0

        if entry.word in query_set:
            score += 1000.0

        score -= entry.depth * 5.0

        results.append((score, entry))

    return results
