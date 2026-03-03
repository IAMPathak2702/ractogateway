"""Pure-Python BM25 index and decision-tree inverted index.

No external dependencies required — everything is implemented with the
Python standard library.

Two components work together for two-stage retrieval:

1. :class:`_DecisionIndex` — an inverted keyword index that maps content
   terms to page entry IDs.  Given a tokenised query it returns the *union*
   of candidate entry IDs in O(|query terms|) time.  This is the
   "decision tree" routing layer.

2. :class:`BM25Index` — Okapi BM25 (k1=1.5, b=0.75) that scores the
   candidates returned by the decision index.  Only candidates are scored,
   so the full corpus is never re-ranked on every query.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

# ---------------------------------------------------------------------------
# Stopwords (small, hardcoded — no NLTK required)
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "shall", "should", "may", "might", "can", "could", "not",
        "no", "nor", "so", "yet", "both", "either", "neither", "that", "this",
        "these", "those", "it", "its", "as", "up", "out", "about", "into",
        "over", "after", "their", "there", "here", "which", "who", "what",
        "how", "when", "where", "why", "all", "each", "every", "more", "most",
        "also", "just", "than", "then", "my", "your", "his", "her", "our",
        "we", "you", "he", "she", "they", "i", "me", "us", "him", "them",
    }
)


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    """Lowercase, split on non-word characters, drop short / stopword tokens."""
    tokens = re.split(r"\W+", text.lower())
    return [t for t in tokens if len(t) >= 2 and t not in _STOPWORDS]


def extract_keywords(text: str, top_n: int = 20) -> list[str]:
    """Return the top-*n* most frequent content tokens from *text*."""
    counts = Counter(_tokenise(text))
    return [tok for tok, _ in counts.most_common(top_n)]


# ---------------------------------------------------------------------------
# Decision index (stage 1: routing / candidate selection)
# ---------------------------------------------------------------------------

class _DecisionIndex:
    """Inverted keyword index that routes query terms to candidate page IDs.

    This is the "decision tree" layer: each term is a branch that points to
    the set of pages containing that term.  A query traverses all matching
    branches and returns their union as candidates for BM25 scoring.
    """

    def __init__(self) -> None:
        # term → {entry_id, …}
        self._index: dict[str, set[str]] = {}

    def add(self, entry_id: str, terms: Sequence[str]) -> None:
        """Register *entry_id* under each of its content *terms*."""
        for term in terms:
            self._index.setdefault(term, set()).add(entry_id)

    def remove(self, entry_id: str) -> None:
        """Remove *entry_id* from every term bucket (used by ``clear()``)."""
        for bucket in self._index.values():
            bucket.discard(entry_id)
        # Prune empty buckets to keep memory tidy
        self._index = {k: v for k, v in self._index.items() if v}

    def candidates(self, query_terms: Sequence[str]) -> set[str]:
        """Return all entry IDs that contain at least one query term."""
        result: set[str] = set()
        for term in query_terms:
            result |= self._index.get(term, set())
        return result

    def clear(self) -> None:
        self._index.clear()


# ---------------------------------------------------------------------------
# BM25 index (stage 2: scoring)
# ---------------------------------------------------------------------------

class BM25Index:
    """Okapi BM25 scorer over a corpus of :class:`PageEntry` texts.

    Parameters
    ----------
    k1:
        Term-frequency saturation parameter (default 1.5).
    b:
        Length normalisation parameter (default 0.75).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        # entry_id → token counts
        self._corpus: dict[str, Counter[str]] = {}
        # entry_id → document length
        self._lengths: dict[str, int] = {}
        # term → document-frequency count
        self._df: Counter[str] = Counter()
        self._avg_dl: float = 0.0

    # ------------------------------------------------------------------
    # Build / update
    # ------------------------------------------------------------------

    def add(self, entry_id: str, text: str) -> None:
        """Tokenise *text* and add the entry to the index."""
        tokens = _tokenise(text)
        tf: Counter[str] = Counter(tokens)
        self._corpus[entry_id] = tf
        self._lengths[entry_id] = len(tokens)
        for term in tf:
            self._df[term] += 1
        self._avg_dl = sum(self._lengths.values()) / len(self._lengths)

    def remove(self, entry_id: str) -> None:
        """Remove *entry_id* from the index."""
        if entry_id not in self._corpus:
            return
        for term in self._corpus[entry_id]:
            self._df[term] -= 1
            if self._df[term] <= 0:
                del self._df[term]
        del self._corpus[entry_id]
        del self._lengths[entry_id]
        if self._lengths:
            self._avg_dl = sum(self._lengths.values()) / len(self._lengths)
        else:
            self._avg_dl = 0.0

    def clear(self) -> None:
        self._corpus.clear()
        self._lengths.clear()
        self._df.clear()
        self._avg_dl = 0.0

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _idf(self, term: str) -> float:
        n = len(self._corpus)
        df = self._df.get(term, 0)
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def score(
        self,
        query: str,
        candidate_ids: set[str] | None = None,
    ) -> list[tuple[str, float, list[str]]]:
        """Score candidates against *query* and return ranked results.

        Parameters
        ----------
        query:
            Raw query string.
        candidate_ids:
            Subset of entry IDs to score.  When ``None`` the entire corpus
            is scored (full-scan fallback).

        Returns
        -------
        list of (entry_id, bm25_score, matched_terms)
            Sorted descending by score, ties broken by entry_id for stability.
        """
        query_terms = _tokenise(query)
        targets = candidate_ids if candidate_ids is not None else set(self._corpus)
        if not targets or not query_terms:
            return []

        results: list[tuple[str, float, list[str]]] = []
        k1 = self._k1
        b = self._b
        avg_dl = self._avg_dl if self._avg_dl > 0 else 1.0

        for eid in targets:
            if eid not in self._corpus:
                continue
            tf_map = self._corpus[eid]
            dl = self._lengths[eid]
            score = 0.0
            matched: list[str] = []
            for term in query_terms:
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                idf = self._idf(term)
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
                score += idf * tf_norm
                if term not in matched:
                    matched.append(term)
            if score > 0:
                results.append((eid, score, matched))

        results.sort(key=lambda x: (-x[1], x[0]))
        return results

    @property
    def entry_count(self) -> int:
        return len(self._corpus)
