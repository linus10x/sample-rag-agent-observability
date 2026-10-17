"""Lexical BM25 retrieval over a synthetic corpus. No embedding API."""

from __future__ import annotations

import math
import re

_STOP = frozenset(
    "a an the of to and or for in on with is are be this that it as at by from".split()
)
_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1]


def bm25_scores(
    query_tokens: list[str],
    documents: list[list[str]],
    k1: float = 1.2,
    b: float = 0.75,
) -> list[float]:
    if not documents:
        return []
    n_docs = len(documents)
    avgdl = sum(len(doc) for doc in documents) / n_docs
    df: dict[str, int] = {}
    for doc in documents:
        for token in set(doc):
            df[token] = df.get(token, 0) + 1
    scores: list[float] = []
    for doc in documents:
        tf: dict[str, int] = {}
        for token in doc:
            tf[token] = tf.get(token, 0) + 1
        score = 0.0
        for token in query_tokens:
            freq = tf.get(token)
            if not freq:
                continue
            n_q = df.get(token, 0)
            idf = math.log(1.0 + (n_docs - n_q + 0.5) / (n_q + 0.5))
            denom = freq + k1 * (1.0 - b + b * len(doc) / avgdl)
            score += idf * (freq * (k1 + 1.0)) / denom
        scores.append(score)
    return scores


class Index:
    def __init__(self, docs: list[dict]) -> None:
        self.docs = list(docs)
        self._tokens = [tokenize(d["text"]) for d in self.docs]

    def search(self, query: str, k: int = 3) -> list[dict]:
        q = tokenize(query)
        scores = bm25_scores(q, self._tokens)
        ranked = sorted(range(len(self.docs)), key=lambda i: (-scores[i], self.docs[i]["id"]))
        hits = []
        for i in ranked[:k]:
            if scores[i] <= 0:
                continue
            hits.append({**self.docs[i], "score": round(scores[i], 6)})
        return hits
