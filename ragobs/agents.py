"""Retriever, drafter and critic. Extractive only. No model API."""

from __future__ import annotations

from ragobs.retrieve import Index, tokenize

# Sample price table. Not a vendor invoice. $2 per 1,000,000 tokens.
USD_PER_TOKEN = 2.0 / 1_000_000


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def sentences(text: str) -> list[str]:
    parts = []
    for chunk in text.replace("?", ".").replace("!", ".").split("."):
        sentence = chunk.strip()
        if sentence:
            parts.append(sentence)
    return parts


def grounded_sentences(candidates: list[str], context: str) -> list[str]:
    ctx = set(tokenize(context))
    kept = []
    for sentence in candidates:
        words = tokenize(sentence)
        if words and all(word in ctx for word in words):
            kept.append(sentence)
    return kept


class Orchestrator:
    def __init__(self, index: Index, tracer, k: int = 3) -> None:
        self.index = index
        self.tracer = tracer
        self.k = k

    def run(self, query: str) -> dict:
        with self.tracer.span("orchestrate", query=query) as root:
            with self.tracer.span("retrieve") as span:
                hits = self.index.search(query, k=self.k)
                self.tracer.set(span, retrieved_ids=[h["id"] for h in hits], hit_count=len(hits))
            context = " ".join(h["text"] for h in hits)
            with self.tracer.span("draft") as span:
                ranked = self._draft(query, hits)
                self.tracer.set(span, candidate_count=len(ranked))
            with self.tracer.span("critique") as span:
                kept = grounded_sentences(ranked, context)
                dropped = len(ranked) - len(kept)
                grounded = bool(ranked) and dropped == 0
                self.tracer.set(span, grounded=grounded, kept=len(kept), dropped=dropped)
                if dropped:
                    span["status"] = "error"
            if dropped:
                with self.tracer.span("revise") as span:
                    self.tracer.set(span, kept=len(kept), dropped=dropped)
            answer = ". ".join(kept) + ("." if kept else "")
            tokens = estimate_tokens(query) + estimate_tokens(context) + estimate_tokens(answer)
            cost = round(tokens * USD_PER_TOKEN, 8)
            self.tracer.set(
                root,
                **{
                    "eval.grounded": bool(kept),
                    "eval.tokens": tokens,
                    "eval.cost_usd": cost,
                    "gen_ai.operation.name": "rag_orchestrate",
                },
            )
        latency_ms = (root["end_time_unix_nano"] - root["start_time_unix_nano"]) / 1_000_000
        root["attributes"]["eval.latency_ms"] = round(latency_ms, 3)
        return {
            "query": query,
            "answer": answer,
            "hits": [{"id": h["id"], "score": h["score"]} for h in hits],
            "grounded": bool(kept),
            "tokens": tokens,
            "cost_usd": cost,
            "latency_ms": round(latency_ms, 3),
            "trace_id": root["trace_id"],
        }

    def _draft(self, query: str, hits: list[dict]) -> list[str]:
        q = set(tokenize(query))
        scored = []
        for hit in hits:
            for sentence in sentences(hit["text"]):
                overlap = len(q & set(tokenize(sentence)))
                if overlap:
                    scored.append((overlap, hit["id"], sentence))
        scored.sort(key=lambda row: (-row[0], row[1], row[2]))
        seen = set()
        picked = []
        for _overlap, _doc_id, sentence in scored:
            if sentence in seen:
                continue
            seen.add(sentence)
            picked.append(sentence)
            if len(picked) == 2:
                break
        return picked
