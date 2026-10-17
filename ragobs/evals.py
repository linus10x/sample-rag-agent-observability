"""Eval metrics computed from runs. Labels are part of this sample, not a client set."""

from __future__ import annotations

GOLD = {
    "how long are call recordings kept": "doc-retention",
    "who rejects an uncited claim": "doc-agents",
    "what price does the sample use": "doc-cost",
}


def retrieval_hit(hits: list[dict], expected_id: str) -> bool:
    return any(hit["id"] == expected_id for hit in hits)


def summarize(runs: list[dict], gold: dict[str, str] = GOLD) -> dict:
    labeled = [run for run in runs if run["query"] in gold]
    hits = sum(1 for run in labeled if retrieval_hit(run["hits"], gold[run["query"]]))
    grounded = sum(1 for run in runs if run["grounded"])
    latencies = sorted(run["latency_ms"] for run in runs)
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    return {
        "runs": len(runs),
        "labeled": len(labeled),
        "retrieval_hit_rate": (hits / len(labeled)) if labeled else 0.0,
        "grounded_rate": (grounded / len(runs)) if runs else 0.0,
        "total_cost_usd": round(sum(run["cost_usd"] for run in runs), 8),
        "p50_latency_ms": p50,
    }
