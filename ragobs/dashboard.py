"""Text dashboard over finished spans. Cost and latency only."""

from __future__ import annotations


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[idx]


def render(spans: list[dict], daily_alert_usd: float = 5.0) -> str:
    groups: dict[str, list[dict]] = {}
    for span in spans:
        groups.setdefault(span["name"], []).append(span)
    lines = ["name | calls | errors | p50_ms | tokens | est_usd"]
    total_cost = 0.0
    for name in sorted(groups):
        rows = groups[name]
        durations = [
            (row["end_time_unix_nano"] - row["start_time_unix_nano"]) / 1_000_000
            for row in rows
            if row.get("end_time_unix_nano")
        ]
        tokens = sum(int(row["attributes"].get("eval.tokens", 0)) for row in rows)
        cost = sum(float(row["attributes"].get("eval.cost_usd", 0.0)) for row in rows)
        total_cost += cost
        errors = sum(1 for row in rows if row["status"] != "ok")
        lines.append(
            f"{name} | {len(rows)} | {errors} | {_percentile(durations, 0.5):.3f} | {tokens} | {cost:.8f}"
        )
    alert = "ALERT" if total_cost >= daily_alert_usd else "ok"
    lines.append(f"total_est_usd | {total_cost:.8f} | budget_5usd | {alert}")
    return "\n".join(lines) + "\n"
