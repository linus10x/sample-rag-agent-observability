"""Run the sample questions and print the dashboard. Stdlib only."""

from ragobs.agents import Orchestrator
from ragobs.corpus import DOCS
from ragobs.dashboard import render
from ragobs.evals import GOLD, summarize
from ragobs.retrieve import Index
from ragobs.trace import Tracer


def main() -> int:
    tracer = Tracer()
    orch = Orchestrator(Index(DOCS), tracer, k=3)
    runs = [orch.run(query) for query in GOLD]
    print(render(tracer.finished()), end="")
    stats = summarize(runs)
    print(
        "hit_rate {retrieval_hit_rate:.2f} grounded_rate {grounded_rate:.2f} cost {total_cost_usd:.8f}".format(
            **stats
        )
    )
    if stats["retrieval_hit_rate"] < 1.0 or stats["grounded_rate"] < 1.0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
