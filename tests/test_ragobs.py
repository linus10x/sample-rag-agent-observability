import unittest

from ragobs.agents import Orchestrator, grounded_sentences
from ragobs.corpus import DOCS
from ragobs.dashboard import render
from ragobs.evals import GOLD, retrieval_hit, summarize
from ragobs.retrieve import Index, bm25_scores, tokenize
from ragobs.trace import Tracer


class RetrieveTests(unittest.TestCase):
    def test_retention_query_hits_retention_memo(self):
        hits = Index(DOCS).search("how long are call recordings kept", k=3)
        self.assertEqual(hits[0]["id"], "doc-retention")
        self.assertGreater(hits[0]["score"], 0)

    def test_empty_query_returns_nothing(self):
        self.assertEqual(Index(DOCS).search("the a of"), [])

    def test_bm25_zero_when_no_overlap(self):
        scores = bm25_scores(["zebra"], [tokenize("call recordings kept")])
        self.assertEqual(scores, [0.0])


class AgentTests(unittest.TestCase):
    def test_gold_set_is_grounded_and_traced(self):
        tracer = Tracer()
        orch = Orchestrator(Index(DOCS), tracer)
        runs = [orch.run(q) for q in GOLD]
        stats = summarize(runs)
        self.assertEqual(stats["retrieval_hit_rate"], 1.0)
        self.assertEqual(stats["grounded_rate"], 1.0)
        parents = {s["span_id"]: s for s in tracer.finished()}
        for span in tracer.finished():
            if span["parent_span_id"]:
                self.assertEqual(parents[span["parent_span_id"]]["trace_id"], span["trace_id"])
        self.assertTrue(any(s["name"] == "critique" for s in tracer.finished()))
        self.assertGreater(runs[0]["cost_usd"], 0)
        self.assertIn("eval.latency_ms", [a for s in tracer.finished() if s["name"] == "orchestrate" for a in s["attributes"]])

    def test_ungrounded_sentence_is_dropped(self):
        kept = grounded_sentences(
            ["Call recordings are kept for 90 days", "The moon is made of cheese"],
            "Call recordings are kept for 90 days.",
        )
        self.assertEqual(kept, ["Call recordings are kept for 90 days"])

    def test_critic_marks_error_when_a_candidate_is_outside_the_context(self):
        tracer = Tracer()
        orch = Orchestrator(Index(DOCS), tracer)
        # Force a candidate the critic cannot ground by wrapping _draft.
        orch._draft = lambda query, hits: ["Call recordings are kept for 90 days", "Invented vendor sla"]
        result = orch.run("how long are call recordings kept")
        self.assertIn("90 days", result["answer"])
        self.assertNotIn("Invented", result["answer"])
        critique = [s for s in tracer.finished() if s["name"] == "critique"][-1]
        self.assertEqual(critique["status"], "error")
        self.assertTrue(any(s["name"] == "revise" for s in tracer.finished()))

    def test_dashboard_lists_orchestrate_cost(self):
        tracer = Tracer()
        Orchestrator(Index(DOCS), tracer).run("how long are call recordings kept")
        text = render(tracer.finished())
        self.assertIn("orchestrate", text)
        self.assertIn("budget_5usd | ok", text)


class HitHelperTests(unittest.TestCase):
    def test_retrieval_hit_helper(self):
        self.assertTrue(retrieval_hit([{"id": "doc-cost"}], "doc-cost"))
        self.assertFalse(retrieval_hit([{"id": "doc-hours"}], "doc-cost"))


if __name__ == "__main__":
    unittest.main()
