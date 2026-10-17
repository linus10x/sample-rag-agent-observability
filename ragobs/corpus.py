"""Synthetic internal memos. No client, no employer, no real customer."""

DOCS = [
    {
        "id": "doc-retention",
        "text": (
            "Call recordings are kept for 90 days. Chat transcripts are kept for 30 days. "
            "After the keep window the sample desk deletes the file."
        ),
    },
    {
        "id": "doc-rag",
        "text": (
            "Answers must cite a retrieved memo. The index returns the top 4 chunks. "
            "A sentence with no overlap against those chunks is dropped."
        ),
    },
    {
        "id": "doc-agents",
        "text": (
            "Three agents run in order. The retriever searches the index. "
            "The drafter copies the best matching sentences. The critic rejects an uncited claim."
        ),
    },
    {
        "id": "doc-cost",
        "text": (
            "The sample price is 2 dollars per million tokens. "
            "A day that crosses 5 dollars raises a cost alert on the dashboard."
        ),
    },
    {
        "id": "doc-hours",
        "text": (
            "The sample desk is open from 8 to 6 Central Time on weekdays. "
            "Weekend notes wait until the next weekday."
        ),
    },
    {
        "id": "doc-escalation",
        "text": (
            "Escalate when the critic flags an ungrounded sentence twice in one trace. "
            "The second flag ends the run with status error."
        ),
    },
]
