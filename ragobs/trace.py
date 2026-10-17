"""OpenTelemetry-style spans. Local only. No exporter account and no collector."""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager


def _hex(n: int) -> str:
    return uuid.uuid4().hex[:n]


class Tracer:
    def __init__(self) -> None:
        self.spans: list[dict] = []
        self._stack: list[dict] = []

    @contextmanager
    def span(self, name: str, **attributes: object):
        trace_id = self._stack[-1]["trace_id"] if self._stack else _hex(32)
        current = {
            "trace_id": trace_id,
            "span_id": _hex(16),
            "parent_span_id": self._stack[-1]["span_id"] if self._stack else None,
            "name": name,
            "start_time_unix_nano": time.time_ns(),
            "end_time_unix_nano": None,
            "attributes": {k: attributes[k] for k in attributes},
            "status": "ok",
        }
        self._stack.append(current)
        try:
            yield current
        except Exception:
            current["status"] = "error"
            raise
        finally:
            current["end_time_unix_nano"] = time.time_ns()
            self.spans.append(current)
            self._stack.pop()

    def set(self, span: dict, **attributes: object) -> None:
        span["attributes"].update(attributes)

    def finished(self) -> list[dict]:
        return [s for s in self.spans if s["end_time_unix_nano"] is not None]
