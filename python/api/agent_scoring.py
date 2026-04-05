# Agent Scoring Metrics API
# Exposes task performance metrics in a format compatible with Zorin's TaskRecord schema.
# Used by ops.sanmarcsoft.com Grafana dashboard for unified agent scoring.
#
# Implements MHH-EI specification: "Language Enabled Emotional Intelligence
# and Theory of Mind Algorithms" by Sean Webb (AGPL-3.0)
# Webb Equation of Emotion: EP ∆ P = ER

import os
import json
import time
from datetime import datetime, timezone

from python.helpers.api import ApiHandler, Input, Output
from flask import Request, Response

# In-memory task scoring history (persisted across requests, lost on restart)
# TODO: persist to ChromaDB for cross-restart durability
_task_history: list[dict] = []
_MAX_HISTORY = 50
_task_counter = 0


class AgentScoring(ApiHandler):
    """Expose task scoring metrics for the Q Branch autonomous agent.

    GET /agent_scoring — returns performance report (Prometheus-compatible JSON)
    POST /agent_scoring — record a new task score
    """

    @classmethod
    def requires_auth(cls) -> bool:
        return False  # metrics endpoint needs to be scrapeable by Prometheus exporter

    @classmethod
    def requires_csrf(cls) -> bool:
        return False

    @classmethod
    def get_methods(cls) -> list[str]:
        return ["GET", "POST"]

    async def process(self, input: Input, request: Request) -> Output:
        if request.method == "GET":
            # For GET, merge query params into input (ApiHandler only parses JSON body)
            get_input = dict(request.args)
            get_input.update(input)
            return self._performance_report(get_input)
        else:
            return self._record_task(input)

    def _performance_report(self, input: Input) -> Output:
        """Return aggregated performance metrics compatible with Zorin's TaskRecord schema."""
        fmt = input.get("format", "json")

        if len(_task_history) == 0:
            report = {
                "agent": "q",
                "model": os.environ.get("A0_SET_chat_model_name", "unknown"),
                "total_tasks": 0,
                "avg_latency_ms": 0,
                "avg_score": 0,
                "avg_tokens_per_second": 0,
                "success_rate": 0,
                "last_task": None,
                "history": [],
            }
        else:
            successes = [t for t in _task_history if not t.get("error")]
            avg_latency = sum(t["latency_ms"] for t in _task_history) / len(_task_history)
            avg_score = sum(t["score"] for t in _task_history) / len(_task_history)
            avg_tps = (
                sum(t["tokens_per_second"] for t in successes) / len(successes)
                if successes
                else 0
            )

            report = {
                "agent": "q",
                "model": os.environ.get("A0_SET_chat_model_name", "unknown"),
                "total_tasks": len(_task_history),
                "avg_latency_ms": round(avg_latency),
                "avg_score": round(avg_score, 1),
                "avg_tokens_per_second": round(avg_tps, 1),
                "success_rate": round(len(successes) / len(_task_history) * 100, 1),
                "last_task": _task_history[-1] if _task_history else None,
                "history": _task_history[-10:],
            }

        if fmt == "prometheus":
            return self._prometheus_format(report)

        return report

    def _record_task(self, input: Input) -> Output:
        """Record a completed task with its score and metrics."""
        global _task_counter
        _task_counter += 1

        record = {
            "id": f"QT-{_task_counter}-{int(time.time() * 1000)}",
            "task": input.get("task", ""),
            "response": input.get("response", "")[:500],  # truncate for storage
            "score": min(10, max(1, int(input.get("score", 5)))),
            "latency_ms": int(input.get("latency_ms", 0)),
            "tokens": int(input.get("tokens", 0)),
            "tokens_per_second": float(input.get("tokens_per_second", 0)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "spoke": bool(input.get("spoke", False)),
            "webb_equation": input.get("webb_equation", None),
            "error": input.get("error", False),
        }

        _task_history.append(record)
        if len(_task_history) > _MAX_HISTORY:
            _task_history.pop(0)

        return {"status": "recorded", "record": record}

    def _prometheus_format(self, report: dict) -> Response:
        """Render metrics in Prometheus text exposition format."""
        lines = [
            "# HELP agent_task_total Total tasks executed by this agent",
            "# TYPE agent_task_total counter",
            f'agent_task_total{{agent="q"}} {report["total_tasks"]}',
            "",
            "# HELP agent_task_avg_score Average self-scored task quality (1-10)",
            "# TYPE agent_task_avg_score gauge",
            f'agent_task_avg_score{{agent="q"}} {report["avg_score"]}',
            "",
            "# HELP agent_task_avg_latency_ms Average task execution latency in milliseconds",
            "# TYPE agent_task_avg_latency_ms gauge",
            f'agent_task_avg_latency_ms{{agent="q"}} {report["avg_latency_ms"]}',
            "",
            "# HELP agent_task_avg_tokens_per_second Average token generation throughput",
            "# TYPE agent_task_avg_tokens_per_second gauge",
            f'agent_task_avg_tokens_per_second{{agent="q"}} {report["avg_tokens_per_second"]}',
            "",
            "# HELP agent_task_success_rate Percentage of non-error tasks",
            "# TYPE agent_task_success_rate gauge",
            f'agent_task_success_rate{{agent="q"}} {report["success_rate"]}',
            "",
        ]
        return Response(
            "\n".join(lines) + "\n",
            content_type="text/plain; version=0.0.4; charset=utf-8",
        )
