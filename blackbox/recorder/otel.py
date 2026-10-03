"""OpenTelemetry spans for recorded runs, using OpenInference attribute conventions.

Export is enabled when OTEL_EXPORTER_OTLP_ENDPOINT is set (OTLP/HTTP, e.g. Phoenix
or Jaeger) or BLACKBOX_OTEL_CONSOLE=true. ``spans_for_run`` always works offline.
"""
from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache

from ..config import settings

SPAN_KIND = {"planner": "LLM", "router": "LLM", "reasoner": "LLM", "final": "LLM",
             "tool": "TOOL", "retriever": "RETRIEVER", "memory": "CHAIN"}


def _text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)[:8000]


def span_attributes(run: dict, step: dict) -> dict:
    kind = SPAN_KIND.get(step.get("node_type"), "CHAIN")
    attrs = {
        "openinference.span.kind": kind, "session.id": run["run_id"],
        "input.value": _text(step.get("input")), "input.mime_type": "application/json",
        "output.value": _text(step.get("output")), "output.mime_type": "application/json",
        "blackbox.step_id": step["step_id"], "blackbox.node": step["node_name"],
        "blackbox.checkpoint_id": step.get("checkpoint_id", ""), "blackbox.cache_key": step.get("cache_key", ""),
        "blackbox.action": step.get("action", ""), "blackbox.parent_steps": json.dumps(step.get("parent_step_ids", [])),
    }
    if kind == "LLM":
        attrs.update({"llm.model_name": step.get("model") or "", "llm.provider": step.get("provider") or "",
                      "llm.token_count.prompt": step.get("tokens_in", 0),
                      "llm.token_count.completion": step.get("tokens_out", 0),
                      "llm.token_count.total": step.get("tokens_in", 0) + step.get("tokens_out", 0),
                      "llm.invocation_parameters": json.dumps({"temperature": step.get("temperature"), "seed": step.get("seed")})})
        if step.get("prompt"):
            attrs["llm.input_messages.0.message.content"] = step["prompt"][:8000]
    if kind == "TOOL":
        attrs["tool.name"] = step["node_name"]
    if kind == "RETRIEVER":
        for i, doc_id in enumerate(step.get("retrieved_doc_ids") or []):
            attrs[f"retrieval.documents.{i}.document.id"] = str(doc_id)
    if step.get("tool_error"):
        attrs["exception.message"] = _text((step.get("output") or {}).get("error", "error"))
    return attrs


def spans_for_run(run: dict) -> list[dict]:
    """Plain-dict spans (one AGENT root + one span per step) for inspection and tests."""
    root = {"name": f"agent.{run['task_family']}", "attributes": {
        "openinference.span.kind": "AGENT", "session.id": run["run_id"], "input.value": run.get("prompt", ""),
        "output.value": _text(run.get("final_answer")), "blackbox.status": run.get("status", "")}}
    return [root] + [{"name": step["node_name"], "attributes": span_attributes(run, step)} for step in run.get("steps", [])]


@lru_cache(maxsize=1)
def _tracer():
    cfg = settings()
    if not cfg.otel_endpoint and not cfg.otel_console:
        return None
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    provider = TracerProvider(resource=Resource.create({"service.name": "blackbox-agent"}))
    if cfg.otel_endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        endpoint = cfg.otel_endpoint.rstrip("/")
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
            endpoint=endpoint if endpoint.endswith("/v1/traces") else endpoint + "/v1/traces")))
    if cfg.otel_console:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    return provider.get_tracer("blackbox.recorder")


def export_run(run: dict) -> bool:
    tracer = _tracer()
    if tracer is None:
        return False
    from opentelemetry.trace import Status, StatusCode
    start = int(datetime.fromisoformat(run["created_at"]).timestamp() * 1e9)
    cursor = start
    root_attrs = spans_for_run(run)[0]["attributes"]
    root = tracer.start_span(f"agent.{run['task_family']}", start_time=start, attributes=root_attrs)
    from opentelemetry import trace
    with trace.use_span(root, end_on_exit=False):
        for step in run["steps"]:
            duration = int(max(step.get("latency_ms", 0), 0.001) * 1e6)
            span = tracer.start_span(step["node_name"], start_time=cursor, attributes=span_attributes(run, step))
            if step.get("tool_error"):
                span.set_status(Status(StatusCode.ERROR))
            cursor += duration
            span.end(end_time=cursor)
    root.set_status(Status(StatusCode.OK if run.get("success") else StatusCode.ERROR))
    root.end(end_time=max(cursor, start + int(run.get("total_time", 0) * 1e6)))
    return True
