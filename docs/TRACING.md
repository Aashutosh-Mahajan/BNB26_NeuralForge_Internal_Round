# Viewing Black Box runs in Phoenix or Jaeger (OpenTelemetry)

Every recorded run can be exported as OpenTelemetry spans using OpenInference conventions:
one `AGENT` root span per run and one span per step (`LLM`, `TOOL`, `RETRIEVER` or `CHAIN`),
with inputs, outputs, token counts, the checkpoint id and the cache key as attributes.

## Arize Phoenix (recommended: understands OpenInference)

```powershell
docker compose --profile tracing up -d phoenix     # UI at http://localhost:6006
```

Add to `.env` and restart the Black Box server:

```ini
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:6006
```

Run anything (Live run, `examples/expense_agent.py`, a replay) and open Phoenix → project `blackbox-agent`.

## Jaeger

```powershell
docker run -d --name jaeger -p 16686:16686 -p 4318:4318 jaegertracing/all-in-one:latest
```

```ini
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

Open http://localhost:16686 and search service `blackbox-agent`.

## Without a collector

`BLACKBOX_OTEL_CONSOLE=true` prints spans to the server log, and `GET /api/runs/{id}/spans`
returns the same spans as JSON.
