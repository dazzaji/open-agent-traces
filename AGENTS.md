# Agento — Module-1 Observability Guidance for Codex
> Make Module-1 the reference implementation for spec-clean OTEL tracing.

## 1. Project Snapshot
*   **Purpose**: Capture *every* prompt, decision, output, and timing signal in a single OTLP stream so later analyst modules can run causal, performance, and quality audits.
*   **Stack**: Python 3.12, OpenTelemetry SDK 1.27, `opentelemetry-exporter-otlp-proto-http`, pytest.

## 2. Repo Context
You have two key inputs:
*   `module1.py` — file to refactor.
*   `DEVELOPER_PLAN.md` — authoritative change list.

## 3. Coding Conventions
1.  Format with **Black**, import-sort with **isort**, lint with **flake8**.
2.  Span names **snake_case**; attributes **snake_case**.
3.  Every new function needs Google-style docstrings.

## 4. Observability Tasks for Codex
Implement incrementally; commit after each row.

| #  | Commit prefix                       | Core task                                        | Verification                       |
|:---|:------------------------------------|:-------------------------------------------------|:-----------------------------------|
| 1  | `feat(obs): setup otlp exporter`    | Add OTLP exporter & Resource block.              | `pytest -q`                        |
| 2  | `refactor(obs): implement safe_set` | Add `safe_set` utility & refactor spans.         | `pytest tests/test_tracing.py -k safe_set` |
| 3  | `refactor(obs): add otel link`      | Replace `openai_trace_id` attr with OTEL Link.   | `pytest -q`                        |
| 4  | `feat(obs): add legacy flag`        | Wrap `EnhancedFileTracingProcessor` in `TRACE_LEGACY`. | `TRACE_LEGACY=1 pytest -q`         |
| 5  | `test(obs): add collector test`     | Add full unit-test harness.                      | `pytest tests/test_tracing.py`     |
| 6  | `docs(obs): add migration notes`    | Write `docs/tracing_migration.md`.               | `ls docs/`                         |

## 5. Test Execution
*   `make otel-local` spins up an **otelcol-contrib** container receiver.
*   Pipeline must pass `pytest -q && otel-validate logs/*.json`.
*   *If your distro ships `otelvalidate` not `otel-validate`, symlink accordingly.*

## 6. Safety & Limits
Only the following commands may be executed (mirrors `.claude/settings.json`):
`pytest`, `make`, `docker`, `otel-validate`, `otelcol-contrib`, `git`.

Large payloads (> 8 KB) go to an `ai.payload.large` event; store the first 8 KB in a `_truncated` attribute.