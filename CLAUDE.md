# Claude Guidance – Agento Observability Cleanup (WORK_PLAN_V4)

You are the coding agent for this repository.  
Follow **WORK_PLAN_V4.md** *exactly*, step by step. Do **NOT** skip ahead.

---

## Environment & Safety
* Allowed shell commands: anything **except** `rm*`, `shutdown*`, `curl*`, `wget*`.
* Runtime, env-vars, and the shell deny-list live in `.claude/settings.json`.
* If `OTEL_EXPORTER_OTLP_ENDPOINT` is `disabled`, all tests must still pass.

---

## Workflow

1. **Confirm → Implement → Validate → Commit**  
   * Start every session by summarising the *next unchecked* work-plan item and asking for approval.  
   * After coding, run:  
     ```bash
     pytest -q
     ```  
     If tests succeed:  
     ```bash
     ./otelcol-contrib --config testdata/otelcol_file.yaml || otelcol --config testdata/otelcol_file.yaml
     python module1.py  # supply a sample goal when prompted
     python tests/validate_otlp_traces.py ./test-traces.json
     ```  
   * When all commands succeed, show `git diff --stat` and ask whether to commit.  
   * Commit messages: one-line summary prefixed with the tag from **AGENTS.md § 4**.

2. **Test-first**  
   * If a work-plan item lacks adequate coverage, write or modify tests *before* changing code.  
   * Primary suite lives in `tests/test_module1_tracing.py`.

3. **Plan boundaries**  
   * Touch only files explicitly mentioned by the current work-plan item.  
   * If a required change isn’t covered, pause and ask before proceeding.

---

## Quick reference

| Purpose | Path |
|---------|------|
| Source of truth work plan | `WORK_PLAN_V4.md` |
| Shared OTEL utility | `agento_tracing.py` |
| Module 1 entry point | `module1.py` |
| Unit tests | `tests/test_module1_tracing.py` |
| OTEL collector config | `testdata/otelcol_file.yaml` |
| Trace validation helper | `tests/validate_otlp_traces.py` |
| Migration guide | `docs/tracing_migration.md` |