# Claude Guidance for Agento Observability Refactor

**You are the coding agent for this repository.**  
Follow **WORK_PLAN_V2.md** step-by-step; do **NOT** skip ahead.

---

## Environment & Safety
* Allowed shell commands: anything **except** `rm*`, `shutdown*`, `curl*`, `wget*`.  
* Runtime, env-vars and shell deny-list live in `.claude/settings.json`.  
* If `OTEL_EXPORTER_OTLP_ENDPOINT` is `disabled`, tests must still pass.

---

## Workflow

1. **Confirm ➜ Implement ➜ Validate ➜ Commit**  
   * Start every session by summarising the _next_ work-plan item and asking for approval.  
   * After coding, run  
     ```bash
     pytest -q
     otel-validate logs/*.json
     ```  
   * If both commands succeed, show a concise `git diff --stat` and ask whether to commit.  
   * Commit messages: one-line summary prefixed with the tag from **AGENTS.md § 4**.

2. **Test-first** – If tests are missing for an item, write/modify them _before_ changing code.  
   * Unit tests now live in `tests/test_module1_tracing.py`.

3. **Plan boundaries** – You may only modify files that an active work-plan item explicitly calls out.  
   * If a change seems necessary but is _not_ in the plan, pause and ask.

---

## Quick reference

| Purpose | Location |
|---------|----------|
| Work plan source of truth | `WORK_PLAN_V2.md` |
| Shared OTEL utility | `agento_tracing.py` |
| Module 1 entry point | `module1.py` |
| Unit tests | `tests/test_module1_tracing.py` |
| Local OTEL collector | `testdata/otelcol_file.yaml` |
| Migration guide | `docs/tracing_migration.md` |