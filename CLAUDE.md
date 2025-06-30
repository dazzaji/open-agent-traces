# Claude Guidance – Agento Observability Logic Restore (WORK_PLAN_V5)

You are the coding agent for this repository.  
Follow **WORK_PLAN_V5.md** *exactly*; do **NOT** skip ahead.

---

## Environment & Safety
* Allowed shell commands: anything **except** `rm*`, `shutdown*`, `curl*`, `wget*`.
* Runtime, env-vars, and deny-list live in `.claude/settings.json`.
* The project must work whether `OTEL_EXPORTER_OTLP_ENDPOINT` points to a collector or is set to `disabled`.

---

## Workflow

1. **Confirm → Implement → Validate → Commit**
   * Start every turn by summarising the **next unchecked** work-plan item and asking for approval.  
   * After coding, run:
     ```bash
     pytest -q
     ```
   * If the tests pass:
     ```bash
     uv pip install -e ".[dev]"  # install/update deps if needed
     ./otelcol-contrib --config testdata/otelcol_file.yaml || otelcol --config testdata/otelcol_file.yaml &
     python module1.py           # supply a dummy goal when prompted
     python tests/validate_otlp_traces.py ./test-traces.json
     ```
   * Show `git diff --stat` and ask whether to commit.  
   * Commit messages: one-line summary prefixed with the tag from **AGENTS.md § 4**.

2. **Test-first**
   * If an item lacks adequate coverage, extend `tests/test_module1_tracing.py` before changing code.

3. **Stay within plan scope**
   * Modify only the files explicitly mentioned in the current work-plan item.  
   * If a change seems necessary but isn’t covered, pause and ask for guidance.

---

## Quick reference

| Purpose                       | Path                                         |
|-------------------------------|----------------------------------------------|
| Source-of-truth work plan     | `WORK_PLAN_V5.md`                            |
| Shared OTEL utility           | `agento_tracing.py`                          |
| Module 1 entry point          | `module1.py`                                 |
| Unit tests                    | `tests/test_module1_tracing.py`              |
| OTEL collector config         | `testdata/otelcol_file.yaml`                 |
| Trace validation helper       | `tests/validate_otlp_traces.py`              |
| Migration guide               | `docs/tracing_migration.md` (if present)     |

---

### Claude-specific tips

* Keep task prompts **concise and explicit** (≤ 200 tokens).  
* Use the *diff → confirm → commit* loop to minimise context drift.  
* When restoring agent logic, patch only the network call (`AsyncOpenAI.chat.completions.create`) in tests; don’t re-introduce the OpenAI Agent SDK.