# Claude Guidance for Agento Observability Refactor

Your job is to follow **DEVELOPER_PLAN.md** exactly; environment & permissions are in `.claude/settings.json`.

Follow **DEVELOPER_PLAN.md** exactly.  Runtime, env vars and explicit shell‐deny list are defined in `.claude/settings.json`.

---

## Workflow

1. **Safety** – you may run *any* shell command **except** `rm*`, `shutdown*`, `curl*`, `wget*`.  
2. **Implement plan** – complete Sections 3.1 → 3.8 in order.  
3. **Red/green** – write or update tests first, then code; finish each cycle with `pytest -q`.  
4. **Trace validation** – when tests pass, run  
   ```bash
   otel-validate logs/*.json
````

to confirm OTLP compliance.
5\. **Commit** – one-line summary, optional body. Prefix with the tag from **AGENTS.md § 4**.

---

## Quick reference

* Unit tests: `tests/test_tracing.py`
* Local collector config: `testdata/otelcol_file.yaml`
* Analyst rollout doc: `docs/tracing_migration.md`