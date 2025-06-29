# **Developer Plan: Implementing Comprehensive Observability for Agento**   <!-- (original top-level heading retained) -->

---

### **1. Overview: The Goal and Context of Agento**

Agento is a chain-of-expert-modules that starts with a user’s high-level objective and ends with a finished deliverable, letting each module hand structured JSON to the next. Your immediate goal is to make **Module 1 (“Success-Criteria Generator”)** the gold-standard for *observability*: every prompt, every intermediate decision, every output, and every timing signal must flow into a single, standards-compliant OpenTelemetry (OTEL) trace. This isn’t decoration—rich traces are the evidence base for the deep evaluations you plan: benchmarking agent reasoning quality, regression spotting, data-provenance auditing, and eventually automated self-diagnosis and optimisation of the whole pipeline.

The current modules in the pipeline are:

* **Module 1:** Defines Success Criteria from the user's goal.
* **Module 2:** Generates and selects a high-level plan outline.
* **Module 3:** Expands the selected plan outline into detailed, actionable items and evaluates them.
* **Module 4:** Identifies deficiencies in the expanded plan and requests specific revisions.
* **Module 5:** Implements the approved revisions to produce the final, polished plan.

The ultimate success of Agento depends not just on whether it completes a task, but on the *quality* of its decisions at every step. We need to move beyond simple "pass/fail" metrics and build a system for deep, rigorous analysis of the agents' performance. This development plan is the first and most critical step toward achieving that goal by instrumenting Agento with a comprehensive, standards-compliant observability framework.

---

### **A. Starting Point: Multiple, Disparate Data Streams**

Our starting point for this work is `module1-opentelemetry-gm-1156.py`. This module already contains three separate, parallel systems for capturing execution data:

1. **A custom "Manual" trace (`manual_traces.json`):** Rich with complete data on inputs and outputs, but structurally naive and non-standard.
2. **The native `openai-agents` SDK trace (`traces...json`):** Provides low-level detail on LLM API calls but lacks application context and is in a proprietary format.
3. **A custom OpenTelemetry (OTel) implementation (`otel_traces...json`):** Structurally sound and hierarchical, but informationally incomplete—missing the full data payloads needed for deep analysis **and lacking full OTEL `resource` / `scope` blocks.**

We currently have all the data we need, but it is fragmented across three different files in three different formats. This makes holistic analysis difficult, inefficient, and non-scalable.

---

### **B. The Objective: A Single, Authoritative, and Rich Source of Truth**

The primary objective of this development effort is to consolidate these fragmented data streams into a **single, authoritative, and spec-compliant OpenTelemetry trace**. This single source of truth will serve as the bedrock for all future analysis, debugging, and evaluation of the Agento system.

**Why this level of detail is critical:** Our goal is not merely to log that an agent ran. We need to be able to perform complex, multi-faceted evaluations that answer questions like:

* **Causal Analysis:** *Why* did the `CriteriaEvaluator` in Module 1 choose criterion A over criterion B, even though both had the same rating? We need the full prompt, including all candidate criteria and their reasoning, to answer this.
* **Performance Diagnostics:** Did the `RevisionApplier` in Module 5 successfully integrate the feedback from the `ImplementationEvaluator` on its second attempt? We need to compare the full text of `attempt_1` and `attempt_2` against the specific improvement suggestions it was given.
* **Quality Judgements:** How well did the `ItemExpander` in Module 3 address all five success criteria in its expanded text? This requires comparing the full generated text against the complete list of criteria it was provided.

With this depth we can:

* run dataset-level audits of agent reasoning (“show me all spans where the evaluator picked criterion X but later modules flagged mis-alignment”);
* measure latency and cost hotspots;
* correlate user-visible failures back to a single malformed prompt;
* feed traces into model-based critics that suggest guardrail improvements.

---

### **C. Path Forward: Considerations for Subsequent Modules**

The plan laid out for Module 1 will establish the definitive pattern for observability across the entire Agento pipeline. As we instrument Modules 2 through 5, we must carry forward the same principles to ensure a consistent and coherent end-to-end trace.

1. **Shared tracing utilities** – move the new `setup_opentelemetry()`, `safe_set()`, and `capture_event()` helpers into `agento_tracing.py` and have every module import them.
2. **Uniform span taxonomy** – keep the span names predictable: `agent.run`, `validation`, `file_export`, etc., with `agent.name` attribute distinguishing roles.
3. **Payload size discipline** – reuse the 8 KB rule everywhere so collectors never choke.
4. **Trace-ID hand-off** – each module should *always* link to the trace ID received in the incoming JSON, forming a continuous causal chain.
5. **Regression tests** – extend the new pytest harness so that each module’s test verifies:

   * presence of root span with `service.name = "Agento-Module-X"`
   * at least one `ai.prompt` and `ai.response` event
   * correct `Link` to previous module’s span.
6. **Fail-fast policy** – CI should reject any PR that drops span coverage for required attributes/events.
7. **Telemetry contract tests** – feed Module N-1 output into Module N in CI and assert the link and trace-id continuity; prevents accidental breaks.

By bedding these principles into Module 1 now, we lock in the pattern the rest of Agento will follow—guaranteeing that, when you later run cross-module analytics or automated quality gates, every crumb of reasoning is present, searchable, and attributable from the first user keystroke to the final outcome.

---

## **2. Action plan**

**2.1. Upgrade to pure-SDK OTLP export (no custom exporter in the hot path).**
   *Remember to set* `OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf` *in dev and CI.*

**2.2. Add a service-wide `Resource` block** (`service.name`, `service.version`, `service.namespace`, `process.pid`).

**2.3. Introduce `safe_set()` utility**

* ≤ 8 KB → span attribute.
* > 8 KB → span **event** called `"ai.payload.large"` **plus** a `"payload_truncated"` attribute.

**2.4. Wrap each logical step in one high-level span (as today).**

* Add attributes: `agent.name`, `agent.model`, `ai.prompt` (may be truncated), counts, booleans.
* Add events: `full_prompt`, `full_response`, `capture_step` (structured).

**2.5. Inside `capture_step()`**

* Continue writing to the python dict (for the legacy JSON) **and**
* Immediately `add_event()` to the *current* span when tracing is on.

**2.6. Create an OTEL `Link` between `module1_execution` span and the upstream `openai_trace_id`, using a non-zero `span_id`.**

**2.7. Feature flag for legacy files** (`TRACE_LEGACY`). Accept `"1"`, `"true"`, or `"yes"`. Default **off** after two green deploys.

**2.8. Unit-test harness (`tests/test_tracing.py`)**

* Start a local OpenTelemetry collector (e.g. `otelcol-contrib` on `localhost:4318`) **or** use a FileSpanExporter for tests.
* Runs `run_module_1()` with a dummy goal.
* Asserts the emitted OTLP data contains ≥ six spans, each with `agent.name`.
* Asserts at least one span has an event named `full_prompt` **and** one named `full_response`.
* Asserts the root span has a `Link` whose `attributes["source"] == "openai-agents-sdk"`.

**2.9. CI pipeline** – fail build if the test above fails, or if `otel-validate` (open-source JSON schema checker) reports errors.

---

## **3. Item-by-item fix-list (with line references)**

> **All citations point to the *current* code so you know exactly where to edit.**

### 3.1 `setup_opentelemetry()` – replace custom exporter

Delete the inner class **`EnhancedJSONFileExporter`** and the two `SimpleSpanProcessor` rows that add it (lines 35-47 in the current file).

```python
# NEW imports
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
```

```python
# REPLACE lines 69-78
resource = Resource.create({
    SERVICE_NAME: service_name,
    SERVICE_VERSION: "1.1.0",
    "service.namespace": "agento",
    "process.pid": os.getpid(),
})
tracer_provider = TracerProvider(resource=resource)
otel_trace.set_tracer_provider(tracer_provider)

# ADD processors
tracer_provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(
        endpoint=os.getenv("OTEL_ENDPOINT", "http://localhost:4318/v1/traces"),
        insecure=True))
)
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
```

### 3.2 Utility: `safe_set`

Add **above** `traced_span` (just before line 55):

```python
MAX_ATTR = 8_192  # 8 KB

def safe_set(span, key: str, value):
    """Attach data safely – big payloads become events, plus a truncated attr."""
    if not isinstance(value, str):
        value = json.dumps(value, default=str)
    if len(value) > MAX_ATTR:
        span.set_attribute(f"{key}_truncated", value[:MAX_ATTR])
        span.add_event("ai.payload.large", {key: value})
    else:
        span.set_attribute(key, value)
```

### 3.3 `traced_span` helper – use `safe_set`

Replace the attribute loop at lines 61-67 with:

```python
for k, v in attributes.items():
    safe_set(span, k, v)
```

### 3.4 High-level spans – add events and safe-set

#### *search\_agent* block (lines 34-40)

```python
safe_set(search_span, "ai.prompt", search_input)
search_span.add_event("full_prompt", {"text": search_input})
safe_set(search_span, "ai.response_excerpt", search_summary[:1000])
search_span.add_event("full_response", {"search_summary": search_summary})
```

Apply symmetric changes in **criteria\_generation** and **criteria\_evaluation** blocks (replace direct `set_attribute` calls with `safe_set`, add `full_prompt` / `full_response` events).

#### *output\_creation* span

```python
output_span.add_event("module_output.full",
                      {"json": json.dumps(module_1_output.model_dump())})
```

### 3.5 `capture_step()` – dual write to OTEL

```python
current_span = otel_trace.get_current_span()
if current_span and current_span.is_recording():
    current_span.add_event(
        f"capture.{stage}",
        {"inputs": json.dumps(inputs, default=str) if inputs else "",
         "outputs": json.dumps(outputs, default=str) if outputs else ""}
    )
```

### 3.6 OpenAI trace **Link**

```python
from opentelemetry.trace import Link, SpanContext, TraceFlags
import random

openai_ctx = SpanContext(
    trace_id=int(openai_trace_id, 16),
    span_id=random.getrandbits(64) or 1,  # ensure non-zero
    trace_flags=TraceFlags(TraceFlags.SAMPLED),
    is_remote=True
)
with tracer.start_as_current_span(
        "run_module_1",
        links=[Link(openai_ctx, {"source": "openai-agents-sdk"})]) as module_span:
```

(Remove the earlier `module_span.set_attribute("openai_trace_id", …)` line.)

### 3.7 Feature-flag legacy exporters

```python
TRACE_LEGACY = os.getenv("TRACE_LEGACY", "0").lower() in ("1", "true", "yes")
if TRACE_LEGACY:
    file_processor = EnhancedFileTracingProcessor()
    add_trace_processor(file_processor)
    atexit.register(file_processor.shutdown)
```

### 3.8 Remove custom JSON exporter after deprecation window

Add `# TODO: remove after 2025-Q3` above the class definition.

### 3.9 New unit tests (outline)

```python
import json, subprocess, time, os, asyncio, builtins, tempfile
from module1_opentelemetry_gm_1156 import run_module_1

def test_otlp_trace(monkeypatch):
    # spin up a lightweight collector that writes OTLP to a temp file
    tmp_dir = tempfile.mkdtemp()
    out_file = os.path.join(tmp_dir, "trace.json")
    collector = subprocess.Popen(
        ["otelcol-contrib", "--config", "testdata/otelcol_file.yaml",
         "--set", f"exporters.file.path={out_file}"]
    )
    time.sleep(2)  # give collector time to start

    monkeypatch.setenv("OTEL_ENDPOINT", "http://localhost:4318/v1/traces")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    monkeypatch.setattr(builtins, "input", lambda _: "dummy goal")

    asyncio.run(run_module_1("dummy goal", "/dev/null", tracer=None, process_id="test"))

    collector.terminate()
    with open(out_file) as f:
        data = json.load(f)
    spans = data["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert any(s["name"] == "search_agent" for s in spans)
    assert any(e["name"] == "full_prompt" for s in spans for e in s.get("events", []))
    assert any(sp.get("links") for sp in spans), "openai link missing"
```

### 3.10 Documentation & migration note

Create `docs/tracing_migration.md`:

* how to set `OTEL_ENDPOINT` & `OTEL_EXPORTER_OTLP_PROTOCOL`
* how to enable legacy files (`TRACE_LEGACY=1`)
* sample Jaeger queries (`service.name:"Agento-Module-1"`)

### 3.11 Post-merge validation checklist

1. Deploy to staging with `TRACE_LEGACY=1`, ingest traces for one hour.
2. In Jaeger: search `service.name = "Agento-Module-1"`; confirm `full_prompt` & `full_response` events.
3. Disable `TRACE_LEGACY`; repeat validation.
4. Delete `EnhancedFileTracingProcessor` when OTLP and Jaeger look identical.

---

### **4. Closing note**

With these edits you will have **one authoritative, spec-clean OTLP stream** that carries— in a size-safe way—every prompt, thought chain, guardrail verdict and file-write generated by Module 1, while providing a straightforward deprecation path for the bespoke JSON artefacts.