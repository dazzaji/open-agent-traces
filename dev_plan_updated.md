# **Updated Developer Plan: Implementing Comprehensive Observability for Agento**   <!-- (original top-level heading retained) -->

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

Our starting point for this work is `module1.py`. This module already contains three separate, parallel systems for capturing execution data:

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
4. **Trace Propagation Contract** – To create a continuous trace, each module (from Module 2 onwards) MUST read a `trace_metadata` block from its input JSON. The `extract_parent_context` helper in `agento_tracing.py` will be used to start its root span as a child of the previous module's span. Each module MUST write its own `trace_metadata` to its output JSON.
5. **Regression tests** – extend the new pytest harness so that each module’s test verifies:

   * presence of root span with `service.name = "Agento-Module-X"`
   * at least one `ai.prompt` and `ai.response` event
   * correct `Link` to previous module’s span.
6. **Dual-mode CI Testing** – The CI pipeline must run tests twice: (1) with a live OTEL collector to test successful export, and (2) with `OTEL_EXPORTER_OTLP_ENDPOINT=disabled` to verify graceful degradation.
7. **Fail-fast policy** – CI should reject any PR that drops span coverage for required attributes/events.
8. **Telemetry contract tests** – feed Module N-1 output into Module N in CI and assert the link and trace-id continuity; prevents accidental breaks.

By bedding these principles into Module 1 now, we lock in the pattern the rest of Agento will follow—guaranteeing that, when you later run cross-module analytics or automated quality gates, every crumb of reasoning is present, searchable, and attributable from the first user keystroke to the final outcome.

---

## **2. Action plan**

### **2.A. Core Principles for Robust Observability**

1. **SDK Decoupling:** Modules MUST NOT depend on any vendor SDK's tracing. Instrumentation must use the standard OpenTelemetry API via `agento_tracing.py`.
2. **Centralized Utilities:** All modules MUST use `agento_tracing.py` for OTEL setup, span creation, and attribute setting.
3. **Collector Fallback:** Modules must handle a missing OTLP collector gracefully.
4. **Filename Convention:** Module scripts must follow `module<N>.py`.

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

---

## **5. Updated Learnings to Guide Revisions for Modules 2-5**

Based on our implementation experience with Module 1, here are comprehensive learnings and refined guidance for updating Modules 2-5. These learnings address both the technical implementation details and the strategic approach needed for a successful migration.

### **5.1 Critical Implementation Discoveries**

#### **5.1.1 OTLP Exporter Configuration**
**Learning:** The `insecure=True` parameter is NOT supported by the HTTP OTLP exporter. This caused immediate failures during testing.

**Action for Modules 2-5:**
```python
# INCORRECT (will fail):
BatchSpanProcessor(OTLPSpanExporter(
    endpoint=os.getenv("OTEL_ENDPOINT", "http://localhost:4318/v1/traces"),
    insecure=True))  # ❌ This parameter doesn't exist

# CORRECT:
BatchSpanProcessor(OTLPSpanExporter(
    endpoint=os.getenv("OTEL_ENDPOINT", "http://localhost:4318/v1/traces")))
```

#### **5.1.2 Import Strategy for Python Modules with Hyphens**
**Learning:** Standard Python import statements fail for files with hyphens in their names (e.g., `module2-plan-generator.py`).

**Action for Modules 2-5:**
```python
# For testing and importing modules with hyphens:
import importlib.util
spec = importlib.util.spec_from_file_location(
    "module2", 
    os.path.join(parent_dir, "module2-plan-generator.py")
)
module2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module2)

# Then access functions/classes:
run_module_2 = module2.run_module_2
setup_opentelemetry = module2.setup_opentelemetry
```

#### **5.1.3 Validation Tool Reality**
**Learning:** The "otel-validate" tool mentioned in the plan doesn't exist as a standard OpenTelemetry tool. We created our own validation script.

**Action for Modules 2-5:**
- Use our custom `validate_otlp_traces.py` script for validation
- Consider adding module-specific validation rules (e.g., Module 2 must have "plan_generation" spans)
- Enhance the validator to check inter-module trace continuity

### **5.2 Enhanced Resource Configuration**

**Learning:** The Resource configuration needs to be created with `Resource.create()` method, not just `Resource()`.

**Refined Implementation for Modules 2-5:**
```python
def setup_opentelemetry(service_name="Agento-Module-X"):  # Replace X with module number
    resource = Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: "1.1.0",
        "service.namespace": "agento",
        "process.pid": os.getpid(),
        "module.type": "plan_generator",  # Add module-specific metadata
        "module.number": 2,  # Numeric identifier for sorting
        "deployment.environment": os.getenv("DEPLOYMENT_ENV", "development")
    })
```

### **5.3 Safe Payload Handling Refinements**

**Learning:** The 8KB limit is crucial for preventing collector overload, but we need consistent handling across all modules.

**Enhanced safe_set for Modules 2-5:**
```python
MAX_ATTR = 8_192  # 8 KB

def safe_set(span, key: str, value, compress_large=False):
    """Attach data safely with optional compression for very large payloads."""
    if not isinstance(value, str):
        value = json.dumps(value, default=str)
    
    if len(value) > MAX_ATTR:
        span.set_attribute(f"{key}_truncated", value[:MAX_ATTR])
        span.set_attribute(f"{key}_size", len(value))  # Add size metadata
        
        # For extremely large payloads in modules 3-5 (expanded plans)
        if compress_large and len(value) > 50_000:
            import gzip
            compressed = gzip.compress(value.encode())
            span.add_event("ai.payload.large.compressed", {
                key: compressed.hex(),  # Store as hex string
                "original_size": len(value),
                "compressed_size": len(compressed)
            })
        else:
            span.add_event("ai.payload.large", {key: value})
    else:
        span.set_attribute(key, value)
```

### **5.4 Inter-Module Trace Linking Strategy**

**Learning:** The Link mechanism requires careful trace ID propagation between modules.

**Implementation Pattern for Modules 2-5:**
```python
async def run_module_2(goal: str, module1_output: dict, output_file: str, tracer, process_id: str):
    """Module 2 with trace linking to Module 1."""
    
    # Extract previous module's trace ID from output
    previous_trace_id = module1_output.get("trace_metadata", {}).get("trace_id")
    previous_span_id = module1_output.get("trace_metadata", {}).get("root_span_id")
    
    # Create link to previous module if trace info available
    links = []
    if previous_trace_id and previous_span_id:
        previous_ctx = SpanContext(
            trace_id=int(previous_trace_id, 16),
            span_id=int(previous_span_id, 16),
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
            is_remote=True
        )
        links.append(Link(previous_ctx, {
            "source": "agento-module-1",
            "relationship": "predecessor"
        }))
    
    # Start this module's root span with link
    with tracer.start_as_current_span("run_module_2", links=links) as module_span:
        # Store this module's trace info for next module
        trace_metadata = {
            "trace_id": format(module_span.get_span_context().trace_id, '032x'),
            "root_span_id": format(module_span.get_span_context().span_id, '016x'),
            "service_name": "Agento-Module-2"
        }
        
        # ... module logic ...
        
        # Include trace metadata in output
        module_output["trace_metadata"] = trace_metadata
```

### **5.5 Shared Tracing Utilities (agento_tracing.py)**

**Learning:** Copy-pasting tracing code across modules leads to inconsistencies and maintenance nightmares.

**Create `agento_tracing.py`:**
```python
"""Shared OpenTelemetry utilities for all Agento modules."""
import os
import json
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.trace import Link, SpanContext, TraceFlags
from contextlib import contextmanager
import random

MAX_ATTR = 8_192  # 8 KB limit

def setup_opentelemetry(service_name: str, module_number: int):
    """Standard OTEL setup for all Agento modules."""
    resource = Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: "1.1.0",
        "service.namespace": "agento",
        "process.pid": os.getpid(),
        "module.number": module_number,
        "deployment.environment": os.getenv("DEPLOYMENT_ENV", "development")
    })
    
    tracer_provider = TracerProvider(resource=resource)
    otel_trace.set_tracer_provider(tracer_provider)
    
    # OTLP export
    tracer_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(
            endpoint=os.getenv("OTEL_ENDPOINT", "http://localhost:4318/v1/traces")))
    )
    
    # Console export for debugging
    if os.getenv("OTEL_CONSOLE_EXPORT", "false").lower() == "true":
        tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    
    return otel_trace.get_tracer(__name__)

def safe_set(span, key: str, value):
    """Safely set attributes with large payload handling."""
    if not isinstance(value, str):
        value = json.dumps(value, default=str)
    
    if len(value) > MAX_ATTR:
        span.set_attribute(f"{key}_truncated", value[:MAX_ATTR])
        span.set_attribute(f"{key}_size", len(value))
        span.add_event("ai.payload.large", {key: value})
    else:
        span.set_attribute(key, value)

def create_module_link(previous_output: dict, source_module: str) -> List[Link]:
    """Create trace links from previous module output."""
    links = []
    trace_meta = previous_output.get("trace_metadata", {})
    
    if trace_meta.get("trace_id") and trace_meta.get("root_span_id"):
        ctx = SpanContext(
            trace_id=int(trace_meta["trace_id"], 16),
            span_id=int(trace_meta["root_span_id"], 16),
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
            is_remote=True
        )
        links.append(Link(ctx, {
            "source": source_module,
            "relationship": "predecessor"
        }))
    
    return links

@contextmanager
def traced_span(tracer, name, attributes=None, record_exception=True):
    """Standard span creation with safe attribute setting."""
    attributes = attributes or {}
    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            safe_set(span, key, value)
        
        try:
            yield span
        except Exception as e:
            if record_exception:
                span.record_exception(e)
                span.set_status(otel_trace.StatusCode.ERROR, str(e))
            raise

def capture_event(stage: str, inputs: dict = None, outputs: dict = None):
    """Dual-write to manual traces and OTEL spans."""
    current_span = otel_trace.get_current_span()
    if current_span and current_span.is_recording():
        current_span.add_event(
            f"capture.{stage}",
            {
                "inputs": json.dumps(inputs, default=str) if inputs else "",
                "outputs": json.dumps(outputs, default=str) if outputs else ""
            }
        )
```

### **5.6 Testing Strategy Updates**

**Learning:** Testing with mocked dependencies is complex but essential. Dynamic imports require special handling.

**Test Template for Modules 2-5:**
```python
# tests/test_module_X_tracing.py
import os
import sys
import json
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import importlib.util

def test_module_X_otlp_export():
    """Test OTLP export for Module X."""
    # Set up environment
    os.environ["OTEL_ENDPOINT"] = "http://localhost:4318/v1/traces"
    os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/protobuf"
    
    # Dynamic import
    spec = importlib.util.spec_from_file_location(
        "moduleX", 
        "moduleX-description.py"
    )
    moduleX = importlib.util.module_from_spec(spec)
    
    # Mock dependencies BEFORE loading module
    with patch.dict('sys.modules', {
        'agents': MagicMock(),
        'openai': MagicMock()
    }):
        spec.loader.exec_module(moduleX)
    
    # Test with mocked agents
    with patch.object(moduleX, "Runner") as mock_runner:
        # Module-specific mocking...
        pass
```

### **5.7 Module-Specific Considerations**

#### **Module 2: Plan Generator**
- **Large Payload Risk:** Generated plans can be very large
- **Solution:** Implement plan summarization for attributes, full plan in events
- **Span Names:** `plan_generation`, `plan_evaluation`, `plan_selection`

#### **Module 3: Plan Expander**
- **Extreme Payload Risk:** Expanded plans can exceed 100KB
- **Solution:** Consider compression for events, store plan sections separately
- **Span Names:** `plan_expansion`, `item_generation`, `expansion_evaluation`

#### **Module 4: Deficiency Identifier**
- **Comparison Challenge:** Need to compare original vs suggested revisions
- **Solution:** Create `comparison` events with side-by-side data
- **Span Names:** `deficiency_analysis`, `revision_generation`, `revision_validation`

#### **Module 5: Revision Implementer**
- **Final Output Size:** Complete plans can be massive
- **Solution:** Store final plan externally, reference in trace
- **Span Names:** `revision_implementation`, `final_assembly`, `quality_check`

### **5.8 Environment Configuration Best Practices**

**Learning:** Environment variables need careful documentation and validation.

**Standard Environment Variables for All Modules:**
```bash
# Required
export OTEL_ENDPOINT="http://localhost:4318/v1/traces"
export OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf"
export OPENAI_API_KEY="your-key"

# Optional
export TRACE_LEGACY="0"  # Enable legacy JSON export during transition
export OTEL_CONSOLE_EXPORT="false"  # Enable console debugging
export DEPLOYMENT_ENV="development"  # Environment metadata
export OTEL_SERVICE_NAME_PREFIX="Agento"  # Override service name prefix
export TRACE_COMPRESSION_THRESHOLD="50000"  # Bytes before compression
```

### **5.9 Validation and Quality Assurance**

**Learning:** Manual validation is error-prone. Automated validation is essential.

**Enhanced Validation Checklist:**
```python
# validation/module_trace_validator.py
def validate_module_traces(module_number: int, trace_file: str):
    """Validate module-specific trace requirements."""
    
    checks = {
        # Universal checks
        "has_resource_attributes": False,
        "has_module_number": False,
        "has_service_version": False,
        
        # Module-specific checks
        1: {
            "has_search_agent_span": False,
            "has_criteria_generation_span": False,
            "has_criteria_evaluation_span": False,
        },
        2: {
            "has_plan_generation_span": False,
            "has_plan_selection_span": False,
            "links_to_module_1": False,
        },
        # ... etc for modules 3-5
    }
    
    # Validation logic...
```

### **5.10 Migration Rollout Strategy**

**Learning:** Big-bang migrations fail. Gradual rollout with feature flags is essential.

**Phased Rollout Plan:**

**Phase 1: Dual-Write (Current State)**
- `TRACE_LEGACY=1` by default
- Both legacy JSON and OTLP export active
- Monitor both outputs for parity

**Phase 2: OTLP Primary (After 2 Weeks)**
- `TRACE_LEGACY=0` by default
- OTLP is primary, legacy available via flag
- Alert on any OTLP export failures

**Phase 3: Legacy Deprecation (After 1 Month)**
- Remove `EnhancedFileTracingProcessor`
- Remove legacy JSON generation code
- Archive existing JSON traces

**Phase 4: Pure OTLP (After 2 Months)**
- Remove all legacy code
- Remove `TRACE_LEGACY` checks
- Celebrate! 🎉

### **5.11 Next Development Plan: Native OTLP Without OpenAI SDK**

**Objective:** Create pure OpenTelemetry instrumentation without dependency on OpenAI Agents SDK tracing.

**Approach:**
1. **Direct LLM Call Instrumentation**
   ```python
   @traced_span(tracer, "llm_call", {"model": "gpt-4", "purpose": "plan_generation"})
   async def call_llm(prompt: str, model: str = "gpt-4"):
       span = otel_trace.get_current_span()
       
       # Capture prompt
       safe_set(span, "llm.prompt", prompt)
       span.add_event("llm.prompt.full", {"content": prompt})
       
       # Make direct OpenAI API call
       response = await openai.ChatCompletion.acreate(
           model=model,
           messages=[{"role": "user", "content": prompt}]
       )
       
       # Capture response
       safe_set(span, "llm.response", response.choices[0].message.content)
       span.add_event("llm.response.full", {
           "content": response.choices[0].message.content,
           "usage": response.usage.dict()
       })
       
       return response
   ```

2. **Custom Agent Instrumentation**
   ```python
   class InstrumentedAgent:
       """Base class for OTEL-instrumented agents."""
       
       def __init__(self, name: str, model: str = "gpt-4"):
           self.name = name
           self.model = model
           self.tracer = otel_trace.get_tracer(f"agento.{name}")
       
       async def run(self, input: str, context: dict = None):
           with self.tracer.start_as_current_span(f"{self.name}.run") as span:
               safe_set(span, "agent.name", self.name)
               safe_set(span, "agent.model", self.model)
               safe_set(span, "agent.input", input)
               
               # Agent logic here
               result = await self._execute(input, context)
               
               safe_set(span, "agent.output", result)
               return result
   ```

3. **Structured Workflow Tracing**
   ```python
   class TracedWorkflow:
       """Base class for instrumented multi-step workflows."""
       
       def __init__(self, name: str):
           self.name = name
           self.tracer = otel_trace.get_tracer(f"agento.workflow.{name}")
       
       async def execute_step(self, step_name: str, func, *args, **kwargs):
           with self.tracer.start_as_current_span(f"{self.name}.{step_name}") as span:
               span.set_attribute("workflow.name", self.name)
               span.set_attribute("workflow.step", step_name)
               
               result = await func(*args, **kwargs)
               
               capture_event(step_name, 
                           inputs={"args": args, "kwargs": kwargs},
                           outputs={"result": result})
               
               return result
   ```

**Benefits of Native OTLP:**
- No dependency on proprietary SDK
- Full control over trace structure
- Easier to extend and customize
- Better integration with standard OTEL tooling
- Portable to any LLM provider

**Implementation Timeline:**
1. Week 1-2: Create base instrumentation classes
2. Week 3-4: Migrate Module 1 to native OTLP
3. Week 5-6: Migrate Modules 2-3
4. Week 7-8: Migrate Modules 4-5
5. Week 9-10: Remove OpenAI SDK dependency
6. Week 11-12: Performance optimization and testing

### **5.12 Key Success Factors**

1. **Standardization:** Use `agento_tracing.py` religiously
2. **Testing:** Every module needs trace validation tests
3. **Documentation:** Update docs with each module migration
4. **Monitoring:** Set up alerts for OTLP export failures
5. **Gradual Rollout:** Use feature flags, never big-bang
6. **Team Training:** Ensure everyone understands OTEL concepts

By following these learnings and implementing the suggested improvements, the migration of Modules 2-5 will be significantly smoother and more reliable than our Module 1 experience. The key is to treat observability as a first-class requirement, not an afterthought.