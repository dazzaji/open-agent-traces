# Updates to open-agent-traces Project

## **FINAL WORK PLAN: Refactor Agento Observability**

**Objective:** This plan details the necessary steps to refactor `Agento`'s Module 1 to produce standards-compliant OpenTelemetry (OTEL) traces, fully decouple it from the `openai-agents` SDK, and establish a robust, scalable observability pattern for all subsequent modules.

### **Part 1: Foundational Cleanup & Code Centralization**

**Objective:** Create a shared, robust utility for tracing and standardize project structure before refactoring any module logic.

*   **Item 1.1: Standardize File Naming and References**
    *   **Action:** Rename `module1-opentelemetry-gm-1156.py` to `module1.py`.
    *   **Command:** `git mv module1-opentelemetry-gm-1156.py module1.py`
    *   **Verification:** Perform a project-wide search for `module1-opentelemetry-gm-1156.py` and replace all occurrences with `module1.py`. Check: `tests/*.py`, `README.md`, `.github/workflows/ci.yml`, `dev_plan_updated.md`, `pytest.ini`, and `conftest.py`.

*   **Item 1.2: Update Project Dependencies**
    *   **Action:** Pin the OpenTelemetry SDK version in `pyproject.toml`.
    *   **Change in `pyproject.toml`:** Ensure the `opentelemetry-sdk` dependency line reads `opentelemetry-sdk~=1.34`.

*   **Item 1.3: Create Shared Tracing Utility (`agento_tracing.py`)**
    *   **Action:** Create a new file named `agento_tracing.py` in the root directory.
    *   **Contents for `agento_tracing.py`:**
        ```python
        # agento_tracing.py
        import os
        import sys
        import json
        import logging
        import re
        import random
        import hashlib
        from contextlib import contextmanager
        from typing import List, Dict, Any, Optional

        from opentelemetry import trace as otel_trace
        from opentelemetry.sdk.trace import TracerProvider, Span
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
            SpanExporter,
            SpanExportResult,
            SimpleSpanProcessor,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
        from opentelemetry.trace import Link, SpanContext, TraceFlags, StatusCode, SpanKind

        class NoOpSpanExporter(SpanExporter):
            def export(self, spans: list[Span]) -> SpanExportResult: return SpanExportResult.SUCCESS
            def shutdown(self) -> None: pass

        def setup_opentelemetry(service_name: str, module_number: int) -> otel_trace.Tracer:
            """Standard OTEL setup for all Agento modules with a graceful fallback."""
            resource = Resource.create({
                SERVICE_NAME: service_name,
                SERVICE_VERSION: "1.1.0",
                "service.namespace": "agento",
                "process.pid": os.getpid(),
                "module.number": module_number,
                "deployment.environment": os.getenv("DEPLOYMENT_ENV", "development")
            })
            tracer_provider = TracerProvider(resource=resource)
            endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
            
            if endpoint.lower() == 'disabled':
                logging.warning("OTEL_EXPORTER_OTLP_ENDPOINT is 'disabled'. Using NoOpExporter.")
                exporter = NoOpSpanExporter()
            else:
                try:
                    exporter = OTLPSpanExporter(endpoint=endpoint, timeout=5)
                except Exception as e:
                    logging.error(f"Failed to initialize OTLP exporter to '{endpoint}': {e}. Spans will be dropped.")
                    exporter = NoOpSpanExporter()
            
            tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
            if os.getenv("OTEL_CONSOLE_EXPORT", "true").lower() == "true":
                tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
            
            otel_trace.set_tracer_provider(tracer_provider)
            return otel_trace.get_tracer(f"agento.module{module_number}")

        MAX_ATTR_BYTES = 8192

        def safe_set(span: Span, key: str, value: Any):
            """Safely set attributes, handling large payloads by creating events."""
            if not isinstance(value, str):
                value = json.dumps(value, default=str)
            value_bytes = value.encode("utf-8")
            if len(value_bytes) > MAX_ATTR_BYTES:
                sha256_hash = hashlib.sha256(value_bytes).hexdigest()
                span.set_attribute(f"{key}_truncated", value_bytes[:MAX_ATTR_BYTES].decode("utf-8", "ignore"))
                span.set_attribute(f"{key}_sha256", sha256_hash)
                span.add_event(name="ai.payload.large", attributes={key: value})
            else:
                span.set_attribute(key, value)

        @contextmanager
        def traced_span(tracer: otel_trace.Tracer, name: str, attributes: Optional[Dict] = None):
            with tracer.start_as_current_span(name) as span:
                if attributes:
                    for k, v in attributes.items():
                        safe_set(span, k, v)
                try:
                    yield span
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(StatusCode.ERROR, str(e))
                    raise
        
        def extract_parent_context(trace_metadata: Optional[Dict]) -> Optional[SpanContext]:
            """Safely extracts a parent SpanContext from a dictionary."""
            if not trace_metadata: return None
            trace_id_str, parent_span_id_str = trace_metadata.get("trace_id"), trace_metadata.get("parent_span_id")
            if not (trace_id_str and parent_span_id_str): return None
            try:
                return SpanContext(
                    trace_id=int(trace_id_str, 16),
                    span_id=int(parent_span_id_str, 16),
                    is_remote=True,
                    trace_flags=TraceFlags(TraceFlags.SAMPLED)
                )
            except (ValueError, TypeError):
                logging.warning("Could not parse incoming trace_metadata.")
                return None
        ```

---

### **Part 2: Refactor `module1.py`**

**Objective:** Fully decouple Module 1 from the `openai-agents` SDK, fix runtime errors, and implement the new observability patterns.

*   **Item 2.1: Update Imports and Setup in `module1.py`**
    *   **Action:** Replace all old tracing and agent-related imports with the new shared utility. Initialize the tracer and OpenAI client *once* at the top of the script.
    *   **Replace the entire header of `module1.py` with:**
        ```python
        import os
        import json
        import logging
        from typing import Any, List, Dict, Optional
        
        from openai import AsyncOpenAI
        from pydantic import BaseModel, Field, field_validator
        
        # Import from our new shared utility
        from agento_tracing import setup_opentelemetry, traced_span, safe_set
        
        # Import OTEL types needed for hints and logic
        from opentelemetry import trace as otel_trace
        from opentelemetry.trace import StatusCode
        
        # Define Pydantic models locally or import from a shared models file
        class SuccessCriteria(BaseModel):
            criteria: str; reasoning: str; rating: int
        
        class Module1Output(BaseModel):
            goal: str
            success_criteria: List[SuccessCriteria]
            selected_criteria: List[SuccessCriteria]
            trace_metadata: Optional[Dict[str, str]] = None
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        tracer = setup_opentelemetry(service_name="Agento-Module-1", module_number=1)
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        ```

*   **Item 2.2: Implement Stand-in for LLM Logic**
    *   **Action:** Create a placeholder function to simulate agent calls, keeping parameter order consistent.
    *   **Add to `module1.py`:**
        ```python
        async def call_llm(client: AsyncOpenAI, agent_name: str, prompt: str) -> Any:
            """Placeholder for direct LLM calls."""
            logging.info(f"Simulating LLM call for {agent_name}...")
            if "SearchAgent" in agent_name:
                return "Mocked search summary from direct OpenAI call."
            elif "CriteriaGenerator" in agent_name:
                return [{"criteria": "Mock Criterion 1", "reasoning": "Reason 1", "rating": 8}]
            elif "CriteriaEvaluator" in agent_name:
                return [{"criteria": "Mock Selected Criterion", "reasoning": "Selected", "rating": 9}]
            return ""
        ```

*   **Item 2.3: Refactor `run_module_1` to be SDK-Independent**
    *   **Action:** Rewrite the `run_module_1` function to use the new helpers and the `call_llm` placeholder.
    *   **Replace the existing `run_module_1` function with:**
        ```python
        async def run_module_1(user_goal: str, output_file: str, tracer: otel_trace.Tracer, client: AsyncOpenAI):
            """Runs Module 1 with comprehensive, SDK-independent tracing."""
            with traced_span(tracer, "Agento.Module1.run", {"user.goal": user_goal}) as module_span:
                try:
                    # 1. Search
                    with traced_span(tracer, "search", {"agent.name": "SearchAgent"}) as search_span:
                        search_input = f"Find information about success criteria for: {user_goal}"
                        safe_set(search_span, "ai.prompt", search_input)
                        search_summary = await call_llm(client, "SearchAgent", search_input)
                        safe_set(search_span, "ai.response", search_summary)

                    # 2. Generate Criteria
                    with traced_span(tracer, "generate_criteria", {"agent.name": "CriteriaGenerator"}) as gen_span:
                        gen_input = f"Goal: {user_goal}\nSearch Results:\n{search_summary}"
                        safe_set(gen_span, "ai.prompt", gen_input)
                        gen_data = await call_llm(client, "CriteriaGenerator", gen_input)
                        generated_criteria = [SuccessCriteria(**c) for c in gen_data]
                        safe_set(gen_span, "ai.response", [c.model_dump() for c in generated_criteria])

                    # 3. Evaluate Criteria
                    with traced_span(tracer, "evaluate_criteria", {"agent.name": "CriteriaEvaluator"}) as eval_span:
                        criteria_json = json.dumps([c.model_dump() for c in generated_criteria], indent=2)
                        eval_input = f"Goal: {user_goal}\nSearch Results:\n{search_summary}\nCriteria:\n{criteria_json}"
                        safe_set(eval_span, "ai.prompt", eval_input)
                        selected_data = await call_llm(client, "CriteriaEvaluator", eval_input)
                        selected_criteria = [SuccessCriteria(**c) for c in selected_data]
                        safe_set(eval_span, "ai.response", [c.model_dump() for c in selected_criteria])

                    # 4. Finalize Output & Propagate Trace Context
                    with traced_span(tracer, "finalize_output") as final_span:
                        module_span_context = module_span.get_span_context()
                        trace_metadata = {
                            "trace_id": format(module_span_context.trace_id, '032x'),
                            "parent_span_id": format(module_span_context.span_id, '016x'),
                            "service_name": "Agento-Module-1"
                        }
                        
                        module_1_output = Module1Output(
                            goal=user_goal,
                            success_criteria=generated_criteria,
                            selected_criteria=selected_criteria,
                            trace_metadata=trace_metadata
                        )
                        
                        output_json = module_1_output.model_dump_json(indent=4)
                        with open(output_file, "w") as f:
                            f.write(output_json)
                        logging.info(f"Module 1 completed. Output saved to {output_file}")
                
                except Exception as e:
                    module_span.record_exception(e)
                    module_span.set_status(StatusCode.ERROR, str(e))
                    logging.error(f"Module 1 failed: {e}", exc_info=True)
                    raise
        ```

*   **Item 2.4: Cleanup `module1.py`**
    *   **Action:** Delete all now-unused code from `module1.py`. The final file should only contain the new imports, Pydantic models, `call_llm`, `run_module_1`, and the `if __name__ == "__main__":` block.
    *   **Clarification:** This involves deleting all legacy code, including but not limited to: the old `setup_logging` function, the `MANUAL_TRACES` dictionary, the `capture_step` helper, the `EnhancedFileTracingProcessor` class, and all classes and functions related to the `openai-agents` SDK (e.g., `DetailedLoggingHooks`, `search_agent`, `Runner`).

---

### **Part 3: Update `dev_plan_updated.md`**

**Objective:** Codify the revised, robust strategy for all future module development.

*   **Item 3.1: Add Core Principles to Dev Plan**
    *   **Action:** Add a new section `2.A. Core Principles for Robust Observability`.
    *   **Content:**
        1.  **SDK Decoupling:** Modules MUST NOT depend on any vendor SDK's tracing. Instrumentation must use the standard OpenTelemetry API via `agento_tracing.py`.
        2.  **Centralized Utilities:** All modules MUST use `agento_tracing.py` for OTEL setup, span creation, and attribute setting.
        3.  **Collector Fallback:** Modules must handle a missing OTLP collector gracefully.
        4.  **Filename Convention:** Module scripts must follow `module<N>.py`.

*   **Item 3.2: Define the Trace Propagation Contract**
    *   **Action:** Replace all mentions of "linking" between modules with "context propagation." Define the `trace_metadata` object as the official contract.
    *   **Add to Dev Plan:** "To create a continuous trace, each module (from Module 2 onwards) MUST read a `trace_metadata` block from its input JSON. The `extract_parent_context` helper in `agento_tracing.py` will be used to start its root span as a child of the previous module's span. Each module MUST write its own `trace_metadata` to its output JSON."

*   **Item 3.3: Update CI and Testing Requirements**
    *   **Action:** Add a requirement for dual-mode CI testing.
    *   **Add to Dev Plan:** "The CI pipeline must run tests twice: (1) with a live OTEL collector to test successful export, and (2) with `OTEL_EXPORTER_OTLP_ENDPOINT=disabled` to verify graceful degradation."

---

### **Part 4: Update Tests and Documentation**

*   **Item 4.1: Update Test Suite**
    *   **Action:** Refactor `tests/test_tracing.py` to `tests/test_module1_tracing.py`. Update the test logic to mock `call_llm` instead of `agents.Runner`.
    *   **Remove:** Assertions that check for OTEL `Links`.
    *   **Add:** An assertion to load `module1_output.json` and verify that a correctly formatted `trace_metadata` block is present.

*   **Item 4.2: Update `README.md` Documentation**
    *   **Action:** Add a section to the main `README.md` file documenting the required environment variables.
    *   **Content for `README.md`:**
        ```markdown
        ## Required Environment Variables

        - `OPENAI_API_KEY`: Your API key for OpenAI services.
        - `OTEL_EXPORTER_OTLP_ENDPOINT`: The URL of your OpenTelemetry collector (e.g., `http://localhost:4318/v1/traces`). Set to `disabled` to run without a collector.
        - `DEPLOYMENT_ENV`: (Optional) The deployment environment, e.g., `development` or `production`.
        - `OTEL_CONSOLE_EXPORT`: (Optional) Set to `true` to see trace data printed to the console for debugging.
        ```