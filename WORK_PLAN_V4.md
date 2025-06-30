# Work Plan for Updtes to open-agent-traces Project

## **WORK PLAN (v4): Finalize Module 1 Refactoring and Repository Cleanup**

**Objective:** This plan details the final, precise steps required to complete the observability refactor for Module 1. The goal is to eliminate all legacy artifacts, ensure repository consistency, and finalize the code so it is robust, maintainable, and ready to serve as the template for Modules 2-5.

### **Part 1: Repository Cleanup and Consistency**

**Objective:** Purge all legacy files and references to ensure the project is consistent and that only the new, refactored code is used.

*   **Item 1.1: Standardize File Naming and References**
    *   **Action:** Rename `module1-opentelemetry-gm-1156.py` to `module1.py`.
    *   **Command:** `git mv module1-opentelemetry-gm-1156.py module1.py`
    *   **Verification:** Perform a project-wide search for `module1-opentelemetry-gm-1156.py` and replace all occurrences with `module1.py`. Check: `tests/*.py`, `README.md`, `.github/workflows/ci.yml`, `dev_plan_updated.md`, `pytest.ini`, and `conftest.py`.

*   **Item 1.2: Update Project Dependencies**
    *   **Action:** Pin the OpenTelemetry SDK version in `pyproject.toml`.
    *   **Change in `pyproject.toml`:** Ensure the `opentelemetry-sdk` dependency line reads `opentelemetry-sdk~=1.34`.

*   **Item 1.3: Consolidate or Remove Redundant Test Scripts**
    *   **Action:** The tests in `tests/test_tracing_simple.py` and potentially other old test files are now redundant. Delete them to simplify the test suite. Use the `-f` flag to avoid errors if a file doesn't exist.
    *   **Command:** `git rm -f tests/test_tracing_simple.py tests/test_functions_only.py`

---

### **Part 2: Code Implementation and Refinements**

**Objective:** Address the final code-level gaps to ensure Module 1 is fully functional and the shared utility is well-documented.

*   **Item 2.1: Create Shared Tracing Utility (`agento_tracing.py`)**
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
            """A SpanExporter that does nothing, for graceful fallback."""
            def export(self, spans: list[Span]) -> SpanExportResult: return SpanExportResult.SUCCESS
            def shutdown(self) -> None: pass

        def setup_opentelemetry(service_name: str, module_number: int) -> otel_trace.Tracer:
            """Initializes the OpenTelemetry SDK for a specific Agento module.

            Sets up a TracerProvider with a service resource, a BatchSpanProcessor,
            and an OTLP exporter. Includes a graceful fallback to a NoOpExporter if the
            collector endpoint is disabled or unreachable.

            Args:
                service_name: The name of the service (e.g., "Agento-Module-1").
                module_number: The numerical identifier for the module (e.g., 1).

            Returns:
                An OpenTelemetry Tracer instance configured for the service.
            """
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
            """A context manager for creating OTEL spans with safe attribute setting."""
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
*   **Item 2.2: Refactor `module1.py`**
    *   **Action:** Replace the entire contents of `module1.py` with the following clean, SDK-independent implementation.
    *   **New Contents for `module1.py`:**
        ```python
        import os
        import json
        import logging
        from typing import Any, List, Dict, Optional
        
        from openai import AsyncOpenAI
        from pydantic import BaseModel, Field, field_validator
        
        from agento_tracing import setup_opentelemetry, traced_span, safe_set
        from opentelemetry import trace as otel_trace
        from opentelemetry.trace import StatusCode

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

        async def run_module_1(user_goal: str, output_file: str, tracer: otel_trace.Tracer, client: AsyncOpenAI):
            """Runs Module 1 with comprehensive, SDK-independent tracing."""
            with traced_span(tracer, "Agento.Module1.run", {"user.goal": user_goal}) as module_span:
                try:
                    with traced_span(tracer, "search", {"agent.name": "SearchAgent"}) as search_span:
                        search_input = f"Find information about success criteria for: {user_goal}"
                        safe_set(search_span, "ai.prompt", search_input)
                        search_summary = await call_llm(client, "SearchAgent", search_input)
                        safe_set(search_span, "ai.response", search_summary)

                    with traced_span(tracer, "generate_criteria", {"agent.name": "CriteriaGenerator"}) as gen_span:
                        gen_input = f"Goal: {user_goal}\nSearch Results:\n{search_summary}"
                        safe_set(gen_span, "ai.prompt", gen_input)
                        gen_data = await call_llm(client, "CriteriaGenerator", gen_input)
                        generated_criteria = [SuccessCriteria(**c) for c in gen_data]
                        safe_set(gen_span, "ai.response", [c.model_dump() for c in generated_criteria])

                    with traced_span(tracer, "evaluate_criteria", {"agent.name": "CriteriaEvaluator"}) as eval_span:
                        criteria_json = json.dumps([c.model_dump() for c in generated_criteria], indent=2)
                        eval_input = f"Goal: {user_goal}\nSearch Results:\n{search_summary}\nCriteria:\n{criteria_json}"
                        safe_set(eval_span, "ai.prompt", eval_input)
                        selected_data = await call_llm(client, "CriteriaEvaluator", eval_input)
                        selected_criteria = [SuccessCriteria(**c) for c in selected_data]
                        safe_set(eval_span, "ai.response", [c.model_dump() for c in selected_criteria])

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
                        with open(output_file, "w") as f: f.write(output_json)
                        logging.info(f"Module 1 completed. Output saved to {output_file}")
                
                except Exception as e:
                    module_span.record_exception(e)
                    module_span.set_status(StatusCode.ERROR, str(e))
                    logging.error(f"Module 1 failed: {e}", exc_info=True)
                    raise

        if __name__ == "__main__":
            import asyncio
            async def main():
                logging.info("Module 1 script starting in standalone mode.")
                user_goal = input("Please enter your goal or idea: ")
                output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, "module1_output.json")
                await run_module_1(user_goal, output_file, tracer, client)
                logging.info("Module 1 script finished.")
            asyncio.run(main())
        ```

---

### **Part 3: Project-Level Metadata and Configuration**

*   **Item 3.1: Enhance `pyproject.toml` with Project Metadata**
    *   **Action:** Add a complete `[project]` table to `pyproject.toml`.
    *   **Code to Add/Replace in `pyproject.toml`:**
        ```toml
        [project]
        name = "agento-observability"
        version = "0.1.0"
        description = "A prototype for implementing comprehensive, SDK-independent observability in the Agento multi-module AI system."
        readme = "README.md"
        requires-python = ">=3.9"
        license = { text = "Apache-2.0" }
        authors = [
            { name = "Your Name", email = "your.email@example.com" }
        ]
        dependencies = [
            "opentelemetry-sdk~=1.34",
            "opentelemetry-exporter-otlp-proto-http",
            "openai>=1.0.0",
            "pydantic>=2.0.0",
            "python-dotenv",
        ]

        [project.optional-dependencies]
        dev = [
            "pytest>=7.0",
            "pytest-asyncio",
        ]
        ```

---

### **Part 4: Update Documentation and Dev Plan**

*   **Item 4.1: Update `dev_plan_updated.md`**
    *   **Action:** Add/update sections to codify the new strategy.
    *   **Add Section `2.A. Core Principles`:** Include points on SDK Decoupling, Centralized Utilities, Collector Fallback, and Filename Convention.
    *   **Update Trace Propagation Section:** Replace all mentions of "linking" with "context propagation." Define the `trace_metadata` object as the official contract.
    *   **Update CI Section:** Add the requirement for dual-mode CI testing (with and without a collector).

*   **Item 4.2: Update `README.md` Documentation**
    *   **Action:** Add a section to `README.md` documenting the required environment variables.
    *   **Content for `README.md`:**
        ```markdown
        ## Required Environment Variables

        - `OPENAI_API_KEY`: Your API key for OpenAI services.
        - `OTEL_EXPORTER_OTLP_ENDPOINT`: The URL of your OpenTelemetry collector (e.g., `http://localhost:4318/v1/traces`). Set to `disabled` to run without a collector.
        - `DEPLOYMENT_ENV`: (Optional) The deployment environment, e.g., `development` or `production`.
        - `OTEL_CONSOLE_EXPORT`: (Optional) Set to `true` to see trace data printed to the console for debugging.
        ```

---

### **Part 5: Final Validation and Handoff**

**Objective:** Perform a final series of checks to confirm all fixes have been applied correctly.

*   **Action:** Complete the following checklist.
    *   [ ] **Run Unit Tests:** Execute `pytest -q`. All tests in `tests/test_module1_tracing.py` must pass.
    *   [ ] **Validate Trace Output:**
        1.  Start the OTel collector: `./otelcol-contrib --config testdata/otelcol_file.yaml` (Note: if `otelcol-contrib` is not in your `$PATH`, use `otelcol` or the full path to the binary).
        2.  In a separate terminal, run `python module1.py` interactively.
        3.  Run the validation script: `python tests/validate_otlp_traces.py ./test-traces.json`. The script must exit with code 0.
    *   [ ] **Verify Interactive Mode:** Confirm `python module1.py` prompts for input and completes without errors.
    *   [ ] **Confirm Legacy File Deletion:** Run `git status`. The file `module1-opentelemetry-gm-1156.py` must be in the "deleted" state.
    *   [ ] **Final Grep Check:** Run `grep -R "module1-opentelemetry-gm-1156" .`. The command must produce no output.