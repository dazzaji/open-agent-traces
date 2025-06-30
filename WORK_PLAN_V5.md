# WORK PLAN FOR open-agent-traces Project

## **FINAL WORK PLAN (v5): Restore Logic & Finalize Module 1**

**Objective:** This definitive plan details the final steps to restore the original multi-agent business logic to `module1.py`, fix the critical dependency error, and complete all documentation and cleanup, resulting in a fully functional and observable module.

### **Part 1: Fix Project Installation and Configuration**

**Objective:** Correct the `pyproject.toml` file to make the project installable and clean up the test configuration.

*   **Item 1.1: Correct `pyproject.toml` Dependencies**
    *   **Action:** Edit `pyproject.toml` to remove the invalid dependency and add the proper project metadata.
    *   **Replace the entire contents of `pyproject.toml` with:**
        ```toml
        [project]
        name = "agento-observability"
        version = "0.1.0"
        description = "A prototype for implementing comprehensive, SDK-independent observability in the Agento multi-module AI system."
        readme = "README.md"
        requires-python = ">=3.9"
        license = { text = "Apache-2.0" }
        authors = [ { name = "Dazza Greenwood", email = "dazza@law.mit.edu" } ]
        dependencies = [
            "opentelemetry-sdk~=1.25",
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
    *   **Verification:** After saving, run `uv pip install -e ".[dev]"`. The command must succeed.

*   **Item 1.2: Correct Malformed YAML Config**
    *   **Action:** The `testdata/otelcol_file.yaml` has an extra line of text at the end. Remove this line.
    *   **File:** `testdata/otelcol_file.yaml`
    *   **Change:** Ensure the file ends after the `exporters: [file]` line.

---

### **Part 2: Restore Business Logic to `module1.py`**

**Objective:** Re-integrate the original multi-agent workflow (Search -> Generate -> Evaluate) into the new, observable `module1.py` structure, completely removing the placeholder `call_llm` function.

*   **Item 2.1: Re-introduce Agent Definitions**
    *   **Action:** Re-define the agents for search, generation, and evaluation in `module1.py`. These definitions will now use the standard `openai` library, not the `openai-agents` SDK.
    *   **Code to Add (at the top of `module1.py`, after the Pydantic models):**
        ```python
        # Agent-like structures using direct OpenAI calls
        class Agent:
            def __init__(self, name: str, instructions: str, model: str = "gpt-4o"):
                self.name = name
                self.instructions = instructions
                self.model = model

            async def run(self, client: AsyncOpenAI, prompt: str, output_type: Any) -> Any:
                logging.info(f"Running agent: {self.name}")
                full_prompt = f"{self.instructions}\n\n---PROMPT---\n{prompt}"
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": full_prompt}],
                    response_format={"type": "json_object"} if isinstance(output_type, type) else None,
                )
                content = response.choices[0].message.content
                
                if isinstance(output_type, type) and issubclass(output_type, BaseModel):
                    return output_type.model_validate_json(content)
                elif hasattr(output_type, '__origin__') and output_type.__origin__ is list:
                     return [output_type.__args__[0](**item) for item in json.loads(content)]
                return content

        # Define the agents for the workflow
        search_agent = Agent(name="SearchAgent", instructions="You are a web search assistant...")
        generate_criteria_agent = Agent(name="CriteriaGenerator", instructions="You are a helpful assistant...", output_type=List[SuccessCriteria])
        evaluate_criteria_agent = Agent(name="CriteriaEvaluator", instructions="You are an expert evaluator...", output_type=List[SuccessCriteria])
        ```

*   **Item 2.2: Refactor `run_module_1` to Use Real Agents**
    *   **Action:** Delete the placeholder `call_llm` function and update `run_module_1` to call the newly defined agent structures.
    *   **Delete:** The entire `call_llm` function.
    *   **Replace `run_module_1` with:**
        ```python
        async def run_module_1(user_goal: str, output_file: str, tracer: otel_trace.Tracer, client: AsyncOpenAI):
            """Runs Module 1 with real agent logic and comprehensive tracing."""
            with traced_span(tracer, "Agento.Module1.run", {"user.goal": user_goal}) as module_span:
                try:
                    # 1. Search
                    with traced_span(tracer, "search", {"agent.name": search_agent.name}) as search_span:
                        search_input = f"Find information about success criteria for: {user_goal}"
                        safe_set(search_span, "ai.prompt", search_input)
                        search_summary = await search_agent.run(client, search_input, str)
                        safe_set(search_span, "ai.response", search_summary)

                    # 2. Generate Criteria
                    with traced_span(tracer, "generate_criteria", {"agent.name": generate_criteria_agent.name}) as gen_span:
                        gen_input = f"Goal: {user_goal}\nSearch Results:\n{search_summary}"
                        safe_set(gen_span, "ai.prompt", gen_input)
                        generated_criteria = await generate_criteria_agent.run(client, gen_input, List[SuccessCriteria])
                        safe_set(gen_span, "ai.response", [c.model_dump() for c in generated_criteria])

                    # 3. Evaluate Criteria
                    with traced_span(tracer, "evaluate_criteria", {"agent.name": evaluate_criteria_agent.name}) as eval_span:
                        criteria_json = json.dumps([c.model_dump() for c in generated_criteria], indent=2)
                        eval_input = f"Goal: {user_goal}\nSearch Results:\n{search_summary}\nCriteria:\n{criteria_json}"
                        safe_set(eval_span, "ai.prompt", eval_input)
                        selected_criteria = await evaluate_criteria_agent.run(client, eval_input, List[SuccessCriteria])
                        safe_set(eval_span, "ai.response", [c.model_dump() for c in selected_criteria])

                    # 4. Finalize Output & Propagate Trace Context
                    with traced_span(tracer, "finalize_output"):
                        module_span_context = module_span.get_span_context()
                        trace_metadata = {
                            "trace_id": format(module_span_context.trace_id, '032x'),
                            "parent_span_id": format(module_span_context.span_id, '016x'),
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
        ```
    *   **Note:** The simple `Agent` class above is a placeholder. A more robust implementation would handle retries, structured output parsing, etc., but this structure correctly restores the workflow logic while remaining decoupled from any specific agent SDK.

---

### **Part 3: Finalize Documentation and Tests**

*   **Item 3.1: Add Docstrings to `agento_tracing.py`**
    *   **Action:** Add Google-style docstrings to the public functions in `agento_tracing.py` (`setup_opentelemetry`, `safe_set`, `traced_span`, `extract_parent_context`).

*   **Item 3.2: Update `README.md`**
    *   **Action:** Replace the entire contents of `README.md` with project-specific documentation, including the "Required Environment Variables" section.

*   **Item 3.3: Update `tests/test_module1_tracing.py`**
    *   **Action:** The current tests mock the `call_llm` placeholder. Update them to mock the network layer instead, specifically `openai.AsyncOpenAI.chat.completions.create`. This ensures the test exercises the real `Agent.run` logic.

---

### **Part 4: Final Validation**

*   **Action:** Complete the following checklist to confirm the project is fully functional.
    *   [ ] **Install Dependencies:** Run `uv pip install -e ".[dev]"` and confirm it succeeds.
    *   [ ] **Run Unit Tests:** Execute `pytest -q`. All tests must pass.
    *   [ ] **Run Module Interactively:**
        1.  Start the OTel collector: `./otelcol-contrib --config testdata/otelcol_file.yaml` (or `otelcol`).
        2.  In a separate terminal, run `python module1.py`.
        3.  Confirm it prompts for input and runs to completion, producing `data/module1_output.json`.
        4.  Confirm `test-traces.json` is created and populated.
    *   [ ] **Run with Collector Disabled:** Set `export OTEL_EXPORTER_OTLP_ENDPOINT=disabled` and run `python module1.py` again. Confirm it completes without crashing and prints the "NoOpExporter" warning.