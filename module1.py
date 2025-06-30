# uv venv
# source .venv/bin/activate

# Install all dependencies from pyproject.toml
# uv pip install -e .

# Install dev dependencies for testing (optional)
# uv pip install -e ".[dev]"

# Set up environment variables
# export OPENAI_API_KEY="your-api-key-here"  # Required (use "dummy" for mock mode)
# export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318/v1/traces"  # Optional
# export OTEL_CONSOLE_EXPORT="true"  # Optional - to see traces in console

# Option 1: Run WITHOUT OpenTelemetry Collector (traces printed to console)
# export OTEL_EXPORTER_OTLP_ENDPOINT="disabled"
# python module1.py

# Option 2: Run WITH OpenTelemetry Collector
# First, start the collector in another terminal:
# ./otelcol-contrib --config testdata/otelcol_file.yaml

# Then run module1 (in your main terminal):
# export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318/v1/traces"
# python module1.py

import json
import logging
import os
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

# Import OTEL types needed for hints and logic
from opentelemetry import trace as otel_trace
from opentelemetry.trace import StatusCode
from pydantic import BaseModel, Field, field_validator

# Import from our new shared utility
from agento_tracing import safe_set, setup_opentelemetry, traced_span


# Define Pydantic models locally or import from a shared models file
class SuccessCriteria(BaseModel):
    criteria: str
    reasoning: str
    rating: int


class Module1Output(BaseModel):
    goal: str
    success_criteria: List[SuccessCriteria]
    selected_criteria: List[SuccessCriteria]
    trace_metadata: Optional[Dict[str, str]] = None


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Agent-like structures using direct OpenAI calls
class Agent:
    def __init__(
        self,
        name: str,
        instructions: str,
        model: str = "gpt-4o",
        output_type: Any = str,
    ):
        self.name = name
        self.instructions = instructions
        self.model = model
        self.output_type = output_type

    async def run(self, client: AsyncOpenAI, prompt: str) -> Any:
        logging.info(f"Running agent: {self.name}")

        is_json_mode = self.output_type != str

        # Add JSON instruction to prompt if needed
        full_prompt = f"{self.instructions}\n\n---PROMPT---\n{prompt}"
        if is_json_mode:
            full_prompt += "\n\nRespond ONLY with a valid JSON object that adheres to the requested schema."

        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": full_prompt}],
            response_format={"type": "json_object"} if is_json_mode else None,
        )
        content = response.choices[0].message.content

        if is_json_mode:
            if (
                hasattr(self.output_type, "__origin__")
                and self.output_type.__origin__ is list
            ):
                model_type = self.output_type.__args__[0]
                data = json.loads(content)
                return [model_type(**item) for item in data]
            elif issubclass(self.output_type, BaseModel):
                return self.output_type.model_validate_json(content)
        return content


# Define the agents for the workflow
search_agent = Agent(
    name="SearchAgent",
    instructions="You are a web search assistant. When given a topic, provide a comprehensive summary of relevant information you find about success criteria and best practices for that topic.",
    output_type=str,
)

generate_criteria_agent = Agent(
    name="CriteriaGenerator",
    instructions="You are a helpful assistant that generates success criteria. Given a goal and search results, create a list of specific, measurable success criteria.",
    output_type=List[SuccessCriteria],
)

evaluate_criteria_agent = Agent(
    name="CriteriaEvaluator",
    instructions="You are an expert evaluator. Given a goal, search results, and proposed criteria, select the most relevant and impactful criteria.",
    output_type=List[SuccessCriteria],
)

tracer = setup_opentelemetry(service_name="Agento-Module-1", module_number=1)


async def run_module_1(
    user_goal: str, output_file: str, tracer: otel_trace.Tracer, client: AsyncOpenAI
):
    """Runs Module 1 with real agent logic and comprehensive tracing."""
    with traced_span(
        tracer, "Agento.Module1.run", {"user.goal": user_goal}
    ) as module_span:
        try:
            # 1. Search
            with traced_span(
                tracer, "search", {"agent.name": search_agent.name}
            ) as search_span:
                search_input = (
                    f"Find information about success criteria for: {user_goal}"
                )
                safe_set(search_span, "ai.prompt", search_input)
                search_summary = await search_agent.run(client, search_input)
                safe_set(search_span, "ai.response", search_summary)

            # 2. Generate Criteria
            with traced_span(
                tracer,
                "generate_criteria",
                {"agent.name": generate_criteria_agent.name},
            ) as gen_span:
                gen_input = f"Goal: {user_goal}\nSearch Results:\n{search_summary}"
                safe_set(gen_span, "ai.prompt", gen_input)
                generated_criteria = await generate_criteria_agent.run(
                    client, gen_input
                )
                safe_set(
                    gen_span,
                    "ai.response",
                    [c.model_dump() for c in generated_criteria],
                )

            # 3. Evaluate Criteria
            with traced_span(
                tracer,
                "evaluate_criteria",
                {"agent.name": evaluate_criteria_agent.name},
            ) as eval_span:
                criteria_json = json.dumps(
                    [c.model_dump() for c in generated_criteria], indent=2
                )
                eval_input = f"Goal: {user_goal}\nSearch Results:\n{search_summary}\nCriteria:\n{criteria_json}"
                safe_set(eval_span, "ai.prompt", eval_input)
                selected_criteria = await evaluate_criteria_agent.run(
                    client, eval_input
                )
                safe_set(
                    eval_span,
                    "ai.response",
                    [c.model_dump() for c in selected_criteria],
                )

            # 4. Finalize Output & Propagate Trace Context
            with traced_span(tracer, "finalize_output"):
                module_span_context = module_span.get_span_context()
                trace_metadata = {
                    "trace_id": format(module_span_context.trace_id, "032x"),
                    "parent_span_id": format(module_span_context.span_id, "016x"),
                }
                module_1_output = Module1Output(
                    goal=user_goal,
                    success_criteria=generated_criteria,
                    selected_criteria=selected_criteria,
                    trace_metadata=trace_metadata,
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


if __name__ == "__main__":
    import asyncio

    async def main():
        user_goal = input("Please enter your goal or idea: ")
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "module1_output.json")

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        await run_module_1(user_goal, output_file, tracer, client)

    asyncio.run(main())
