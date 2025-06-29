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
    criteria: str
    reasoning: str
    rating: int

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

if __name__ == "__main__":
    import asyncio
    
    async def main():
        user_goal = input("Please enter your goal or idea: ")
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "module1_output.json")
        
        await run_module_1(user_goal, output_file, tracer, client)
    
    asyncio.run(main())