# python module2.py
import asyncio
import json
import os
import logging
import datetime
from typing import Any, List, Dict, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator

from agents import Agent, GuardrailFunctionOutput, OutputGuardrail, Runner
from agents.handoffs import Handoff  # Keep for potential future use
from agents.run_context import RunContextWrapper
from agents.lifecycle import AgentHooks

load_dotenv()  # Load environment variables

# --- Setup Logging --- (No changes - kept for completeness)
def setup_logging(module_name):
    """Set up logging to both console and file."""
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(logs_dir, f"{module_name}_{timestamp}.log")
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers = []
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    return logger

logger = setup_logging("module2")

# --- Custom Agent Hooks for Detailed Logging --- (No changes - kept for completeness)
class DetailedLoggingHooks(AgentHooks):
    def __init__(self, logger):
        self.logger = logger

    async def on_start(
        self, context: RunContextWrapper[Any], agent: Agent
    ):
        self.logger.info(f"===== API CALL: {agent.name} =====")
        self.logger.info(f"Starting agent: {agent.name}")
        return

    async def on_end(
        self, context: RunContextWrapper[Any], agent: Agent, output: Any
    ):
        self.logger.info(f"===== API RESPONSE: {agent.name} =====")
        try:
            response_content = json.dumps(output.final_output, indent=2) if hasattr(output, 'final_output') else str(output)
            self.logger.info(f"Response from {agent.name}: {response_content}")
        except Exception as e:
            self.logger.info(f"Response from {agent.name}: {str(output)}")
            self.logger.info(f"Could not format response as JSON: {e}")
        return output

    async def on_tool_start(
        self, context: RunContextWrapper[Any], agent: Agent, tool: Any
    ):
        """Called before a tool is invoked."""
        self.logger.info(f"Tool being called by {agent.name}: {tool}")
        return

    async def on_tool_end(
        self, context: RunContextWrapper[Any], agent: Agent, tool: Any, result: str
    ):
        """Called after a tool is invoked."""
        self.logger.info(f"Tool result for {agent.name}: {result}")
        return result

logging_hooks = DetailedLoggingHooks(logger)

# --- Pydantic Models --- (No changes - kept for completeness)
class SuccessCriteria(BaseModel):
    criteria: str
    reasoning: str
    rating: int = Field(..., description="Rating of the criterion (1-10)")

    @field_validator('rating')
    def check_rating(cls, v):
        if not 1 <= v <= 10:
            raise ValueError('Rating must be between 1 and 10')
        return v

class Module1Output(BaseModel):
    goal: str
    success_criteria: list[SuccessCriteria]
    selected_criteria: list[SuccessCriteria]

class PlanItem(BaseModel):
    item_title: str = Field(..., description="A concise title for this plan item.")
    item_description: str = Field(..., description="A description of this step in the plan.")

class PlanOutline(BaseModel):
    plan_title: str = Field(..., description="A title for the overall plan.")
    plan_description: str = Field(..., description="A brief summary of the plan approach")
    plan_items: list[PlanItem] = Field(..., description="A list of plan items.")
    reasoning: str = Field(..., description="Reasoning for why this plan is suitable.")
    rating: int = Field(..., description="Rating of the plan's suitability (1-10).")
    created_by: str = Field(..., description="The name of the agent that created this plan")

    @field_validator('plan_items')
    def check_plan_items(cls, v):
        if len(v) < 3:
            raise ValueError('Must provide at least three plan items')
        return v

    @field_validator('rating')
    def check_rating(cls, v):
        if not 1 <= v <= 10:
            raise ValueError('Rating must be between 1 and 10')
        return v

class Module2Output(BaseModel):
    goal: str
    selected_criteria: list[SuccessCriteria]
    plan_outlines: list[PlanOutline]
    selected_outline: PlanOutline

# --- Specialized Planning Agents --- (Simplified - Only Balanced Planner)

# Base instructions for plan generators.  Removed the {domain} formatting.
base_plan_instructions = """
You are a strategic planner. Given a goal and success criteria,
generate THREE distinct, high-level outlines for plans to achieve the goal.
Each plan outline MUST consist of a title, an overall approach description, and at least THREE distinct plan items.
Each plan item should have a short title and a concise description of the step.
Provide a brief reasoning for each overall plan and a rating from 1 to 10.
Ensure that your plans address ALL of the success criteria.

Important: For each plan outline, include a 'created_by' field with your agent name.
"""

balanced_agent = Agent(
    name="Balanced Planner",
    instructions=base_plan_instructions,  # Use the base instructions directly
    model="gpt-4o",
    output_type=list[PlanOutline],
    # handoff_description="Creates balanced plans that combine practical and creative elements",  # Removed handoff_description
    hooks=logging_hooks,
)


# --- Evaluation agent to select the best plan --- (No changes)
evaluate_outline_agent = Agent(
    name="PlanEvaluator",
    instructions=(
        "You are an expert plan evaluator. You are given a goal, multiple success "
        "criteria, and several plan outlines. Each plan outline contains multiple items. "
        "Select the ONE plan outline that, if implemented, would be most likely "
        "to achieve the goal and satisfy ALL of the success criteria. "
        "Consider how well each plan addresses each criterion. "
        "Provide detailed reasoning for your selection. Output only the selected plan."
    ),
    model="gpt-4o",
    output_type=PlanOutline,
    hooks=logging_hooks,
)

# --- Validation Function --- (No changes)
async def validate_module2_output(
    context: RunContextWrapper[None], agent: Agent, agent_output: Any
) -> GuardrailFunctionOutput:
    """Validates the output of Module 2."""
    try:
        logger.info("Validating Module 2 output...")
        logger.info(f"Output to validate: {json.dumps(agent_output.model_dump() if hasattr(agent_output, 'model_dump') else agent_output, indent=2)}")
        Module2Output.model_validate(agent_output)
        logger.info("Module 2 output validation passed")
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)
    except ValidationError as e:
        logger.error(f"Module 2 output validation failed: {e}")
        return GuardrailFunctionOutput(
            output_info={"error": str(e)}, tripwire_triggered=True
        )

# --- Main Execution Function --- (Modified to call balanced_agent directly)
async def run_module_2(input_file: str, output_file: str) -> None:
    """Runs Module 2."""
    context = RunContextWrapper(context=None)

    try:
        logger.info(f"Starting Module 2, reading input from {input_file}")
        with open(input_file, "r") as f:
            module_1_data = json.load(f)
            logger.info(f"Successfully loaded data from {input_file}")

        module_1_output = Module1Output.model_validate(module_1_data)
        goal = module_1_output.goal
        selected_criteria = module_1_output.selected_criteria

        logger.info(f"Goal: {goal}")
        logger.info(f"Number of selected criteria: {len(selected_criteria)}")

        # Prepare input for the balanced_agent (no triage agent)
        criteria_text = "\n".join([f"- {c.criteria}" for c in selected_criteria])
        planner_input = (
            f"Goal: {goal}\n\n"
            f"Success Criteria:\n{criteria_text}\n\n"
            f"Based on this goal and these success criteria, generate three plan outlines."
        )
        logger.info(f"Planner agent input: {planner_input}")

        # Run the balanced_agent directly
        logger.info("Running balanced planner agent...")
        planner_result = await Runner.run(
            balanced_agent,
            input=planner_input,
            context=context,
        )
        plan_outlines = planner_result.final_output
        logger.info(f"Generated {len(plan_outlines)} plans")

        # Ensure each plan has the created_by field
        logger.info("Processing plan outlines...")
        processed_plan_outlines = []
        for plan in plan_outlines:
            plan_dict = plan.model_dump()
            if 'created_by' not in plan_dict or not plan_dict['created_by']:
                plan_dict['created_by'] = balanced_agent.name # Use agent's name
            processed_plan = PlanOutline.model_validate(plan_dict)
            processed_plan_outlines.append(processed_plan)
            logger.info(f"Processed plan: '{processed_plan.plan_title}' by {processed_plan.created_by}")

        # Format criteria for evaluation input (no change)
        criteria_json = json.dumps([c.model_dump() for c in selected_criteria], indent=2)

        # Evaluate plan outlines (no change)
        logger.info("Evaluating plan outlines...")
        evaluation_input = (
            f"Goal: {goal}\n"
            f"Success Criteria: {criteria_json}\n\n"
            f"Outlines:\n{json.dumps([o.model_dump() for o in processed_plan_outlines], indent=2)}"
        )
        logger.info(f"Evaluation agent input: {evaluation_input[:500]}...")

        evaluation_result = await Runner.run(
            evaluate_outline_agent,
            input=evaluation_input,
            context=context,
        )
        selected_outline = evaluation_result.final_output
        logger.info(f"Selected outline: '{selected_outline.plan_title}'")

        # Prepare and Save Output (no change)
        module_2_output = Module2Output(
            goal=goal,
            selected_criteria=selected_criteria,
            plan_outlines=processed_plan_outlines,
            selected_outline=selected_outline,
        )

        logger.info("Applying output guardrail...")
        guardrail = OutputGuardrail(guardrail_function=validate_module2_output)
        guardrail_result = await guardrail.run(
            agent=evaluate_outline_agent,
            agent_output=module_2_output,
            context=context,
        )
        if guardrail_result.output.tripwire_triggered:
            logger.error(f"Guardrail failed: {guardrail_result.output.output_info}")
            return


        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.basename(output_file)
        name, ext = os.path.splitext(filename)
        timestamped_file = os.path.join(output_dir, f"{name}_{timestamp}{ext}")

        with open(output_file, "w") as f:
            json.dump(module_2_output.model_dump(), f, indent=4)
        with open(timestamped_file, "w") as f:
            json.dump(module_2_output.model_dump(), f, indent=4)

        logger.info(f"Module 2 completed. Output saved to {output_file}")
        logger.info(f"Timestamped output saved to {timestamped_file}")

    except Exception as e:
        logger.error(f"An error occurred in Module 2: {e}")
        import traceback
        logger.error(traceback.format_exc())

async def main():
    logger.info("Starting main function")
    input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    input_file = os.path.join(input_dir, "module1_output.json")
    output_file = os.path.join(input_dir, "module2_output.json")
    await run_module_2(input_file, output_file)
    logger.info("Main function completed")

if __name__ == "__main__":
    logger.info("Module 2 script starting")
    asyncio.run(main())
    logger.info("Module 2 script completed")