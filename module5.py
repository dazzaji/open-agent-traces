# python module5.py

import asyncio
import json
import os
import logging
import datetime
import re
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator

from agents import Agent, GuardrailFunctionOutput, OutputGuardrail, Runner
from agents.run_context import RunContextWrapper
from agents.lifecycle import AgentHooks

load_dotenv()  # Load environment variables

# --- Setup Logging ---
def setup_logging(module_name):
    """Set up logging to both console and file."""
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a timestamp for the log filename
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(logs_dir, f"{module_name}_{timestamp}.log")
    
    # Also create a verbose log file that captures everything without truncation
    verbose_log_file = os.path.join(logs_dir, f"{module_name}_verbose_{timestamp}.log")
    
    # Configure logging
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # Standard file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Verbose file handler (same level, but we won't truncate messages logged to this)
    verbose_file_handler = logging.FileHandler(verbose_log_file)
    verbose_file_handler.setLevel(logging.INFO)
    verbose_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    verbose_file_handler.setFormatter(verbose_format)
    
    # Create a separate logger for verbose logging
    verbose_logger = logging.getLogger(f"{module_name}_verbose")
    verbose_logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if verbose_logger.handlers:
        verbose_logger.handlers = []
    
    verbose_logger.addHandler(verbose_file_handler)
    
    return logger, verbose_logger

# Initialize loggers
logger, verbose_logger = setup_logging("module5")

# Helper function to log both normal and verbose
def log_info(message, truncate=False, max_length=None):
    """Log to both normal and verbose logs, with optional truncation for normal log."""
    # Always log full message to verbose log
    verbose_logger.info(message)
    
    # Optionally truncate for normal log
    if truncate and max_length and len(message) > max_length:
        truncated = message[:max_length] + f"... [truncated, full message in verbose log]"
        logger.info(truncated)
    else:
        logger.info(message)

# --- Text Validation Functions ---
def sanitize_text(text: str) -> str:
    """Clean and validate text to prevent corruption."""
    if not isinstance(text, str):
        return str(text)
        
    # Remove any non-printable or control characters
    text = ''.join(char for char in text if char.isprintable() or char in ['\n', '\t', ' '])
    
    # Check for obvious corruption patterns (random Unicode characters, etc.)
    # This regex looks for clusters of non-English characters that might indicate corruption
    corruption_pattern = r'[\u0400-\u04FF\u0600-\u06FF\u0900-\u097F\u3040-\u309F\u30A0-\u30FF\u3130-\u318F\uAC00-\uD7AF]{3,}'
    
    # Replace corrupted sections with a note
    text = re.sub(corruption_pattern, '[corrupted text removed]', text)
    
    # We don't truncate text here - we want to preserve the full content
    return text

# --- Custom Agent Hooks for Detailed Logging ---
class DetailedLoggingHooks(AgentHooks):
    def __init__(self, logger, verbose_logger):
        self.logger = logger
        self.verbose_logger = verbose_logger

    async def on_start(
        self, context: RunContextWrapper[Any], agent: Agent
    ):
        """Log details before LLM generation."""
        log_info(f"===== API CALL: {agent.name} =====")
        log_info(f"Starting agent: {agent.name}")
        return
    
    async def on_end(
        self, context: RunContextWrapper[Any], agent: Agent, output: Any
    ):
        """Log details after LLM generation."""
        log_info(f"===== API RESPONSE: {agent.name} =====")
        
        # Format the response for better readability
        try:
            if hasattr(output, 'final_output'):
                # Handle different response types with sanitization
                if hasattr(output.final_output, 'revised_text'):
                    output.final_output.revised_text = sanitize_text(output.final_output.revised_text)
                if hasattr(output.final_output, 'reasoning'):
                    output.final_output.reasoning = sanitize_text(output.final_output.reasoning)
                if hasattr(output.final_output, 'feedback'):
                    output.final_output.feedback = sanitize_text(output.final_output.feedback)
                if hasattr(output.final_output, 'criteria_fulfillment'):
                    output.final_output.criteria_fulfillment = sanitize_text(output.final_output.criteria_fulfillment)
                
                response_content = json.dumps(output.final_output, indent=2) 
                
                # Log to verbose logger always
                self.verbose_logger.info(f"Response from {agent.name}: {response_content}")
                
                # Log to regular logger, potentially truncated
                if len(response_content) > 5000:
                    truncated = response_content[:5000] + "... [truncated, full response in verbose log]"
                    self.logger.info(f"Response from {agent.name}: {truncated}")
                else:
                    self.logger.info(f"Response from {agent.name}: {response_content}")
            else:
                self.logger.info(f"Response from {agent.name}: {str(output)}")
                self.verbose_logger.info(f"Response from {agent.name}: {str(output)}")
        except Exception as e:
            self.logger.info(f"Response from {agent.name}: {str(output)}")
            self.logger.info(f"Could not format response as JSON: {e}")
            self.verbose_logger.info(f"Response from {agent.name}: {str(output)}")
            self.verbose_logger.info(f"Could not format response as JSON: {e}")
        return output

# Create logging hooks
logging_hooks = DetailedLoggingHooks(logger, verbose_logger)

# --- Pydantic Models ---
class SuccessCriteria(BaseModel):
    criteria: str
    reasoning: str
    rating: int = Field(..., description="Rating of the criterion (1-10)")
    
    @field_validator('rating')
    def check_rating(cls, v):
        if not 1 <= v <= 10:
            raise ValueError('Rating must be between 1 and 10')
        return v

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

class EvalResult(BaseModel):
    result: str = Field(..., description="Either the word 'pass' or the word 'fail'.")
    reasoning: str = Field(..., description="The evaluator's reasoning")
    criteria: SuccessCriteria = Field(..., description="The success criterion being evaluated against")

    @field_validator('result')
    def check_result(cls, v):
        if v.lower() not in ["pass", "fail"]:
            raise ValueError("Result must be 'pass' or 'fail'")
        return v.lower()
    
    @field_validator('reasoning')
    def validate_reasoning(cls, v):
        """Validate and sanitize reasoning text."""
        return sanitize_text(v)

class RevisionRequest(BaseModel):
    revision_request_content: str = Field(..., description="Specific requested revision.")
    reasoning: str = Field(..., description="Why this revision is necessary.")
    targeted_criteria: List[str] = Field(..., description="The criteria this revision addresses.")
    
    @field_validator('revision_request_content')
    def validate_request(cls, v):
        """Validate and sanitize revision request text."""
        return sanitize_text(v)
    
    @field_validator('reasoning')
    def validate_reasoning(cls, v):
        """Validate and sanitize reasoning text."""
        return sanitize_text(v)

class RevisionEvaluation(BaseModel):
    approved: bool = Field(..., description="Whether the revision is approved (True) or rejected (False).")
    reasoning: str = Field(..., description="Reasoning for approval or rejection.")
    impact_assessment: str = Field(..., description="Assessment of how this revision impacts each criterion.")
    
    @field_validator('reasoning')
    def validate_reasoning(cls, v):
        """Validate and sanitize reasoning text."""
        return sanitize_text(v)
    
    @field_validator('impact_assessment')
    def validate_impact(cls, v):
        """Validate and sanitize impact assessment text."""
        return sanitize_text(v)

class ItemDetail(BaseModel):
    item_title: str = Field(..., description="Title of the plan item")
    original_evaluation: Dict[str, str] = Field(..., description="Summary of original evaluations for this item")
    revision_request: Optional[RevisionRequest] = Field(None, description="The requested revision if any")
    revision_evaluation: Optional[RevisionEvaluation] = Field(None, description="Evaluation of the revision request")

class Module4Output(BaseModel): # For loading the JSON
    goal: str
    selected_criteria: list[SuccessCriteria]
    selected_outline: PlanOutline  # Original outline
    expanded_outline: PlanOutline  # Expanded items with full descriptions
    evaluation_results: list[EvalResult]  # Original evaluation results
    item_details: list[ItemDetail]  # Details about each item's revision
    criteria_coverage_summary: Dict[str, Dict[str, int]] = Field(
        default_factory=dict, description="Summary of original vs post-revision criteria coverage"
    )

class AppliedRevision(BaseModel):
    revised_text: str = Field(..., description="The revised text after applying the revision.")
    feedback: Optional[str] = Field(None, description="Additional feedback about the revision.")
    
    @field_validator('revised_text')
    def validate_revised_text(cls, v):
        """Validate and sanitize revised text."""
        return sanitize_text(v)
    
    @field_validator('feedback')
    def validate_feedback(cls, v):
        """Validate and sanitize feedback text."""
        if v is None:
            return v
        return sanitize_text(v)

class RevisionImplementationEvaluation(BaseModel):
    meets_criteria: bool = Field(..., description="Whether the revision implementation meets the success criteria.")
    reasoning: str = Field(..., description="Reasoning for the evaluation.")
    criteria_fulfillment: str = Field(..., description="How well each targeted criterion is fulfilled as a formatted string.")
    improvement_suggestions: Optional[str] = Field(None, description="Suggestions for further improvement if needed.")
    
    @field_validator('reasoning')
    def validate_reasoning(cls, v):
        """Validate and sanitize reasoning text."""
        return sanitize_text(v)
    
    @field_validator('improvement_suggestions')
    def validate_suggestions(cls, v):
        """Validate and sanitize improvement suggestions."""
        if v is None:
            return v
        return sanitize_text(v)

class RevisionImplementationResult(BaseModel):
    item_title: str
    original_text: str
    revision_request: RevisionRequest
    applied_revision: AppliedRevision
    implementation_evaluation: RevisionImplementationEvaluation
    attempt_count: int = Field(1, description="Number of attempts made to implement the revision.")
    final_text: str = Field(..., description="The final text after all revisions and evaluations.")
    
    @field_validator('final_text')
    def validate_final_text(cls, v):
        """Validate and sanitize final text."""
        return sanitize_text(v)

class Module5Output(BaseModel):
    goal: str
    selected_criteria: list[SuccessCriteria]
    original_outline: PlanOutline  # Original high-level outline
    expanded_outline: PlanOutline  # Original expanded outline
    revision_results: List[RevisionImplementationResult]  # Results of applying revisions
    revised_outline: PlanOutline  # Final outline with applied revisions
    criteria_fulfillment_summary: Dict[str, Dict[str, int]]  # Summary of criteria fulfillment before and after

# --- Agents ---
apply_revision_agent = Agent(
    name="RevisionApplier",
    instructions=(
        "You are a revision implementation specialist. Given a goal, success criteria, "
        "a detailed item description, and a specific revision request, apply the requested "
        "revision to create a new, improved version of the text. "
        "Your revised text should integrate the requested changes while maintaining the "
        "coherence and quality of the original text. Be comprehensive and ensure the revision "
        "addresses the criteria mentioned in the revision request. "
        "Return the complete revised text, not just the changes or additions."
    ),
    model="gpt-4o",
    output_type=AppliedRevision,
    hooks=logging_hooks,
)

evaluate_implementation_agent = Agent(
    name="ImplementationEvaluator",
    instructions=(
        "You are an expert implementation evaluator. Given a goal, success criteria, "
        "an original text, a revision request, and the revised text, evaluate how well "
        "the revision implementation meets the targeted success criteria. "
        "Be thorough in your assessment, considering both the specific revision request "
        "and how well the revised text addresses each targeted criterion. "
        "If the implementation doesn't fully meet the criteria, provide specific suggestions "
        "for improvement. "
        "For each targeted criterion, indicate whether it is 'fully met', 'partially met', or 'not met' "
        "in your criteria_fulfillment field as a formatted string. "
        "Format it as one criterion per line, like: 'Criterion Name: fully met' or 'Criterion Name: partially met'."
    ),
    model="gpt-4o",
    output_type=RevisionImplementationEvaluation,
    hooks=logging_hooks,
)

refine_revision_agent = Agent(
    name="RevisionRefiner",
    instructions=(
        "You are a revision refinement specialist. Given a goal, success criteria, "
        "a revised text that didn't fully meet the criteria, and specific improvement suggestions, "
        "create a further refined version that better addresses the targeted criteria. "
        "Pay close attention to the improvement suggestions and ensure your refined text "
        "fully addresses them. Return the complete refined text, not just the changes."
    ),
    model="gpt-4o",
    output_type=AppliedRevision,
    hooks=logging_hooks,
)

async def validate_module5_output(
    context: RunContextWrapper[None], agent: Agent, agent_output: Any
) -> GuardrailFunctionOutput:
    """Validates the output of Module 5."""
    try:
        log_info("Validating Module 5 output...")
        # Create a truncated version for the standard log
        truncated_output = {
            "goal": agent_output.goal,
            "selected_criteria": [c.criteria for c in agent_output.selected_criteria],
            "revision_results": [
                {"item_title": r.item_title, "attempt_count": r.attempt_count}
                for r in agent_output.revision_results
            ],
            "criteria_fulfillment_summary": agent_output.criteria_fulfillment_summary
        }
        
        # Log to both
        truncated_json = json.dumps(truncated_output, indent=2)
        full_json = json.dumps(agent_output.model_dump(), indent=2)
        
        log_info(f"Output to validate (truncated): {truncated_json}", truncate=True, max_length=5000)
        verbose_logger.info(f"Full output to validate: {full_json}")
        
        Module5Output.model_validate(agent_output)
        log_info("Module 5 output validation passed")
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)
    except ValidationError as e:
        logger.error(f"Module 5 output validation failed: {e}")
        verbose_logger.error(f"Module 5 output validation failed: {e}")
        return GuardrailFunctionOutput(
            output_info={"error": str(e)}, tripwire_triggered=True
        )

async def apply_and_evaluate_revision(
    goal: str,
    selected_criteria: List[SuccessCriteria],
    item: PlanItem,
    revision_request: RevisionRequest,
    context: RunContextWrapper[None],
    max_attempts: int = 2
) -> RevisionImplementationResult:
    """Apply a revision and evaluate it, with multiple attempts if needed."""
    log_info(f"Starting to apply revision for: {item.item_title}")
    log_info(f"Revision request: {revision_request.revision_request_content}", truncate=True, max_length=5000)
    
    original_text = item.item_description
    current_text = original_text
    attempt_count = 0
    last_applied_revision = None
    last_evaluation = None
    
    # Get targeted criteria objects
    targeted_criteria_objects = []
    for criterion_name in revision_request.targeted_criteria:
        for criterion in selected_criteria:
            if criterion.criteria == criterion_name:
                targeted_criteria_objects.append(criterion)
                break
    
    targeted_criteria_json = json.dumps([c.model_dump() for c in targeted_criteria_objects], indent=2)
    
    for attempt in range(1, max_attempts + 1):
        attempt_count = attempt
        log_info(f"Attempt {attempt} to apply revision for {item.item_title}")
        
        # If not the first attempt, include previous feedback
        previous_feedback = ""
        if attempt > 1 and last_evaluation:
            previous_feedback = (
                f"Previous implementation feedback: {last_evaluation.reasoning}\n\n"
                f"Improvement suggestions: {last_evaluation.improvement_suggestions}\n\n"
            )
        
        # Create input for applying revision
        apply_input = (
            f"Goal: {goal}\n\n"
            f"Targeted Success Criteria: {targeted_criteria_json}\n\n"
            f"Item Title: {item.item_title}\n\n"
            f"Original Text: {current_text}\n\n"
            f"Revision Request: {revision_request.revision_request_content}\n\n"
            f"Revision Reasoning: {revision_request.reasoning}\n\n"
            f"{previous_feedback}"
            f"Please apply this revision to create a new, improved version of the text. "
            f"Return the complete revised text, not just the changes."
        )
        
        log_info(f"Apply revision input for attempt {attempt}", truncate=True, max_length=5000)
        verbose_logger.info(f"Full apply revision input for attempt {attempt}: {apply_input}")
        
        # Apply the revision
        if attempt == 1:
            apply_result = await Runner.run(
                apply_revision_agent,
                input=apply_input,
                context=context,
            )
        else:
            apply_result = await Runner.run(
                refine_revision_agent,
                input=apply_input,
                context=context,
            )
        
        applied_revision = apply_result.final_output
        last_applied_revision = applied_revision
        revised_text = applied_revision.revised_text
        
        log_info(f"Applied revision (attempt {attempt}) - length: {len(revised_text)} characters")
        log_info(f"Revised text begins with: {revised_text[:200]}...")
        verbose_logger.info(f"Full revised text (attempt {attempt}): {revised_text}")
        
        # Create input for evaluating implementation
        evaluation_input = (
            f"Goal: {goal}\n\n"
            f"Targeted Success Criteria: {targeted_criteria_json}\n\n"
            f"Item Title: {item.item_title}\n\n"
            f"Original Text: {original_text}\n\n"
            f"Revision Request: {revision_request.revision_request_content}\n\n"
            f"Revision Reasoning: {revision_request.reasoning}\n\n"
            f"Revised Text: {revised_text}\n\n"
            f"Please evaluate how well this revision implementation meets the targeted success criteria. "
            f"For each criterion, indicate whether it is 'fully met', 'partially met', or 'not met' as a string "
            f"in the criteria_fulfillment field. Format each criterion on a new line, "
            f"like: 'Criterion Name: fully met' or 'Criterion Name: partially met'."
        )
        
        log_info(f"Evaluation input for attempt {attempt}", truncate=True, max_length=5000)
        verbose_logger.info(f"Full evaluation input for attempt {attempt}: {evaluation_input}")
        
        # Evaluate the implementation
        evaluation_result = await Runner.run(
            evaluate_implementation_agent,
            input=evaluation_input,
            context=context,
        )
        
        last_evaluation = evaluation_result.final_output
        
        log_info(f"Implementation evaluation (attempt {attempt}): {last_evaluation.meets_criteria}")
        log_info(f"Criteria fulfillment: {last_evaluation.criteria_fulfillment}")
        
        # If the implementation meets the criteria or this is the last attempt, break
        if last_evaluation.meets_criteria or attempt == max_attempts:
            break
        
        # Otherwise, update the current text for the next attempt
        current_text = revised_text
        log_info(f"Implementation doesn't fully meet criteria. Proceeding to attempt {attempt+1}...")
    
    # Return the final result
    return RevisionImplementationResult(
        item_title=item.item_title,
        original_text=original_text,
        revision_request=revision_request,
        applied_revision=last_applied_revision,
        implementation_evaluation=last_evaluation,
        attempt_count=attempt_count,
        final_text=last_applied_revision.revised_text,
    )

def generate_criteria_fulfillment_summary(
    original_evaluations: List[EvalResult],
    revision_results: List[RevisionImplementationResult]
) -> Dict[str, Dict[str, int]]:
    """Generate a summary of criteria fulfillment before and after revisions."""
    # Create a dictionary of criterion name -> result from original evaluations
    original_results = {}
    for eval_result in original_evaluations:
        criterion = eval_result.criteria.criteria
        if criterion not in original_results:
            original_results[criterion] = {"pass": 0, "fail": 0}
        original_results[criterion][eval_result.result] += 1
    
    # Initialize the summary with the original results
    summary = {}
    for criterion, counts in original_results.items():
        summary[criterion] = {
            "original_pass": counts.get("pass", 0),
            "original_fail": counts.get("fail", 0),
            "fully_met_revisions": 0,
            "partially_met_revisions": 0,
            "not_met_revisions": 0
        }
    
    # Update with the revision results
    for result in revision_results:
        criteria_fulfillment = result.implementation_evaluation.criteria_fulfillment
        
        # Parse the criteria fulfillment string to extract status for each criterion
        lines = criteria_fulfillment.strip().split('\n')
        for line in lines:
            if ':' in line:
                parts = line.split(':', 1)
                criterion = parts[0].strip()
                status = parts[1].strip().lower()
                
                if criterion in summary:
                    if "fully" in status:
                        summary[criterion]["fully_met_revisions"] += 1
                    elif "partially" in status:
                        summary[criterion]["partially_met_revisions"] += 1
                    elif "not" in status:
                        summary[criterion]["not_met_revisions"] += 1
    
    return summary

async def run_module_5(input_file: str, output_file: str) -> None:
    """Runs Module 5."""
    context = RunContextWrapper(context=None)

    try:
        log_info(f"Starting Module 5, reading input from {input_file}")
        with open(input_file, "r") as f:
            module_4_data = json.load(f)
            log_info(f"Successfully loaded data from {input_file}")

        # Convert to Pydantic objects
        module_4_output = Module4Output.model_validate(module_4_data)
        goal = module_4_output.goal
        selected_criteria = module_4_output.selected_criteria
        original_outline = module_4_output.selected_outline
        expanded_outline = module_4_output.expanded_outline
        item_details = module_4_output.item_details
        evaluation_results = module_4_output.evaluation_results
        
        log_info(f"Goal: {goal}")
        log_info(f"Number of selected criteria: {len(selected_criteria)}")
        for i, criterion in enumerate(selected_criteria):
            log_info(f"Criterion {i+1}: {criterion.criteria}")
        
        # Filter item_details to only include items with approved revisions
        items_to_revise = []
        for item_detail in item_details:
            if (item_detail.revision_request and
                item_detail.revision_evaluation and 
                item_detail.revision_evaluation.approved):
                # Find the corresponding PlanItem
                for item in expanded_outline.plan_items:
                    if item.item_title == item_detail.item_title:
                        items_to_revise.append((item, item_detail.revision_request))
                        break
        
        log_info(f"Found {len(items_to_revise)} items with approved revisions to apply")
        
        # Apply and evaluate revisions
        revision_results = []
        for item, revision_request in items_to_revise:
            result = await apply_and_evaluate_revision(
                goal=goal,
                selected_criteria=selected_criteria,
                item=item,
                revision_request=revision_request,
                context=context
            )
            revision_results.append(result)
            log_info(f"Completed revision for {item.item_title} after {result.attempt_count} attempts")
        
        # Create a new plan outline with the revised items
        revised_items = []
        for item in expanded_outline.plan_items:
            # Check if this item has a revision result
            revised_item = None
            for result in revision_results:
                if result.item_title == item.item_title:
                    revised_item = PlanItem(
                        item_title=item.item_title,
                        item_description=result.final_text
                    )
                    break
            
            # If no revision, keep the original
            if revised_item is None:
                revised_item = item
            
            revised_items.append(revised_item)
        
        revised_outline = PlanOutline(
            plan_title=expanded_outline.plan_title,
            plan_description=expanded_outline.plan_description,
            plan_items=revised_items,
            reasoning=expanded_outline.reasoning,
            rating=expanded_outline.rating,
            created_by=expanded_outline.created_by
        )
        
        # Generate criteria fulfillment summary
        criteria_fulfillment_summary = generate_criteria_fulfillment_summary(
            evaluation_results, revision_results
        )
        log_info(f"Criteria fulfillment summary: {json.dumps(criteria_fulfillment_summary, indent=2)}")
        
        # Create the output object
        log_info("Creating Module 5 output object")
        module_5_output = Module5Output(
            goal=goal,
            selected_criteria=selected_criteria,
            original_outline=original_outline,
            expanded_outline=expanded_outline,
            revision_results=revision_results,
            revised_outline=revised_outline,
            criteria_fulfillment_summary=criteria_fulfillment_summary,
        )

        # Apply guardrail
        log_info("Applying output guardrail...")
        guardrail = OutputGuardrail(guardrail_function=validate_module5_output)
        guardrail_result = await guardrail.run(
            agent=evaluate_implementation_agent,
            agent_output=module_5_output,
            context=context,
        )
        
        if guardrail_result.output.tripwire_triggered:
            logger.error(f"Guardrail failed: {guardrail_result.output.output_info}")
            verbose_logger.error(f"Guardrail failed: {guardrail_result.output.output_info}")
            return  # Exit if validation fails

        # --- Smart JSON Export ---
        # Create data directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamped version
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.basename(output_file)
        name, ext = os.path.splitext(filename)
        timestamped_file = os.path.join(output_dir, f"{name}_{timestamp}{ext}")
        
        # Export both versions
        output_json = json.dumps(module_5_output.model_dump(), indent=4)
        with open(output_file, "w") as f:
            f.write(output_json)
        with open(timestamped_file, "w") as f:
            f.write(output_json)
        
        log_info(f"Module 5 completed. Output saved to {output_file}")
        log_info(f"Timestamped output saved to {timestamped_file}")

    except Exception as e:
        logger.error(f"An error occurred in Module 5: {e}")
        verbose_logger.error(f"An error occurred in Module 5: {e}")
        import traceback
        error_trace = traceback.format_exc()
        logger.error(error_trace)
        verbose_logger.error(error_trace)  # Log the full stack trace

async def main():
    log_info("Starting main function")
    input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    input_file = os.path.join(input_dir, "module4_output.json")
    output_file = os.path.join(input_dir, "module5_output.json")
    await run_module_5(input_file, output_file)
    log_info("Main function completed")

if __name__ == "__main__":
    log_info("Module 5 script starting")
    asyncio.run(main())
    log_info("Module 5 script completed")