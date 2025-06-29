# python module1-opentelemetry-gm-1156.py

# uv venv
# source .venv/bin/activate
# uv pip install openai-agents
# uv pip install python-dotenv
# uv pip install opentelemetry-sdk opentelemetry-exporter-otlp opentelemetry-instrumentation
# python module1.py  # Input your goal or idea and get success criteria
# python module2.py  # Creates and selects a plan
# python module3.py  # Expands and evaluates the plan
# python module3-claude.py  # Better expands and evaluates the plan
# python module3-gem.py  # Expands and evaluates the plan
# python module4.py  # Identifies needed revisions
# python module5.py  # Implements revisions into a final plan
# python module6.py  # Generate easy to read markdown of final plan

# I need a plan for a workshop on legal aspects of AI agents that conduct transactions to purchase items for users.  It needs to look at 1) the contractual and general liability aspects for users, agent providers, and third party merchants that actually conduct the transactions with the user's AI agents, and 2) fiduciary duties for providers of the agents to the users who are deemed principals. The plan needs a simple scenario for the user/principal, AI agent provider (who is also the legal agent of the user/principal), and the third party merchants so participants can brainstorm the types of contractual provisions the users, agent providers, and third parties would all seek to have in place for these transactions.  The fiduciary relationship between the agent provider and the user will require corresponding contractual provisions.

import asyncio
import json
import os
import logging
import datetime
import re
from typing import Any, List, Dict, Optional
import atexit

# Import OpenAI libraries
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator

from agents import (
    Agent,
    GuardrailFunctionOutput,
    OutputGuardrail,
    Runner,
    WebSearchTool,
    trace,
    add_trace_processor,
    RunConfig
)
from agents.run_context import RunContextWrapper
from agents.lifecycle import AgentHooks
from agents.tracing.processor_interface import TracingProcessor
from agents.tracing.spans import Span
from agents.tracing.traces import Trace
from agents.tracing import span_data
from agents.tracing import set_tracing_export_api_key

# Add OpenTelemetry imports
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from contextlib import contextmanager
import uuid
import copy

load_dotenv()  # Load environment variables

# Set tracing API key right after loading environment variables
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    set_tracing_export_api_key(api_key)
else:
    print("WARNING: OPENAI_API_KEY not found in environment variables")

# --- Setup Logging (Modified for Verbosity) ---
def setup_logging(module_name):
    """Set up logging to console, a standard file, and a verbose file."""
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(logs_dir, f"{module_name}_{timestamp}.log")
    verbose_log_file = os.path.join(logs_dir, f"{module_name}_verbose_{timestamp}.log")

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

    # Verbose logger (no truncation)
    verbose_logger = logging.getLogger(f"{module_name}_verbose")
    verbose_logger.setLevel(logging.INFO)
    if verbose_logger.handlers:
        verbose_logger.handlers = []
    verbose_file_handler = logging.FileHandler(verbose_log_file)
    verbose_file_handler.setLevel(logging.INFO)
    verbose_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    verbose_file_handler.setFormatter(verbose_format)
    verbose_logger.addHandler(verbose_file_handler)

    return logger, verbose_logger

# Initialize loggers
logger, verbose_logger = setup_logging("module1")

# Helper function to log to both loggers
def log_info(message, truncate=False, max_length=5000):
    verbose_logger.info(message)  # Always log full message to verbose
    if truncate:
        if len(message) > max_length:
            message = message[:max_length] + "... [truncated, see verbose log]"
        logger.info(message)
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

    # Ensure the text doesn't exceed a reasonable size (50KB) - adjust as necessary
    max_length = 50000
    if len(text) > max_length:
        text = text[:max_length] + "...[text truncated due to length]"

    return text

# --- Pydantic Models --- (No changes)
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
    selected_criteria: list[SuccessCriteria]  # Changed to a list for multiple criteria

    @field_validator('selected_criteria')
    def validate_selected_criteria(cls, v):
        if not v:
            raise ValueError("At least one criterion must be selected")
        return v
        
# Global dictionary to store manual trace data
MANUAL_TRACES = {
    "traces": [],
    "current_trace": None
}

def capture_step(stage, inputs, outputs, trace_id=None):
    """Explicitly capture workflow data at each significant point."""
    global MANUAL_TRACES
    
    # Create a new trace if needed
    if MANUAL_TRACES["current_trace"] is None or trace_id != MANUAL_TRACES["current_trace"].get("id"):
        if trace_id is None:
            trace_id = f"manual_{uuid.uuid4().hex}"
            
        MANUAL_TRACES["current_trace"] = {
            "id": trace_id,
            "steps": [],
            "timestamp_start": datetime.datetime.now().isoformat()
        }
        MANUAL_TRACES["traces"].append(MANUAL_TRACES["current_trace"])
    
    # Add the step data
    step_data = {
        "stage": stage,
        "timestamp": datetime.datetime.now().isoformat(),
        "inputs": copy.deepcopy(inputs) if inputs else None,
        "outputs": copy.deepcopy(outputs) if outputs else None
    }
    
    MANUAL_TRACES["current_trace"]["steps"].append(step_data)
    
    # Save to file
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    trace_file = os.path.join(logs_dir, "manual_traces.json")
    
    try:
        with open(trace_file, "w") as f:
            json.dump(MANUAL_TRACES, f, indent=4, default=str)
    except Exception as e:
        print(f"Error saving manual trace: {e}")

# --- OpenTelemetry Setup ---
def setup_opentelemetry(service_name="OpenAI-Agents-Tracing"):
    """
    Setup OpenTelemetry tracing with file and console exporters.
    Returns a tracer that can be used throughout the application.
    """
    # Create logs directory for trace output files
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a timestamped filename for this run
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    trace_file = os.path.join(logs_dir, f"otel_traces_{timestamp}.json")
    
    # Create a resource to identify this service
    resource = Resource(attributes={
        SERVICE_NAME: service_name
    })
    
    # Create a tracer provider with the resource
    tracer_provider = TracerProvider(resource=resource)
    
    # Set the global tracer provider
    otel_trace.set_tracer_provider(tracer_provider)
    
    # Create an enhanced JSON file exporter
    class EnhancedJSONFileExporter:
        def __init__(self, filepath):
            self.filepath = filepath
            self.spans = []
            self.traces = {}  # Organize spans by trace_id
            
        def export(self, spans):
            for span in spans:
                # Get the trace ID
                trace_id = format(span.context.trace_id, '032x')
                
                # Convert span to JSON-serializable format
                span_data = {
                    "name": span.name,
                    "context": {
                        "trace_id": trace_id,
                        "span_id": format(span.context.span_id, '016x'),
                    },
                    "parent_id": format(span.parent.span_id, '016x') if span.parent else None,
                    "start_time": span.start_time,
                    "end_time": span.end_time,
                    "duration_ms": (span.end_time - span.start_time) / 1000000,  # Convert to milliseconds
                    "status": {
                        "status_code": span.status.status_code.name,
                        "description": span.status.description
                    },
                    "attributes": dict(span.attributes),
                    "events": [
                        {
                            "name": event.name,
                            "timestamp": event.timestamp,
                            "attributes": dict(event.attributes)
                        } for event in span.events
                    ]
                }
                
                # Add to spans list
                self.spans.append(span_data)
                
                # Organize by trace
                if trace_id not in self.traces:
                    self.traces[trace_id] = {
                        "trace_id": trace_id,
                        "spans": [],
                        "start_time": span.start_time,  # Will be updated if earlier spans are added
                        "end_time": span.end_time,      # Will be updated if later spans are added
                    }
                
                # Update trace timing if needed
                if span.start_time < self.traces[trace_id]["start_time"]:
                    self.traces[trace_id]["start_time"] = span.start_time
                if span.end_time > self.traces[trace_id]["end_time"]:
                    self.traces[trace_id]["end_time"] = span.end_time
                    
                # Add span to its trace
                self.traces[trace_id]["spans"].append(span_data)
                
                # Calculate trace duration
                self.traces[trace_id]["duration_ms"] = (
                    self.traces[trace_id]["end_time"] - self.traces[trace_id]["start_time"]
                ) / 1000000  # Convert to milliseconds
                    
                # Write to file with each span for safety in two formats
                try:
                    # Format 1: All spans in a flat list
                    with open(self.filepath, 'w') as f:
                        json.dump({"spans": self.spans}, f, indent=4, default=str)
                        
                    # Format 2: Spans organized by trace (more readable)
                    trace_filepath = self.filepath.replace(".json", "_by_trace.json")
                    with open(trace_filepath, 'w') as f:
                        json.dump({"traces": list(self.traces.values())}, f, indent=4, default=str)
                except Exception as e:
                    print(f"Error writing span to file: {e}")
                    
            return True
            
        def shutdown(self):
            # Final write - in case any spans were missed
            try:
                with open(self.filepath, 'w') as f:
                    json.dump({"spans": self.spans}, f, indent=4, default=str)
                    
                trace_filepath = self.filepath.replace(".json", "_by_trace.json")
                with open(trace_filepath, 'w') as f:
                    json.dump({"traces": list(self.traces.values())}, f, indent=4, default=str)
            except Exception as e:
                print(f"Error during shutdown: {e}")
    
    # Create and register exporters
    json_exporter = EnhancedJSONFileExporter(trace_file)
    console_exporter = ConsoleSpanExporter()  # Prints to stdout for debugging
    
    # Add the exporters to the tracer provider
    tracer_provider.add_span_processor(SimpleSpanProcessor(json_exporter))
    tracer_provider.add_span_processor(SimpleSpanProcessor(console_exporter))
    
    # Get a tracer from the provider
    tracer = otel_trace.get_tracer(__name__)
    
    log_info(f"OpenTelemetry initialized with trace file: {trace_file}")
    
    return tracer

# Context manager for creating rich spans with attributes
@contextmanager
def traced_span(tracer, name, attributes=None, record_exception=True):
    """Context manager to create spans with rich attributes."""
    attributes = attributes or {}
    with tracer.start_as_current_span(name) as span:
        # Set attributes on the span
        for key, value in attributes.items():
            if isinstance(value, (dict, list)):
                # Convert complex objects to strings to ensure they can be serialized
                span.set_attribute(key, json.dumps(value))
            else:
                span.set_attribute(key, str(value))
        
        try:
            yield span
        except Exception as e:
            if record_exception:
                # Record the exception in the span
                span.record_exception(e)
                span.set_status(otel_trace.StatusCode.ERROR, str(e))
            raise
        
class EnhancedFileTracingProcessor(TracingProcessor):
    """Custom processor to extract maximum content from traces and spans."""

    def __init__(self, filename: str = "traces.json"):
        logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        self.filename = os.path.join(logs_dir, filename)
        
        # Create a timestamped version
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name, ext = os.path.splitext(filename)
        self.timestamped_filename = os.path.join(logs_dir, f"{name}_{timestamp}{ext}")
        
        # Data structures
        self.traces = {}  # Dict to store traces by trace_id
        self.spans = {}   # Dict to store spans by trace_id -> [spans]
        self.raw_content = {}  # Store raw content by span_id
        
        log_info(f"Enhanced FileTracingProcessor initialized with files: {self.filename} and {self.timestamped_filename}")

    def on_trace_start(self, trace: Trace) -> None:
        """Store trace when it starts."""
        trace_id = trace.trace_id
        self.traces[trace_id] = trace
        self.spans.setdefault(trace_id, [])
        log_info(f"Trace started: {trace_id}", truncate=True)

    def on_trace_end(self, trace: Trace) -> None:
        """Save completed trace and its spans."""
        trace_id = trace.trace_id
        self.traces[trace_id] = trace
        log_info(f"Trace ended: {trace_id}", truncate=True)
        self._save_to_file()

    def on_span_start(self, span: Span[Any]) -> None:
        """Record span start."""
        trace_id = span.trace_id
        if trace_id not in self.spans:
            self.spans[trace_id] = []
        self._capture_span_content(span)

    def on_span_end(self, span: Span[Any]) -> None:
        """Store finished span with all details."""
        trace_id = span.trace_id
        self._capture_span_content(span)
        if trace_id in self.spans:
            self.spans[trace_id].append(span)

    def _capture_span_content(self, span: Span[Any]) -> None:
        """Extract maximum content from span using multiple approaches."""
        span_id = span.span_id
        span_data = {}

        # 1. Start with basic export
        exported = span.export()
        if exported:
            span_data.update(exported)

        # 2. Try to get the input data through direct attribute access
        if hasattr(span, "_input"):
            span_data["input"] = span._input
        elif hasattr(span, "input"):
            span_data["input"] = span.input

        # 3. Try to get output data – FOCUS ON THIS
        if hasattr(span, "_data") and hasattr(span._data, "response") and hasattr(span._data.response, "output"):
            # This seems to be the key location based on trace analysis
            span_data["output"] = span._data.response.output
        elif hasattr(span, "output"): # Fallback
            span_data["output"] = span.output
            
        # 4. Try to extract content from span_data based on type
        if hasattr(span, "span_data") and span.span_data:
            data_obj = span.span_data
            
            # Get type info
            span_type = None
            if hasattr(data_obj, "type"):
                span_type = data_obj.type
            elif "span_data" in span_data and "type" in span_data["span_data"]:
                span_type = span_data["span_data"]["type"]
                
            # Add all public attributes from span_data
            for attr_name in dir(data_obj):
                if not attr_name.startswith('_'):  # Public attributes only
                    try:
                        attr_value = getattr(data_obj, attr_name)
                        if not callable(attr_value):  # Skip methods
                            span_data[f"span_data_{attr_name}"] = attr_value
                    except Exception:
                        pass
                        
            # Type-specific extraction
            if span_type == "agent" or hasattr(data_obj, "agent_name") or hasattr(data_obj, "name"):
                for field in ["name", "agent_name", "instructions", "model", "tools", "handoffs"]:
                    if hasattr(data_obj, field):
                        span_data[field] = getattr(data_obj, field)
                        
            elif span_type == "response" or hasattr(data_obj, "response_id"):
                for field in ["response_id", "content", "response", "request"]:
                    if hasattr(data_obj, field):
                        span_data[field] = getattr(data_obj, field)
                        
            elif span_type == "function" or hasattr(data_obj, "function_name"):
                for field in ["function_name", "input", "output", "arguments", "result"]:
                    if hasattr(data_obj, field):
                        span_data[field] = getattr(data_obj, field)
                        
            elif span_type == "generation" or hasattr(data_obj, "prompt"):
                for field in ["prompt", "response", "input_messages", "output", "model"]:
                    if hasattr(data_obj, field):
                        span_data[field] = getattr(data_obj, field)
        
        # 5. Try to access span._data
        if hasattr(span, "_data"):
            try:
                for attr_name in dir(span._data):
                    if not attr_name.startswith('_'):
                        try:
                            attr_value = getattr(span._data, attr_name)
                            if not callable(attr_value):
                                span_data[f"_data_{attr_name}"] = attr_value
                        except Exception:
                            pass
            except Exception:
                pass
                
        # 6. Try using vars() for a complete dump (might fail but worth trying)
        if hasattr(span, "span_data") and span.span_data:
            try:
                all_vars = vars(span.span_data)
                if all_vars:
                    span_data["span_data_vars"] = all_vars
            except:
                pass
                
        # 7. Check for inner span data attributes that might contain content
        if hasattr(span, "span_data") and span.span_data:
            for inner_attr in ["data", "attributes", "content", "message", "messages"]:
                if hasattr(span.span_data, inner_attr):
                    inner_value = getattr(span.span_data, inner_attr)
                    span_data[f"inner_{inner_attr}"] = inner_value

        # Store the captured data
        self.raw_content[span_id] = span_data

    def _save_to_file(self) -> None:
        """Save all completed traces and spans to file with full content."""
        try:
            # Prepare the data structure
            data = {"traces": []}
            
            for trace_id, trace in self.traces.items():
                trace_data = trace.export() or {}
                if not trace_data:
                    continue
                    
                # Add spans with enhanced content
                trace_data["spans"] = []
                for span in self.spans.get(trace_id, []):
                    span_id = span.span_id
                    span_data = self.raw_content.get(span_id, {})
                    
                    if not span_data:
                        span_data = span.export() or {}
                        
                    if span_data:
                        trace_data["spans"].append(span_data)
                
                data["traces"].append(trace_data)
            
            # Add metadata
            data["timestamp"] = datetime.datetime.now().isoformat()
            data["count"] = len(data["traces"])
            
            # Save to both files
            for filename in [self.filename, self.timestamped_filename]:
                with open(filename, "w") as f:
                    json.dump(data, f, indent=4, default=str)
                
            log_info(f"Saved {len(data['traces'])} traces with enhanced content", truncate=True)
            
        except Exception as e:
            logger.error(f"Error saving traces to file: {e}")
            verbose_logger.error(f"Error saving traces to file: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def shutdown(self) -> None:
        """Save final state when shutting down."""
        log_info("Shutting down EnhancedFileTracingProcessor, saving final traces", truncate=True)
        self._save_to_file()

    def force_flush(self) -> None:
        """Force save current state."""
        self._save_to_file()
        

# --- Custom Agent Hooks for Detailed Logging --- (Modified for verbosity)
class DetailedLoggingHooks(AgentHooks):
    def __init__(self, logger, verbose_logger, tracer=None, process_id=None):
        self.logger = logger
        self.verbose_logger = verbose_logger
        self.tracer = tracer  # OpenTelemetry tracer
        self.process_id = process_id  # For manual tracing

    async def on_start(
        self, context: RunContextWrapper[Any], agent: Agent
    ):
        """Called before the agent is invoked."""
        log_info(f"===== API CALL: {agent.name} =====", truncate=True)
        log_info(f"Agent start: {agent.name}", truncate=True)
        self.verbose_logger.info(f"===== API CALL: {agent.name} =====")
        
        # Manual capture if process_id is available
        if self.process_id:
            agent_info = {
                "name": agent.name,
                "instructions": agent.instructions,
                "model": agent.model if hasattr(agent, "model") else "unknown",
                "tools": str(agent.tools) if hasattr(agent, "tools") else "none"
            }
            capture_step(f"agent_start_{agent.name}",
                       agent_info,
                       None,
                       self.process_id)
        
        # OpenTelemetry capture
        if self.tracer:
            current_span = otel_trace.get_current_span()
            if current_span:
                current_span.add_event(
                    "Agent Start",
                    {
                        "agent_name": agent.name,
                        "agent_instructions": agent.instructions if hasattr(agent, "instructions") else "none"
                    }
                )
        return

    async def on_end(
        self, context: RunContextWrapper[Any], agent: Agent, output: Any
    ):
        """Called when the agent produces a final output."""
        log_info(f"===== API RESPONSE: {agent.name} =====", truncate=True)
        self.verbose_logger.info(f"===== API RESPONSE: {agent.name} =====")

        try:
            # Sanitize output if needed
            if hasattr(output, 'final_output'):
                if isinstance(output.final_output, str):
                    output.final_output = sanitize_text(output.final_output)
                elif isinstance(output.final_output, list):
                    for item in output.final_output:
                        if hasattr(item, "criteria"):
                            item.criteria = sanitize_text(item.criteria)
                        if hasattr(item, "reasoning"):
                            item.reasoning = sanitize_text(item.reasoning)

                # Format output
                response_content = json.dumps(output.final_output, indent=2) if hasattr(output, 'final_output') else str(output)
                log_info(f"Response from {agent.name}: {response_content}", truncate=True)
                self.verbose_logger.info(f"Response from {agent.name}: {response_content}")
                
                # Manual capture
                if self.process_id:
                    capture_step(f"agent_response_{agent.name}",
                               {"agent_name": agent.name},
                               {"response": output.final_output},
                               self.process_id)
                
                # OpenTelemetry capture
                if self.tracer:
                    current_span = otel_trace.get_current_span()
                    if current_span:
                        current_span.add_event(
                            "Agent Response",
                            {
                                "agent_name": agent.name,
                                "response": response_content[:5000]
                            }
                        )
            else:
                log_info(f"Response from {agent.name}: {str(output)}", truncate=True)
                self.verbose_logger.info(f"Response from {agent.name}: {str(output)}")
                
                # Manual capture
                if self.process_id:
                    capture_step(f"agent_response_{agent.name}",
                               {"agent_name": agent.name},
                               {"response": str(output)},
                               self.process_id)
                
                # OpenTelemetry capture
                if self.tracer:
                    current_span = otel_trace.get_current_span()
                    if current_span:
                        current_span.add_event(
                            "Agent Response",
                            {
                                "agent_name": agent.name,
                                "response": str(output)[:5000]
                            }
                        )
        except Exception as e:
            log_info(f"Response from {agent.name}: {str(output)}", truncate=True)
            log_info(f"Could not format response as JSON: {e}", truncate=True)
            self.verbose_logger.info(f"Response from {agent.name}: {str(output)}")
            self.verbose_logger.info(f"Could not format response as JSON: {e}")
            
            # Manual capture of error
            if self.process_id:
                capture_step(f"agent_response_error_{agent.name}",
                           {"agent_name": agent.name},
                           {"error": str(e), "raw_output": str(output)},
                           self.process_id)
        return output

    async def on_tool_start(
        self, context: RunContextWrapper[Any], agent: Agent, tool: Any
    ):
        """Called before a tool is invoked."""
        log_info(f"===== TOOL CALL: {agent.name} =====", truncate=True)
        self.verbose_logger.info(f"===== TOOL CALL: {agent.name} =====")
        
        # Get tool information
        tool_name = getattr(tool, "name", str(tool))
        tool_input = None
        if hasattr(tool, "_input"):
            tool_input = str(tool._input)
        
        # Manual capture
        if self.process_id:
            capture_step(f"tool_call_{agent.name}_{tool_name}",
                       {"agent_name": agent.name, "tool_name": tool_name},
                       {"tool_input": tool_input},
                       self.process_id)
        
        # OpenTelemetry capture
        if self.tracer:
            current_span = otel_trace.get_current_span()
            if current_span:
                current_span.add_event(
                    "Tool Call",
                    {
                        "agent_name": agent.name,
                        "tool_name": tool_name,
                        "tool_input": tool_input or "Unknown"
                    }
                )
        return

    async def on_tool_end(
        self, context: RunContextWrapper[Any], agent: Agent, tool: Any, result: str
    ):
        """Called after a tool is invoked."""
        tool_name = getattr(tool, "name", str(tool))
        
        try:
            response_content = json.dumps(result, indent=2)
            log_info(f"Tool Result from {agent.name}: {response_content}", truncate=True)
            self.verbose_logger.info(f"Tool Result from {agent.name}: {response_content}")
            
            # Manual capture
            if self.process_id:
                capture_step(f"tool_result_{agent.name}_{tool_name}",
                           {"agent_name": agent.name, "tool_name": tool_name},
                           {"result": result},
                           self.process_id)
            
            # OpenTelemetry capture
            if self.tracer:
                current_span = otel_trace.get_current_span()
                if current_span:
                    current_span.add_event(
                        "Tool Result",
                        {
                            "agent_name": agent.name,
                            "tool_name": tool_name,
                            "result": response_content[:5000]
                        }
                    )
        except Exception as e:
            log_info(f"Tool Result from {agent.name}: {str(result)}", truncate=True)
            self.verbose_logger.info(f"Tool Result from {agent.name}: {str(result)}")
            log_info(f"Could not format response as JSON: {e}", truncate=True)
            self.verbose_logger.info(f"Could not format response as JSON: {e}")
            
            # Manual capture of error
            if self.process_id:
                capture_step(f"tool_result_error_{agent.name}_{tool_name}",
                           {"agent_name": agent.name, "tool_name": tool_name},
                           {"error": str(e), "raw_result": str(result)},
                           self.process_id)
            
            # OpenTelemetry capture
            if self.tracer:
                current_span = otel_trace.get_current_span()
                if current_span:
                    current_span.add_event(
                        "Tool Result Error",
                        {
                            "agent_name": agent.name,
                            "tool_name": tool_name,
                            "raw_result": str(result)[:5000],
                            "error": str(e)
                        }
                    )

        return result

# --- Search Agent ---
web_search_tool = WebSearchTool()  # Instantiate the tool

search_agent = Agent(
    name="SearchAgent",
    instructions=(
        "You are a web search assistant. Given a user's goal, "
        "perform a web search to find information relevant to defining success criteria. "
        "Return a concise summary of your findings, including citations to sources."
    ),
    model="gpt-4o",
    tools=[web_search_tool],  # Pass the *instance* of the tool
    # hooks=logging_hooks,  # We'll set this in main()
)

# --- Other Agents ---
generate_criteria_agent = Agent(
    name="CriteriaGenerator",
    instructions=(
        "You are a helpful assistant. Given a user's goal or idea, and the results of a web search,"
        "generate five distinct and measurable success criteria. "
        "Provide a brief reasoning for each criterion. "
        "Rate each criterion on a scale of 1-10 based on how strongly it indicates goal achievement."
    ),
    model="gpt-4o",
    output_type=list[SuccessCriteria],
    # hooks=logging_hooks,  # We'll set this in main()
)

evaluate_criteria_agent = Agent(
    name="CriteriaEvaluator",
    instructions=(
        "You are an expert evaluator. Given a goal/idea, search results, and a list of "
        "potential success criteria, select the THREE criteria that, if met together, "
        "would most strongly indicate that the goal has been achieved. "
        "Choose criteria that complement each other and cover different aspects of the goal. "
        "Consider information found by search to assist with your selection. "
        "Provide detailed reasoning for each of your selections."
    ),
    model="gpt-4o",
    output_type=list[SuccessCriteria],  # Changed to expect a list
    # hooks=logging_hooks, # We'll set this in main()
)

async def validate_module1_output(
    context: RunContextWrapper[None], agent: Agent, agent_output: Any
) -> GuardrailFunctionOutput:
    """Validates the output of Module 1."""
    try:
        log_info("Validating Module 1 output...", truncate=True)
        verbose_logger.info("Validating Module 1 output...")

        # Log only key parts for the standard log
        truncated_output = {
            "goal": agent_output.goal,
            "selected_criteria_count": len(agent_output.selected_criteria),
        }

        log_info(f"Output to validate (truncated): {json.dumps(truncated_output, indent=2)}", truncate=True)
        verbose_logger.info(f"Output to validate: {json.dumps(agent_output.model_dump() if hasattr(agent_output, 'model_dump') else agent_output, indent=2)}")

        Module1Output.model_validate(agent_output)
        log_info("Module 1 output validation passed", truncate=True)
        verbose_logger.info("Module 1 output validation passed")
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)
    except ValidationError as e:
        logger.error(f"Module 1 output validation failed: {e}")
        verbose_logger.error(f"Module 1 output validation failed: {e}")
        return GuardrailFunctionOutput(
            output_info={"error": str(e)}, tripwire_triggered=True
        )
        
async def run_module_1(user_goal: str, output_file: str, tracer, process_id: str) -> None:
    """Runs Module 1 with comprehensive tracing."""
    # Start OpenTelemetry span for the module
    with traced_span(tracer, "run_module_1", {"user_goal": user_goal}) as module_span:
        context = RunContextWrapper(context=None)
        openai_trace_id = None

        try:
            log_info(f"Starting Module 1 with goal: {user_goal}", truncate=True)
            verbose_logger.info(f"Starting Module 1 with goal: {user_goal}")
            
            # Manual capture for module start
            capture_step("module_start", {"user_goal": user_goal}, None, process_id)

            # --- Wrap the entire module execution in a trace ---
            with trace("Module 1 Trace") as current_trace:
                # Get the OpenAI trace ID if available
                if hasattr(current_trace, "trace_id"):
                    openai_trace_id = current_trace.trace_id
                    module_span.set_attribute("openai_trace_id", openai_trace_id)
                    # Link manual trace to OpenAI trace
                    capture_step("trace_linked", {"openai_trace_id": openai_trace_id}, None, process_id)
                
                # --- Run Search Agent ---
                with traced_span(tracer, "search_agent", {"agent": "SearchAgent"}) as search_span:
                    log_info("Running Search Agent...", truncate=True)
                    verbose_logger.info("Running Search Agent...")
                    
                    # Manual capture before search
                    search_input = f"Find information about success criteria for: {user_goal}"
                    capture_step("search_start", {"input": search_input}, None, process_id)

                    # Create RunConfig with trace_include_sensitive_data=True
                    run_config = RunConfig(trace_include_sensitive_data=True)

                    try:
                        search_result = await Runner.run(
                            search_agent,
                            input=search_input,
                            context=context,
                            run_config=run_config,  # Include sensitive data
                        )
                        search_summary = search_result.final_output
                        
                        # Manual capture after search
                        capture_step("search_complete", {"input": search_input}, {"search_summary": search_summary}, process_id)
                        
                        # OpenTelemetry capture
                        search_span.set_attribute("search_success", "true")
                        search_span.set_attribute("summary_length", str(len(search_summary)) if search_summary else "0")
                        if search_summary:
                            search_span.set_attribute("summary_excerpt", search_summary[:1000] if len(search_summary) > 1000 else search_summary)
                        
                        log_info(f"Search Agent returned (truncated): {search_summary[:200]}...", truncate=True)
                        verbose_logger.info(f"Search Agent returned (full): {search_summary}")

                    except Exception as e:
                        logger.warning(f"Search Agent failed: {e}. Proceeding without search results.")
                        verbose_logger.warning(f"Search Agent failed: {e}. Proceeding without search results.")
                        search_summary = "No search results available."  # Fallback message
                        
                        # Manual capture of failure
                        capture_step("search_failed", {"input": search_input}, {"error": str(e)}, process_id)
                        
                        # Record in OpenTelemetry
                        search_span.record_exception(e)
                        search_span.set_status(otel_trace.StatusCode.ERROR, str(e))
                        search_span.set_attribute("search_success", "false")

                # --- Generate criteria (with search results) ---
                with traced_span(tracer, "criteria_generation", {"agent": "CriteriaGenerator"}) as criteria_span:
                    log_info("GENERATING CANDIDATE SUCCESS CRITERIA...", truncate=True)
                    verbose_logger.info("GENERATING CANDIDATE SUCCESS CRITERIA...")
                    
                    # Prepare the input
                    criteria_input = f"The user's goal is: {user_goal}\n\nSearch Results:\n{search_summary}"
                    
                    # Manual capture
                    capture_step("criteria_generation_start", {"input": criteria_input}, None, process_id)
                    
                    # OpenTelemetry capture
                    criteria_span.set_attribute("input", criteria_input)

                    # Run with trace_include_sensitive_data=True
                    criteria_result = await Runner.run(
                        generate_criteria_agent,
                        input=criteria_input,
                        context=context,
                        run_config=run_config,  # Include sensitive data
                    )
                    generated_criteria = criteria_result.final_output
                    
                    # Manual capture of results
                    capture_step("criteria_generation_complete", 
                                {"input": criteria_input}, 
                                {"generated_criteria": [c.model_dump() for c in generated_criteria]}, 
                                process_id)
                    
                    # OpenTelemetry capture
                    criteria_span.set_attribute("criteria_count", str(len(generated_criteria)))
                    criteria_summary = [{"criteria": c.criteria, "rating": c.rating} for c in generated_criteria]
                    criteria_span.set_attribute("criteria_summary", json.dumps(criteria_summary))
                    
                    log_info(f"Generated {len(generated_criteria)} success criteria", truncate=True)
                    verbose_logger.info(f"Generated {len(generated_criteria)} success criteria")

                    # Log each criterion
                    for i, criterion in enumerate(generated_criteria, 1):
                        log_info(f"Criterion {i}: {criterion.criteria} (Rating: {criterion.rating})", truncate=True)
                        verbose_logger.info(f"Criterion {i}: {criterion.criteria} (Rating: {criterion.rating})")

                # Select top criteria
                with traced_span(tracer, "criteria_evaluation", {"agent": "CriteriaEvaluator"}) as eval_span:
                    log_info("EVALUATING AND SELECTING TOP CRITERIA...", truncate=True)
                    verbose_logger.info("EVALUATING AND SELECTING TOP CRITERIA...")

                    # Format criteria for the evaluator
                    criteria_json = json.dumps([c.model_dump() for c in generated_criteria], indent=2)
                    evaluation_input = (
                        f"Goal: {user_goal}\n\nSearch Results:\n{search_summary}\n\nCriteria:\n{criteria_json}"
                    )
                    
                    # Manual capture
                    capture_step("criteria_evaluation_start", {"input": evaluation_input}, None, process_id)
                    
                    # OpenTelemetry capture
                    eval_span.set_attribute("input_length", str(len(evaluation_input)))
                    
                    log_info(f"Evaluation input (truncated): {evaluation_input[:500]}...", truncate=True)
                    verbose_logger.info(f"Evaluation input (full): {evaluation_input}")

                    # Run evaluation
                    evaluation_result = await Runner.run(
                        evaluate_criteria_agent,
                        input=evaluation_input,
                        context=context,
                        run_config=run_config,  # Include sensitive data
                    )
                    selected_criteria = evaluation_result.final_output
                    
                    # Manual capture
                    capture_step("criteria_evaluation_complete", 
                               {"input": evaluation_input}, 
                               {"selected_criteria": [c.model_dump() for c in selected_criteria]}, 
                               process_id)
                    
                    # OpenTelemetry capture
                    eval_span.set_attribute("selected_count", str(len(selected_criteria)))
                    selected_summary = [{"criteria": c.criteria, "rating": c.rating} for c in selected_criteria]
                    eval_span.set_attribute("selected_criteria", json.dumps(selected_summary))
                    
                    log_info(f"Selected {len(selected_criteria)} top criteria", truncate=True)
                    verbose_logger.info(f"Selected {len(selected_criteria)} top criteria")

                    # Log selected criteria
                    for i, criterion in enumerate(selected_criteria, 1):
                        log_info(f"Selected Criterion {i}: {criterion.criteria} (Rating: {criterion.rating})", truncate=True)
                        verbose_logger.info(f"Selected Criterion {i}: {criterion.criteria} (Rating: {criterion.rating})")

                # Create output with guardrail check
                with traced_span(tracer, "output_creation", {"output_file": output_file}) as output_span:
                    # Create the output object using Pydantic
                    log_info("CREATING MODULE 1 OUTPUT OBJECT...", truncate=True)
                    verbose_logger.info("CREATING MODULE 1 OUTPUT OBJECT...")

                    module_1_output = Module1Output(
                        goal=user_goal,
                        success_criteria=generated_criteria,
                        selected_criteria=selected_criteria,  # Multiple criteria
                    )
                    
                    # Manual capture
                    capture_step("module_output_created", 
                               {"goal": user_goal}, 
                               {"module_1_output": module_1_output.model_dump()}, 
                               process_id)

                    # Log the complete output (only to verbose log)
                    verbose_logger.info(f"Complete Module 1 output: {json.dumps(module_1_output.model_dump(), indent=2)}")

                    # Add the output guardrail
                    log_info("Applying output guardrail...", truncate=True)
                    verbose_logger.info("Applying output guardrail...")
                    
                    # Manual capture
                    capture_step("guardrail_start", {"module_1_output": module_1_output.model_dump()}, None, process_id)

                    guardrail = OutputGuardrail(guardrail_function=validate_module1_output)
                    
                    # Remove run_config from guardrail.run() call:
                    guardrail_result = await guardrail.run(
                        agent=evaluate_criteria_agent,
                        agent_output=module_1_output,
                        context=context,
                        # run_config=run_config,  <-- REMOVE THIS LINE, it's invalid
                    )

                    # Manual capture
                    guardrail_output = {
                        "tripwire_triggered": guardrail_result.output.tripwire_triggered,
                        "output_info": guardrail_result.output.output_info
                    }
                    capture_step("guardrail_complete", 
                               {"module_1_output": module_1_output.model_dump()}, 
                               guardrail_output, 
                               process_id)
                    
                    # OpenTelemetry capture
                    output_span.set_attribute("guardrail_triggered", str(guardrail_result.output.tripwire_triggered))
                    if guardrail_result.output.output_info:
                        output_span.set_attribute("guardrail_info", str(guardrail_result.output.output_info))

                    if guardrail_result.output.tripwire_triggered:
                        logger.error(f"Guardrail failed: {guardrail_result.output.output_info}")
                        verbose_logger.error(f"Guardrail failed: {guardrail_result.output.output_info}")
                        output_span.set_status(otel_trace.StatusCode.ERROR, "Guardrail triggered")
                        return

                    # --- Smart JSON Export ---
                    log_info("Exporting output to files...", truncate=True)
                    verbose_logger.info("Exporting output to files...")
                    
                    # Create data directory if it doesn't exist
                    output_dir = os.path.dirname(output_file)
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Create timestamped version
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filename = os.path.basename(output_file)
                    name, ext = os.path.splitext(filename)
                    timestamped_file = os.path.join(output_dir, f"{name}_{timestamp}{ext}")
                    
                    # Manual capture
                    capture_step("file_export", 
                               {"module_1_output": module_1_output.model_dump()}, 
                               {"output_file": output_file, "timestamped_file": timestamped_file}, 
                               process_id)
                    
                    # Export both versions
                    with open(output_file, "w") as f:
                        json.dump(module_1_output.model_dump(), f, indent=4)
                    with open(timestamped_file, "w") as f:
                        json.dump(module_1_output.model_dump(), f, indent=4)
                    
                    # Final manual capture
                    capture_step("module_complete", 
                               {"trace_id": openai_trace_id}, 
                               {"status": "success", "files_saved": [output_file, timestamped_file]}, 
                               process_id)
                    
                    log_info(f"Module 1 completed. Output saved to {output_file}", truncate=True)
                    log_info(f"Timestamped output saved to {timestamped_file}", truncate=True)
                    verbose_logger.info(f"Module 1 completed. Output saved to {output_file}")
                    verbose_logger.info(f"Timestamped output saved to {timestamped_file}")

        except Exception as e:
            logger.error(f"An error occurred in Module 1: {e}")
            verbose_logger.error(f"An error occurred in Module 1: {e}")
            import traceback
            error_trace = traceback.format_exc()
            logger.error(error_trace)
            verbose_logger.error(error_trace)  # Log the full stack trace
            
            # Manual capture of error
            capture_step("module_error", 
                       {"user_goal": user_goal}, 
                       {"error": str(e), "traceback": error_trace}, 
                       process_id)
            
            # Record in OpenTelemetry
            module_span.record_exception(e)
            module_span.set_status(otel_trace.StatusCode.ERROR, str(e))

async def main():
    log_info("Starting main function", truncate=True)
    verbose_logger.info("Starting main function")

    # Initialize OpenTelemetry
    tracer = setup_opentelemetry()
    
    # Create a manual trace for the entire process
    process_id = f"process_{uuid.uuid4().hex}"
    capture_step("process_start", {"timestamp": datetime.datetime.now().isoformat()}, None, process_id)
    
    # Start an OpenTelemetry span for the entire process
    with tracer.start_as_current_span("module1_execution") as parent_span:
        user_goal = input("Please enter your goal or idea: ")
        parent_span.set_attribute("user_goal", user_goal)
        capture_step("user_input", None, {"user_goal": user_goal}, process_id)
        
        log_info(f"User input goal: {user_goal}", truncate=True)
        verbose_logger.info(f"User input goal: {user_goal}")

        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "module1_output.json")

        # Set up both tracing systems
        file_processor = EnhancedFileTracingProcessor()
        add_trace_processor(file_processor)
        
        # Verify API key is set
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            log_info(f"API key found (length: {len(api_key)})", truncate=True)
            parent_span.set_attribute("api_key_found", "true")
        else:
            log_info("WARNING: OPENAI_API_KEY not found in environment variables", truncate=True)
            parent_span.set_attribute("api_key_found", "false")

        # Add cleanup at exit
        atexit.register(file_processor.shutdown)

        # Update logging hooks with the tracer AND process_id
        global logging_hooks
        logging_hooks = DetailedLoggingHooks(logger, verbose_logger, tracer, process_id)

        # Update agents with new hooks that have tracer AND process_id
        search_agent.hooks = logging_hooks
        generate_criteria_agent.hooks = logging_hooks
        evaluate_criteria_agent.hooks = logging_hooks

        
        # Run module 1 with tracing
        await run_module_1(user_goal, output_file, tracer, process_id)
        
        # Final capture for manual trace
        capture_step("process_complete", {"timestamp": datetime.datetime.now().isoformat()}, None, process_id)
        
        log_info("Main function completed", truncate=True)
        verbose_logger.info("Main function completed")

if __name__ == "__main__":
    log_info("Module 1 script starting", truncate=True)
    verbose_logger.info("Module 1 script starting")

    asyncio.run(main())
    log_info("Module 1 script completed", truncate=True)
    verbose_logger.info("Module 1 script completed")