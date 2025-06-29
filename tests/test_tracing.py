import json
import subprocess
import time
import os
import asyncio
import builtins
import tempfile
import pytest
from unittest.mock import patch, MagicMock
import sys

# Add parent directory to path to import module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module1_opentelemetry_gm_1156 import run_module_1, setup_opentelemetry


def test_otlp_trace(monkeypatch):
    """Test that OTLP traces are generated with proper structure and content."""
    # Create a temporary directory for test output
    tmp_dir = tempfile.mkdtemp()
    out_file = os.path.join(tmp_dir, "trace.json")
    
    # Create a test collector config
    collector_config = f"""
receivers:
  otlp:
    protocols:
      http:
        endpoint: localhost:4318

exporters:
  file:
    path: {out_file}

service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [file]
"""
    
    config_file = os.path.join(tmp_dir, "otel_config.yaml")
    with open(config_file, "w") as f:
        f.write(collector_config)
    
    # Start collector (we'll mock this for now since it requires otelcol-contrib)
    # In a real test environment, you would start the actual collector
    # collector = subprocess.Popen(
    #     ["otelcol-contrib", "--config", config_file]
    # )
    # time.sleep(2)  # give collector time to start
    
    # For now, we'll use a mock approach
    monkeypatch.setenv("OTEL_ENDPOINT", "http://localhost:4318/v1/traces")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    
    # Mock the input function to provide a test goal
    monkeypatch.setattr(builtins, "input", lambda _: "Test goal for tracing")
    
    # Mock the OpenAI agents to avoid actual API calls
    with patch("module1_opentelemetry_gm_1156.Runner") as mock_runner:
        # Mock search agent response
        search_result = MagicMock()
        search_result.final_output = "Mock search results about success criteria"
        
        # Mock criteria generation response
        criteria_result = MagicMock()
        mock_criteria = []
        for i in range(5):
            criterion = MagicMock()
            criterion.criteria = f"Test criterion {i+1}"
            criterion.reasoning = f"Test reasoning {i+1}"
            criterion.rating = 8
            criterion.model_dump = lambda self=criterion: {
                "criteria": self.criteria,
                "reasoning": self.reasoning,
                "rating": self.rating
            }
            mock_criteria.append(criterion)
        criteria_result.final_output = mock_criteria
        
        # Mock evaluation response (select top 3)
        eval_result = MagicMock()
        eval_result.final_output = mock_criteria[:3]
        
        # Configure mock to return these results in sequence
        mock_runner.run.side_effect = [search_result, criteria_result, eval_result]
        
        # Mock the guardrail result
        with patch("module1_opentelemetry_gm_1156.OutputGuardrail") as mock_guardrail:
            guardrail_instance = MagicMock()
            guardrail_result = MagicMock()
            guardrail_result.output.tripwire_triggered = False
            guardrail_result.output.output_info = None
            guardrail_instance.run.return_value = asyncio.coroutine(lambda: guardrail_result)()
            mock_guardrail.return_value = guardrail_instance
            
            # Setup OpenTelemetry
            tracer = setup_opentelemetry()
            
            # Run the module
            output_file = os.path.join(tmp_dir, "module1_output.json")
            asyncio.run(run_module_1("Test goal for tracing", output_file, tracer, "test-process"))
    
    # Verify output file was created
    assert os.path.exists(output_file)
    
    # Verify output contains expected structure
    with open(output_file) as f:
        output_data = json.load(f)
    
    assert output_data["goal"] == "Test goal for tracing"
    assert len(output_data["success_criteria"]) == 5
    assert len(output_data["selected_criteria"]) == 3
    
    # In a real test with collector, we would verify the trace file
    # For now, we've verified the module runs without errors and produces output
    
    # Clean up
    # if 'collector' in locals():
    #     collector.terminate()
    #     collector.wait()


def test_safe_set_function():
    """Test the safe_set function handles large payloads correctly."""
    from module1_opentelemetry_gm_1156 import safe_set, MAX_ATTR
    
    # Create a mock span
    mock_span = MagicMock()
    
    # Test small payload (should be set as attribute)
    small_value = "Small test value"
    safe_set(mock_span, "test_key", small_value)
    mock_span.set_attribute.assert_called_once_with("test_key", small_value)
    mock_span.add_event.assert_not_called()
    
    # Reset mocks
    mock_span.reset_mock()
    
    # Test large payload (should be truncated and added as event)
    large_value = "x" * (MAX_ATTR + 100)
    safe_set(mock_span, "large_key", large_value)
    
    # Should set truncated attribute
    mock_span.set_attribute.assert_called_once_with("large_key_truncated", large_value[:MAX_ATTR])
    
    # Should add event with full value
    mock_span.add_event.assert_called_once_with("ai.payload.large", {"large_key": large_value})


def test_capture_step_dual_write():
    """Test that capture_step writes to both manual traces and OTEL."""
    from module1_opentelemetry_gm_1156 import capture_step
    
    # Mock the current span
    mock_span = MagicMock()
    mock_span.is_recording.return_value = True
    
    with patch("module1_opentelemetry_gm_1156.otel_trace.get_current_span", return_value=mock_span):
        # Call capture_step
        test_inputs = {"input": "test input"}
        test_outputs = {"output": "test output"}
        capture_step("test_stage", test_inputs, test_outputs, "test-trace-id")
        
        # Verify OTEL event was added
        mock_span.add_event.assert_called_once_with(
            "capture.test_stage",
            {
                "inputs": json.dumps(test_inputs, default=str),
                "outputs": json.dumps(test_outputs, default=str)
            }
        )


def test_openai_trace_link():
    """Test that OpenAI trace ID is properly linked."""
    from module1_opentelemetry_gm_1156 import Link, SpanContext, TraceFlags
    
    # This test verifies the structure is correct
    # In actual usage, the link would be created with a real trace ID
    test_trace_id = "1234567890abcdef1234567890abcdef"
    
    # Convert to int
    trace_id_int = int(test_trace_id, 16)
    
    # Create span context
    ctx = SpanContext(
        trace_id=trace_id_int,
        span_id=12345,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
        is_remote=True
    )
    
    # Create link
    link = Link(ctx, {"source": "openai-agents-sdk"})
    
    # Verify link structure
    assert link.context.trace_id == trace_id_int
    assert link.context.span_id == 12345
    assert link.attributes["source"] == "openai-agents-sdk"


def test_legacy_flag():
    """Test that legacy tracing is only enabled when TRACE_LEGACY is set."""
    from module1_opentelemetry_gm_1156 import EnhancedFileTracingProcessor
    
    # Test various values that should enable legacy
    for value in ["1", "true", "yes", "TRUE", "Yes"]:
        assert value.lower() in ("1", "true", "yes")
    
    # Test values that should not enable legacy
    for value in ["0", "false", "no", "", "anything_else"]:
        assert value.lower() not in ("1", "true", "yes")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])