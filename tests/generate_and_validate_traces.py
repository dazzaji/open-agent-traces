#!/usr/bin/env python3
"""
Generate traces using the module and validate them with otel-validate.
"""
import subprocess
import time
import os
import json
import tempfile
import sys
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import builtins

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def start_collector(config_path, output_file):
    """Start OpenTelemetry Collector with file exporter."""
    # Create a custom config that exports to file
    config_content = f"""
receivers:
  otlp:
    protocols:
      http:
        endpoint: localhost:4318

exporters:
  file:
    path: {output_file}
  logging:
    verbosity: detailed

service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [file, logging]
"""
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    # Start collector
    print("Starting OpenTelemetry Collector...")
    proc = subprocess.Popen(
        ['./otelcol-contrib', '--config', config_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Give it time to start
    time.sleep(3)
    
    # Check if it's running
    if proc.poll() is not None:
        stdout, stderr = proc.communicate()
        print(f"❌ Collector failed to start:")
        print(stderr.decode())
        return None
    
    print("✅ Collector started successfully")
    return proc


async def generate_test_traces():
    """Generate traces by running the module with mocked dependencies."""
    print("\nGenerating test traces...")
    
    # Set environment variables
    os.environ["OTEL_ENDPOINT"] = "http://localhost:4318/v1/traces"
    os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/protobuf"
    os.environ["OPENAI_API_KEY"] = "test-key"
    os.environ["TRACE_LEGACY"] = "0"  # Disable legacy tracing
    
    # Mock the input function
    with patch.object(builtins, 'input', return_value="Test goal for comprehensive tracing validation"):
        # Import the module dynamically
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "module1", 
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "module1-opentelemetry-gm-1156.py")
        )
        module1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module1)
    
    run_module_1 = module1.run_module_1
    setup_opentelemetry = module1.setup_opentelemetry
    Runner = module1.Runner
    OutputGuardrail = module1.OutputGuardrail
    trace = module1.trace
    
    # Mock the OpenAI agents
    with patch.object(module1, "Runner") as mock_runner:
        # Mock search agent response
        search_result = MagicMock()
        search_result.final_output = "Mock search results about success criteria for test goal"
        
        # Mock criteria generation response
        criteria_result = MagicMock()
        mock_criteria = []
        for i in range(5):
            criterion = MagicMock()
            criterion.criteria = f"Test criterion {i+1}: Must achieve specific measurable outcome"
            criterion.reasoning = f"This criterion is important because it measures aspect {i+1}"
            criterion.rating = 8 + (i % 3)
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
        async def mock_run(*args, **kwargs):
            # Return different results based on which agent is being called
            if mock_runner.run.call_count == 1:
                return search_result
            elif mock_runner.run.call_count == 2:
                return criteria_result
            else:
                return eval_result
        
        mock_runner.run = AsyncMock(side_effect=[search_result, criteria_result, eval_result])
        
        # Mock the guardrail
        with patch.object(module1, "OutputGuardrail") as mock_guardrail:
            guardrail_instance = MagicMock()
            guardrail_result = MagicMock()
            guardrail_result.output.tripwire_triggered = False
            guardrail_result.output.output_info = None
            
            async def mock_guardrail_run(*args, **kwargs):
                return guardrail_result
            
            guardrail_instance.run = AsyncMock(side_effect=mock_guardrail_run)
            mock_guardrail.return_value = guardrail_instance
            
            # Mock the trace context to provide an OpenAI trace ID
            with patch.object(module1, "trace") as mock_trace:
                mock_trace_context = MagicMock()
                mock_trace_context.trace_id = "1234567890abcdef1234567890abcdef"
                mock_trace.return_value.__enter__.return_value = mock_trace_context
                
                # Setup OpenTelemetry
                tracer = setup_opentelemetry()
                
                # Run the module
                output_file = os.path.join(tempfile.gettempdir(), "module1_test_output.json")
                await run_module_1("Test goal for comprehensive tracing validation", output_file, tracer, "test-process")
                
                print("✅ Test traces generated successfully")
                
                # Give time for traces to be exported
                time.sleep(2)


def validate_traces(trace_file):
    """Validate the generated traces."""
    print(f"\nValidating traces from: {trace_file}")
    
    # Run our validation script
    result = subprocess.run(
        [sys.executable, "tests/validate_otlp_traces.py", trace_file, "-v"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    # Also check the content manually for our specific requirements
    print("\nChecking for specific requirements...")
    
    try:
        with open(trace_file, 'r') as f:
            # The file might contain multiple JSON objects, one per line
            content = f.read()
            # Try to parse as single JSON first
            try:
                data = json.loads(content)
                traces = [data]
            except json.JSONDecodeError:
                # Try line by line
                traces = []
                for line in content.strip().split('\n'):
                    if line:
                        traces.append(json.loads(line))
        
        requirements_met = {
            "has_full_prompt_events": False,
            "has_full_response_events": False,
            "has_openai_trace_link": False,
            "has_safe_set_truncation": False,
            "has_capture_events": False,
            "has_proper_service_info": False
        }
        
        for trace_data in traces:
            if "resourceSpans" in trace_data:
                for rs in trace_data["resourceSpans"]:
                    # Check resource attributes for service info
                    if "resource" in rs and "attributes" in rs["resource"]:
                        for attr in rs["resource"]["attributes"]:
                            if attr.get("key") == "service.name" and attr.get("value", {}).get("stringValue") == "Agento-Module-1":
                                requirements_met["has_proper_service_info"] = True
                    
                    for ss in rs.get("scopeSpans", []):
                        for span in ss.get("spans", []):
                            # Check for events
                            for event in span.get("events", []):
                                event_name = event.get("name", "")
                                if event_name == "full_prompt":
                                    requirements_met["has_full_prompt_events"] = True
                                elif event_name == "full_response":
                                    requirements_met["has_full_response_events"] = True
                                elif event_name.startswith("capture."):
                                    requirements_met["has_capture_events"] = True
                                elif event_name == "ai.payload.large":
                                    requirements_met["has_safe_set_truncation"] = True
                            
                            # Check for links
                            for link in span.get("links", []):
                                if link.get("attributes"):
                                    for attr in link["attributes"]:
                                        if attr.get("key") == "source" and attr.get("value", {}).get("stringValue") == "openai-agents-sdk":
                                            requirements_met["has_openai_trace_link"] = True
        
        print("\nRequirements Check:")
        all_met = True
        for req, met in requirements_met.items():
            status = "✅" if met else "❌"
            print(f"  {status} {req.replace('_', ' ').title()}")
            if not met:
                all_met = False
        
        return result.returncode == 0 and all_met
        
    except Exception as e:
        print(f"❌ Error checking requirements: {e}")
        return False


def main():
    """Main test function."""
    print("OpenTelemetry Trace Generation and Validation Test")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "collector.yaml")
        output_file = os.path.join(tmpdir, "traces.json")
        
        # Start collector
        collector = start_collector(config_path, output_file)
        if not collector:
            return 1
        
        try:
            # Generate test traces
            asyncio.run(generate_test_traces())
            
            # Give collector time to write
            time.sleep(3)
            
            # Check if traces were exported
            if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                print(f"\n❌ No traces exported to {output_file}")
                # Check collector logs
                stdout, stderr = collector.communicate(timeout=1)
                print("Collector logs:")
                print(stdout.decode())
                print(stderr.decode())
                return 1
            
            # Validate the traces
            if validate_traces(output_file):
                print("\n✅ All validation checks passed!")
                return 0
            else:
                print("\n❌ Validation failed")
                return 1
                
        except Exception as e:
            print(f"\n❌ Error during test: {e}")
            import traceback
            traceback.print_exc()
            return 1
        finally:
            # Stop collector
            print("\nStopping collector...")
            collector.terminate()
            try:
                collector.wait(timeout=5)
            except subprocess.TimeoutExpired:
                collector.kill()
                collector.wait()


if __name__ == "__main__":
    sys.exit(main())