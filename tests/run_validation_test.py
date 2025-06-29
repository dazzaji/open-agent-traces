#!/usr/bin/env python3
"""
Simplified test to generate and validate OpenTelemetry traces.
"""
import subprocess
import time
import os
import json
import tempfile
import sys

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

service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [file]
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


def create_test_script(script_path):
    """Create a simple test script that generates traces."""
    test_code = '''
import os
import sys
import time

# Set up environment
os.environ["OTEL_ENDPOINT"] = "http://localhost:4318/v1/traces"
os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/protobuf"

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and use the setup function
import importlib.util
spec = importlib.util.spec_from_file_location(
    "module1", 
    "module1.py"
)
module1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module1)

# Setup OpenTelemetry
tracer = module1.setup_opentelemetry("test-service")

# Create test spans with all our features
with tracer.start_as_current_span("test_root_span") as root_span:
    # Test safe_set with small and large values
    module1.safe_set(root_span, "small_attr", "This is a small value")
    module1.safe_set(root_span, "large_attr", "X" * 10000)  # Larger than 8KB
    
    # Add events
    root_span.add_event("full_prompt", {"text": "Test prompt content"})
    root_span.add_event("full_response", {"text": "Test response content"})
    
    # Create a child span with capture events
    with tracer.start_as_current_span("test_child_span") as child_span:
        # Test capture_step dual write
        module1.capture_step("test_stage", {"input": "test"}, {"output": "result"}, "test-id")
        
        # Add more attributes
        child_span.set_attribute("agent.name", "TestAgent")
        child_span.set_attribute("ai.prompt", "Test prompt")
        
        # Simulate work
        time.sleep(0.1)
    
    # Add a link (simulating OpenAI trace link)
    from opentelemetry.trace import Link, SpanContext, TraceFlags
    import random
    
    link_ctx = SpanContext(
        trace_id=random.getrandbits(128),
        span_id=random.getrandbits(64) or 1,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
        is_remote=True
    )
    
    with tracer.start_as_current_span(
        "test_linked_span",
        links=[Link(link_ctx, {"source": "openai-agents-sdk"})]
    ) as linked_span:
        linked_span.set_attribute("has_link", True)

print("✅ Test traces generated")

# Give time for export
time.sleep(2)
'''
    
    with open(script_path, 'w') as f:
        f.write(test_code)


def validate_traces(trace_file):
    """Validate the generated traces."""
    print(f"\nValidating traces from: {trace_file}")
    
    # First run our validation script
    result = subprocess.run(
        [sys.executable, "tests/validate_otlp_traces.py", trace_file, "-v"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    # Check specific requirements
    print("\nChecking specific requirements...")
    
    try:
        with open(trace_file, 'r') as f:
            content = f.read()
            
        # Parse JSON (might be line-delimited)
        traces = []
        for line in content.strip().split('\n'):
            if line:
                try:
                    traces.append(json.loads(line))
                except:
                    pass
        
        if not traces:
            traces = [json.loads(content)]
        
        checks = {
            "OTLP JSON structure": False,
            "Valid trace/span IDs": False,
            "Required fields": False,
            "Attribute/event/link structures": False,
            "Custom events (full_prompt/full_response)": False,
            "OpenAI trace links": False,
            "Large payload handling": False,
            "Capture events": False,
        }
        
        for trace_data in traces:
            if "resourceSpans" in trace_data:
                checks["OTLP JSON structure"] = True
                
                for rs in trace_data["resourceSpans"]:
                    for ss in rs.get("scopeSpans", []):
                        for span in ss.get("spans", []):
                            # Check IDs
                            if (span.get("traceId") and len(span["traceId"]) == 32 and
                                span.get("spanId") and len(span["spanId"]) == 16):
                                checks["Valid trace/span IDs"] = True
                            
                            # Check required fields
                            if span.get("name") and span.get("traceId") and span.get("spanId"):
                                checks["Required fields"] = True
                            
                            # Check structures
                            if (isinstance(span.get("attributes", []), list) and
                                isinstance(span.get("events", []), list)):
                                checks["Attribute/event/link structures"] = True
                            
                            # Check events
                            for event in span.get("events", []):
                                event_name = event.get("name", "")
                                if event_name in ["full_prompt", "full_response"]:
                                    checks["Custom events (full_prompt/full_response)"] = True
                                elif event_name.startswith("capture."):
                                    checks["Capture events"] = True
                                elif event_name == "ai.payload.large":
                                    checks["Large payload handling"] = True
                            
                            # Check links
                            for link in span.get("links", []):
                                attrs = link.get("attributes", [])
                                for attr in attrs:
                                    if (attr.get("key") == "source" and 
                                        attr.get("value", {}).get("stringValue") == "openai-agents-sdk"):
                                        checks["OpenAI trace links"] = True
        
        print("\nValidation Results:")
        all_passed = True
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check}")
            if not passed:
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error validating traces: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("OpenTelemetry Trace Validation Test")
    print("=" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "collector.yaml")
        output_file = os.path.join(tmpdir, "traces.json")
        test_script = os.path.join(tmpdir, "test_trace_gen.py")
        
        # Start collector
        collector = start_collector(config_path, output_file)
        if not collector:
            return 1
        
        try:
            # Create and run test script
            print("\nGenerating test traces...")
            create_test_script(test_script)
            
            # Run the test script in the project directory
            result = subprocess.run(
                [sys.executable, test_script],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print("❌ Test script failed:")
                print(result.stdout)
                print(result.stderr)
                return 1
            
            print(result.stdout)
            
            # Give collector time to write
            time.sleep(2)
            
            # Check if traces were exported
            if not os.path.exists(output_file):
                print(f"\n❌ No trace file created at {output_file}")
                return 1
            
            file_size = os.path.getsize(output_file)
            print(f"\n✅ Trace file created: {file_size} bytes")
            
            # Validate the traces
            if validate_traces(output_file):
                print("\n✅ All validation checks passed!")
                
                # Show a sample of the trace data
                with open(output_file, 'r') as f:
                    content = f.read()
                    print(f"\nSample trace data (first 500 chars):")
                    print(content[:500] + "..." if len(content) > 500 else content)
                
                return 0
            else:
                print("\n❌ Some validation checks failed")
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