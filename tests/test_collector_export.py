#!/usr/bin/env python3
"""
Test script to verify OTLP export works with a real OpenTelemetry Collector.
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
  logging:
    loglevel: debug

service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [file, logging]
"""
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    # Try to find otelcol-contrib
    collector_cmd = None
    for cmd in ['otelcol-contrib', 'otelcol', 'otelcorecol']:
        try:
            subprocess.run([cmd, '--version'], capture_output=True, check=True)
            collector_cmd = cmd
            break
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    if not collector_cmd:
        print("❌ OpenTelemetry Collector not found. Please install it:")
        print("   brew install opentelemetry-collector-contrib")
        print("   or download from: https://github.com/open-telemetry/opentelemetry-collector-releases")
        return None
    
    print(f"Starting collector with {collector_cmd}...")
    proc = subprocess.Popen(
        [collector_cmd, '--config', config_path],
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
    
    return proc


def test_module_export():
    """Test module's OTLP export."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Import module
    from module1_opentelemetry_gm_1156 import setup_opentelemetry
    
    # Setup tracing
    os.environ["OTEL_ENDPOINT"] = "http://localhost:4318/v1/traces"
    os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/protobuf"
    
    tracer = setup_opentelemetry("test-service")
    
    # Create a test span with all our features
    with tracer.start_as_current_span("test_span") as span:
        # Test safe_set with small value
        from module1_opentelemetry_gm_1156 import safe_set
        safe_set(span, "test_attribute", "small value")
        
        # Test safe_set with large value
        large_value = "x" * 10000
        safe_set(span, "large_attribute", large_value)
        
        # Add an event
        span.add_event("test_event", {"key": "value"})
        
        # Simulate some work
        time.sleep(0.1)
    
    # Give time for export
    time.sleep(2)
    
    print("✅ Test span sent to collector")


def main():
    """Main test function."""
    print("OpenTelemetry Collector Export Test")
    print("===================================\n")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "collector.yaml")
        output_file = os.path.join(tmpdir, "traces.json")
        
        # Start collector
        collector = start_collector(config_path, output_file)
        if not collector:
            return 1
        
        try:
            # Run test
            test_module_export()
            
            # Give collector time to write
            time.sleep(2)
            
            # Check output
            if os.path.exists(output_file):
                print(f"\n✅ Traces exported to: {output_file}")
                
                # Validate the output
                print("\nValidating exported traces...")
                result = subprocess.run(
                    [sys.executable, "tests/validate_otlp_traces.py", output_file, "-v"],
                    capture_output=True,
                    text=True
                )
                print(result.stdout)
                if result.returncode != 0:
                    print(result.stderr)
                    return 1
            else:
                print("\n❌ No trace file created")
                return 1
            
        finally:
            # Stop collector
            print("\nStopping collector...")
            collector.terminate()
            collector.wait()
    
    print("\n✅ All tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())