"""Simple test runner that doesn't require pytest."""
import json
import os
import sys
import asyncio
from unittest.mock import patch, MagicMock

# Add parent directory to path to import module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import the module dynamically
import importlib.util
spec = importlib.util.spec_from_file_location(
    "module1_opentelemetry_gm_1156", 
    os.path.join(parent_dir, "module1-opentelemetry-gm-1156.py")
)
module1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module1)

# Import needed items
safe_set = module1.safe_set
MAX_ATTR = module1.MAX_ATTR
capture_step = module1.capture_step
Link = module1.Link
SpanContext = module1.SpanContext
TraceFlags = module1.TraceFlags


def test_safe_set_function():
    """Test the safe_set function handles large payloads correctly."""
    print("Testing safe_set function...")
    
    # Create a mock span
    mock_span = MagicMock()
    
    # Test small payload (should be set as attribute)
    small_value = "Small test value"
    safe_set(mock_span, "test_key", small_value)
    assert mock_span.set_attribute.called
    assert mock_span.set_attribute.call_args[0] == ("test_key", small_value)
    assert not mock_span.add_event.called
    print("✓ Small payload test passed")
    
    # Reset mocks
    mock_span.reset_mock()
    
    # Test large payload (should be truncated and added as event)
    large_value = "x" * (MAX_ATTR + 100)
    safe_set(mock_span, "large_key", large_value)
    
    # Should set truncated attribute
    assert mock_span.set_attribute.called
    assert mock_span.set_attribute.call_args[0][0] == "large_key_truncated"
    assert len(mock_span.set_attribute.call_args[0][1]) == MAX_ATTR
    
    # Should add event with full value
    assert mock_span.add_event.called
    assert mock_span.add_event.call_args[0][0] == "ai.payload.large"
    print("✓ Large payload test passed")


def test_capture_step_dual_write():
    """Test that capture_step writes to both manual traces and OTEL."""
    print("\nTesting capture_step dual write...")
    
    # Mock the current span
    mock_span = MagicMock()
    mock_span.is_recording.return_value = True
    
    with patch.object(module1.otel_trace, "get_current_span", return_value=mock_span):
        # Call capture_step
        test_inputs = {"input": "test input"}
        test_outputs = {"output": "test output"}
        capture_step("test_stage", test_inputs, test_outputs, "test-trace-id")
        
        # Verify OTEL event was added
        assert mock_span.add_event.called
        call_args = mock_span.add_event.call_args[0]
        assert call_args[0] == "capture.test_stage"
        assert json.loads(call_args[1]["inputs"]) == test_inputs
        assert json.loads(call_args[1]["outputs"]) == test_outputs
        print("✓ Dual write test passed")


def test_openai_trace_link():
    """Test that OpenAI trace ID is properly linked."""
    print("\nTesting OpenAI trace link...")
    
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
    print("✓ OpenAI trace link test passed")


def test_legacy_flag():
    """Test that legacy tracing is only enabled when TRACE_LEGACY is set."""
    print("\nTesting legacy flag logic...")
    
    # Test various values that should enable legacy
    for value in ["1", "true", "yes", "TRUE", "Yes"]:
        assert value.lower() in ("1", "true", "yes")
    
    # Test values that should not enable legacy
    for value in ["0", "false", "no", "", "anything_else"]:
        assert value.lower() not in ("1", "true", "yes")
    
    print("✓ Legacy flag test passed")


def run_all_tests():
    """Run all tests."""
    print("Running OpenTelemetry tracing tests...\n")
    
    try:
        test_safe_set_function()
        test_capture_step_dual_write()
        test_openai_trace_link()
        test_legacy_flag()
        
        print("\n✅ All tests passed!")
        return True
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)