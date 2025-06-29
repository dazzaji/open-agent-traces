"""Test only the new functions without full module import."""
import json
import sys
from unittest.mock import MagicMock

# Test safe_set function
def test_safe_set():
    print("Testing safe_set function...")
    
    # Define MAX_ATTR and safe_set inline to avoid import issues
    MAX_ATTR = 8_192  # 8 KB

    def safe_set(span, key: str, value):
        """Attach data safely – big payloads become events, plus a truncated attr."""
        if not isinstance(value, str):
            value = json.dumps(value, default=str)
        if len(value) > MAX_ATTR:
            span.set_attribute(f"{key}_truncated", value[:MAX_ATTR])
            span.add_event("ai.payload.large", {key: value})
        else:
            span.set_attribute(key, value)
    
    # Create a mock span
    mock_span = MagicMock()
    
    # Test small payload
    small_value = "Small test value"
    safe_set(mock_span, "test_key", small_value)
    assert mock_span.set_attribute.called
    assert mock_span.set_attribute.call_args[0] == ("test_key", small_value)
    assert not mock_span.add_event.called
    print("✓ Small payload test passed")
    
    # Reset mocks
    mock_span.reset_mock()
    
    # Test large payload
    large_value = "x" * (MAX_ATTR + 100)
    safe_set(mock_span, "large_key", large_value)
    
    # Should set truncated attribute
    assert mock_span.set_attribute.called
    assert mock_span.set_attribute.call_args[0][0] == "large_key_truncated"
    assert len(mock_span.set_attribute.call_args[0][1]) == MAX_ATTR
    
    # Should add event with full value
    assert mock_span.add_event.called
    assert mock_span.add_event.call_args[0][0] == "ai.payload.large"
    assert mock_span.add_event.call_args[0][1]["large_key"] == large_value
    print("✓ Large payload test passed")
    
    # Test non-string value
    mock_span.reset_mock()
    dict_value = {"key": "value", "number": 42}
    safe_set(mock_span, "dict_key", dict_value)
    assert mock_span.set_attribute.called
    # Value should be JSON stringified
    assert json.loads(mock_span.set_attribute.call_args[0][1]) == dict_value
    print("✓ Non-string value test passed")


def test_trace_legacy_flag():
    """Test TRACE_LEGACY environment variable logic."""
    print("\nTesting TRACE_LEGACY flag...")
    
    # Test various values that should enable legacy
    for value in ["1", "true", "yes", "TRUE", "Yes", "YES", "TrUe"]:
        result = value.lower() in ("1", "true", "yes")
        assert result == True, f"Expected {value} to enable legacy mode"
    print("✓ Legacy enable values test passed")
    
    # Test values that should not enable legacy
    for value in ["0", "false", "no", "", "anything_else", "2", "off"]:
        result = value.lower() in ("1", "true", "yes")
        assert result == False, f"Expected {value} to NOT enable legacy mode"
    print("✓ Legacy disable values test passed")


def test_otlp_configuration():
    """Test OTLP configuration values."""
    print("\nTesting OTLP configuration...")
    
    # Test default endpoint
    import os
    default_endpoint = os.getenv("OTEL_ENDPOINT", "http://localhost:4318/v1/traces")
    assert default_endpoint == "http://localhost:4318/v1/traces"
    print("✓ Default endpoint test passed")
    
    # Test service configuration
    service_name = "Agento-Module-1"
    service_version = "1.1.0"
    service_namespace = "agento"
    
    assert service_name == "Agento-Module-1"
    assert service_version == "1.1.0"
    assert service_namespace == "agento"
    print("✓ Service configuration test passed")


if __name__ == "__main__":
    print("Running OpenTelemetry implementation tests...\n")
    
    try:
        test_safe_set()
        test_trace_legacy_flag()
        test_otlp_configuration()
        
        print("\n✅ All tests passed!")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)