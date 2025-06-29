import json
import os
import asyncio
import sys
from unittest.mock import patch, MagicMock, AsyncMock

# Add parent directory to path to import module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import module1
from module1 import run_module_1, setup_opentelemetry, call_llm


@patch('module1.AsyncOpenAI')
def test_module1_with_mocked_llm(mock_openai):
    """Test Module 1 with mocked LLM calls."""
    # Mock the AsyncOpenAI client
    mock_client = AsyncMock()
    mock_openai.return_value = mock_client
    
    # Test that call_llm returns expected mock data
    async def test_call_llm():
        # Test SearchAgent
        result = await call_llm(mock_client, "SearchAgent", "test prompt")
        assert result == "Mocked search summary from direct OpenAI call."
        
        # Test CriteriaGenerator
        result = await call_llm(mock_client, "CriteriaGenerator", "test prompt")
        assert result == [{"criteria": "Mock Criterion 1", "reasoning": "Reason 1", "rating": 8}]
        
        # Test CriteriaEvaluator
        result = await call_llm(mock_client, "CriteriaEvaluator", "test prompt")
        assert result == [{"criteria": "Mock Selected Criterion", "reasoning": "Selected", "rating": 9}]
    
    asyncio.run(test_call_llm())


@patch('module1.AsyncOpenAI')
@patch('module1.setup_opentelemetry')
def test_module1_output_format(mock_setup_otel, mock_openai):
    """Test that Module 1 produces correctly formatted output with trace_metadata."""
    # Mock the tracer
    mock_tracer = MagicMock()
    mock_span = MagicMock()
    mock_span_context = MagicMock()
    mock_span_context.trace_id = 0x12345678901234567890123456789012
    mock_span_context.span_id = 0x1234567890123456
    mock_span.get_span_context.return_value = mock_span_context
    
    # Setup mock for traced_span context manager
    mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
    mock_tracer.start_as_current_span.return_value.__exit__.return_value = None
    
    mock_setup_otel.return_value = mock_tracer
    
    # Mock the AsyncOpenAI client
    mock_client = AsyncMock()
    mock_openai.return_value = mock_client
    
    async def test_run():
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            output_file = f.name
        
        try:
            # Run the module
            await run_module_1("Test goal", output_file, mock_tracer, mock_client)
            
            # Load and verify the output
            with open(output_file, 'r') as f:
                output = json.load(f)
            
            # Verify structure
            assert "goal" in output
            assert output["goal"] == "Test goal"
            assert "success_criteria" in output
            assert "selected_criteria" in output
            assert "trace_metadata" in output
            
            # Verify trace_metadata format
            trace_meta = output["trace_metadata"]
            assert "trace_id" in trace_meta
            assert "parent_span_id" in trace_meta
            assert "service_name" in trace_meta
            assert trace_meta["service_name"] == "Agento-Module-1"
            
            # Verify trace_id and parent_span_id are properly formatted hex strings
            assert len(trace_meta["trace_id"]) == 32  # 32 hex chars
            assert len(trace_meta["parent_span_id"]) == 16  # 16 hex chars
            
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    asyncio.run(test_run())


def test_graceful_degradation_with_disabled_endpoint():
    """Test that module works when OTEL_EXPORTER_OTLP_ENDPOINT is disabled."""
    with patch.dict(os.environ, {'OTEL_EXPORTER_OTLP_ENDPOINT': 'disabled'}):
        # Import the module after setting the env var
        import importlib
        importlib.reload(module1)
        
        # The module should start without errors
        # This test passes if no exception is raised


@patch('module1.OTLPSpanExporter')
def test_collector_fallback(mock_exporter_class):
    """Test graceful fallback when OTLP exporter fails to initialize."""
    # Make the OTLP exporter raise an exception
    mock_exporter_class.side_effect = Exception("Connection failed")
    
    # Import and setup should not crash
    import importlib
    importlib.reload(module1)
    
    # This test passes if no exception is raised during module import