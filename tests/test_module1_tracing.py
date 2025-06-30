import json
import os
import asyncio
import sys
from unittest.mock import patch, MagicMock, AsyncMock
from types import SimpleNamespace

# Add parent directory to path to import module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import module1
from module1 import run_module_1


def create_mock_response(content):
    """Helper to create a mock OpenAI response."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content)
            )
        ]
    )


@patch('module1.AsyncOpenAI')
def test_module1_with_mocked_openai(mock_openai_class):
    """Test Module 1 with mocked OpenAI API calls."""
    # Create mock client
    mock_client = AsyncMock()
    mock_openai_class.return_value = mock_client
    
    # Mock the chat.completions.create method
    mock_create = AsyncMock()
    mock_client.chat = SimpleNamespace(
        completions=SimpleNamespace(create=mock_create)
    )
    
    # Define responses for different agent calls
    async def mock_create_side_effect(*args, **kwargs):
        messages = kwargs.get('messages', [])
        if messages and messages[0]['content']:
            content = messages[0]['content']
            if "SearchAgent" in content or "Find information" in content:
                return create_mock_response("Mocked search summary about success criteria.")
            elif "CriteriaGenerator" in content or "create a list" in content:
                return create_mock_response(json.dumps([
                    {"criteria": "Mock Criterion 1", "reasoning": "Reason 1", "rating": 8},
                    {"criteria": "Mock Criterion 2", "reasoning": "Reason 2", "rating": 7}
                ]))
            elif "CriteriaEvaluator" in content or "select the most relevant" in content:
                return create_mock_response(json.dumps([
                    {"criteria": "Mock Selected Criterion", "reasoning": "Selected as best", "rating": 9}
                ]))
        return create_mock_response("{}")
    
    mock_create.side_effect = mock_create_side_effect
    
    async def test_agent_calls():
        # Test that agents work with the mocked API
        from module1 import search_agent, generate_criteria_agent, evaluate_criteria_agent, SuccessCriteria
        from typing import List
        
        # Test SearchAgent
        result = await search_agent.run(mock_client, "Find information about test prompt", str)
        assert "search summary" in result.lower()
        
        # Test CriteriaGenerator
        result = await generate_criteria_agent.run(mock_client, "test prompt", List[SuccessCriteria])
        assert len(result) == 2
        assert result[0].criteria == "Mock Criterion 1"
        
        # Test CriteriaEvaluator
        result = await evaluate_criteria_agent.run(mock_client, "test prompt", List[SuccessCriteria])
        assert len(result) == 1
        assert result[0].criteria == "Mock Selected Criterion"
    
    asyncio.run(test_agent_calls())


@patch('module1.AsyncOpenAI')
@patch('agento_tracing.setup_opentelemetry')
def test_module1_output_format(mock_setup_otel, mock_openai_class):
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
    mock_openai_class.return_value = mock_client
    
    # Mock the chat.completions.create method
    mock_create = AsyncMock()
    mock_client.chat = SimpleNamespace(
        completions=SimpleNamespace(create=mock_create)
    )
    
    # Setup responses
    async def mock_create_responses(*args, **kwargs):
        messages = kwargs.get('messages', [])
        if messages and messages[0]['content']:
            content = messages[0]['content']
            if "Find information" in content:
                return create_mock_response("Search results for success criteria")
            elif "create a list" in content:
                return create_mock_response(json.dumps([
                    {"criteria": "Test Criterion", "reasoning": "Test reason", "rating": 8}
                ]))
            elif "select the most relevant" in content:
                return create_mock_response(json.dumps([
                    {"criteria": "Selected Criterion", "reasoning": "Best choice", "rating": 9}
                ]))
        return create_mock_response("{}")
    
    mock_create.side_effect = mock_create_responses
    
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
            
            # Verify trace_metadata format (updated to match new format)
            trace_meta = output["trace_metadata"]
            assert "trace_id" in trace_meta
            assert "parent_span_id" in trace_meta
            # Note: service_name is no longer included in trace_metadata
            
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
        import agento_tracing
        importlib.reload(agento_tracing)
        importlib.reload(module1)
        
        # The module should start without errors
        # This test passes if no exception is raised


@patch('agento_tracing.OTLPSpanExporter')
def test_collector_fallback(mock_exporter_class):
    """Test graceful fallback when OTLP exporter fails to initialize."""
    # Make the OTLP exporter raise an exception
    mock_exporter_class.side_effect = Exception("Connection failed")
    
    # Import and setup should not crash
    import importlib
    import agento_tracing
    importlib.reload(agento_tracing)
    importlib.reload(module1)
    
    # This test passes if no exception is raised during module import