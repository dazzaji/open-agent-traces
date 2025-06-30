# Agento Observability

A prototype for implementing comprehensive, SDK-independent observability in the Agento multi-module AI system.

## Overview

This project demonstrates how to implement OpenTelemetry (OTEL) tracing for multi-agent AI systems without coupling to specific agent SDKs. The implementation provides standardized observability across all modules while maintaining flexibility to use different LLM providers.

## Features

- **SDK-Independent Tracing**: Direct OpenAI API calls with comprehensive OTEL instrumentation
- **Multi-Agent Workflow**: Search → Generate → Evaluate pipeline with full tracing
- **Graceful Fallbacks**: Continues operation even when OTEL collector is unavailable
- **Large Payload Handling**: Automatic truncation and event creation for oversized attributes
- **Cross-Module Context Propagation**: Trace context passed between modules via metadata

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/agento-observability.git
cd agento-observability

# Create and activate a virtual environment (using uv)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package with dev dependencies
uv pip install -e ".[dev]"
```

## Required Environment Variables

```bash
# Required - Your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Optional - OTEL collector endpoint (defaults to http://localhost:4318/v1/traces)
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318/v1/traces"

# Optional - Enable/disable console trace output (defaults to true)
export OTEL_CONSOLE_EXPORT="true"

# Optional - Deployment environment (defaults to development)
export DEPLOYMENT_ENV="development"
```

## Usage

### Running with OpenTelemetry Collector

1. Start the OTEL collector in one terminal:
   ```bash
   ./otelcol-contrib --config testdata/otelcol_file.yaml
   ```

2. Run Module 1 in another terminal:
   ```bash
   python module1.py
   ```

3. Enter your goal when prompted and the system will:
   - Search for relevant information
   - Generate success criteria
   - Evaluate and select the best criteria
   - Save output to `data/module1_output.json`
   - Export traces to `test-traces.json`

### Running without OpenTelemetry Collector

To run with tracing disabled (NoOp mode):
```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="disabled"
python module1.py
```

## Project Structure

```
agento-observability/
├── module1.py              # Main module with multi-agent workflow
├── agento_tracing.py       # Shared OTEL utilities
├── tests/                  # Unit tests
│   ├── test_module1_tracing.py
│   └── validate_otlp_traces.py
├── testdata/              # Test configurations
│   └── otelcol_file.yaml  # OTEL collector config
├── data/                  # Output directory (created at runtime)
└── pyproject.toml         # Project metadata and dependencies
```

## Architecture

The system implements a three-agent workflow:

1. **SearchAgent**: Searches for information related to the user's goal
2. **CriteriaGenerator**: Creates success criteria based on search results
3. **CriteriaEvaluator**: Evaluates and selects the most relevant criteria

Each agent operation is traced with:
- Input prompts and outputs
- Timing information
- Error tracking
- Custom attributes

## Testing

Run the test suite:
```bash
pytest -q
```

Validate OTLP trace output:
```bash
python tests/validate_otlp_traces.py ./test-traces.json
```

## Trace Attributes

The system captures the following trace attributes:

- `user.goal`: The initial user input
- `agent.name`: Name of the agent being executed
- `ai.prompt`: The prompt sent to the LLM
- `ai.response`: The LLM response (truncated if > 8KB)
- `service.name`: Module identifier
- `module.number`: Numeric module ID
- Error information when exceptions occur

## Development

### Adding New Agents

1. Define a new `Agent` instance with appropriate instructions
2. Add a traced span for the agent operation
3. Use `safe_set()` for all attributes to handle large payloads

### Extending to Other Modules

1. Import `agento_tracing` utilities
2. Initialize tracer with appropriate service name and module number
3. Use `traced_span` context manager for all operations
4. Pass trace context via `trace_metadata` in output

## License

Apache-2.0