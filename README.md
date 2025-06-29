# OpenTelemetry Collector distributions

> :warning: **Important note:** Git tags in this repository may change at any time to fix any issues found during a release. They are only meant to trigger Github releases and should not be relied upon.

This repository assembles OpenTelemetry Collector distributions, such as the "core" distribution, or "contrib". It may contain non-official distributions, focused on specific use-cases, such as the load-balancer.

Each distribution contains:

- Binaries for a multitude of platforms and architectures (at least linux_amd64, linux_arm64, windows_amd64 and darwin_arm64)
- Multi-arch container images (at least amd64 and arm64)
- Packages to be used with Linux distributions (apk, RPM, deb), Mac OS (brew) for the above-mentioned architectures

More details about each individual distribution can be seen in its own readme files.

Current list of distributions:

- [OpenTelemetry Collector (also known as "otelcol")](./distributions/otelcol)
- [OpenTelemetry Collector Contrib (also known as "otelcol-contrib")](./distributions/otelcol-contrib)

________

# OpenTelemetry Collector Setup

This document explains how to install and set up the OpenTelemetry Collector for local development and testing.

## Installation

### macOS (ARM64)

Download the latest OpenTelemetry Collector Contrib distribution:

```bash
curl -LO https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v0.91.0/otelcol-contrib_0.91.0_darwin_arm64.tar.gz
tar -xzf otelcol-contrib_0.91.0_darwin_arm64.tar.gz
chmod +x otelcol-contrib
```

### macOS (Intel)

```bash
curl -LO https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v0.91.0/otelcol-contrib_0.91.0_darwin_amd64.tar.gz
tar -xzf otelcol-contrib_0.91.0_darwin_amd64.tar.gz
chmod +x otelcol-contrib
```

### Linux

```bash
curl -LO https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v0.91.0/otelcol-contrib_0.91.0_linux_amd64.tar.gz
tar -xzf otelcol-contrib_0.91.0_linux_amd64.tar.gz
chmod +x otelcol-contrib
```

### Alternative: Homebrew (macOS)

While the official Homebrew formula is not available, you can install via the releases as shown above.

## Usage

1. Start the collector with the test configuration:
```bash
./otelcol-contrib --config testdata/otelcol_file.yaml
```

2. The collector will listen on `localhost:4318` for OTLP/HTTP traces.

3. Run your tests or the module to send traces to the collector.

## Important Notes

- The `otelcol-contrib` binary is **NOT** committed to the repository due to its large size (>250MB)
- Download it locally when needed for development
- The binary is listed in `.gitignore` to prevent accidental commits
- For production, use a proper deployment method (Docker, Kubernetes, etc.)

## Docker Alternative

For a more portable solution, you can run the collector in Docker:

```bash
docker run -p 4318:4318 \
  -v $(pwd)/testdata/otelcol_file.yaml:/etc/otelcol-contrib/config.yaml \
  otel/opentelemetry-collector-contrib:latest
```

___________________

# Tracing Migration Guide

This guide covers the migration from legacy file-based tracing to OpenTelemetry Protocol (OTLP) tracing for Agento Module 1.

## Overview

Module 1 has been upgraded to use industry-standard OpenTelemetry (OTEL) tracing with OTLP export. This provides:

- Standards-compliant trace format
- Rich hierarchical span relationships
- Full prompt and response capture via events
- Automatic handling of large payloads
- Integration with OpenAI SDK traces via Links

## Configuration

### Environment Variables

#### Required Settings

```bash
# OTLP endpoint for trace export (default: http://localhost:4318/v1/traces)
export OTEL_ENDPOINT="http://localhost:4318/v1/traces"

# Protocol for OTLP export
export OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf"

# OpenAI API key (required for agent operations)
export OPENAI_API_KEY="your-api-key"
```

#### Legacy Mode (Temporary)

To enable legacy file-based tracing during the transition period:

```bash
# Enable legacy JSON file export (will be removed after 2025-Q3)
export TRACE_LEGACY=1  # or "true" or "yes"
```

## Running with OpenTelemetry Collector

### Local Development Setup

1. Install OpenTelemetry Collector:
```bash
# Using homebrew
brew install opentelemetry-collector-contrib

# Or download from releases
wget https://github.com/open-telemetry/opentelemetry-collector-releases/releases/latest/download/otelcol-contrib_linux_amd64
```

2. Use the provided collector configuration:
```bash
otelcol-contrib --config testdata/otelcol_file.yaml
```

3. Run Module 1:
```bash
python module1.py
```

### Production Setup with Jaeger

1. Start Jaeger (all-in-one):
```bash
docker run -d --name jaeger \
  -e COLLECTOR_OTLP_ENABLED=true \
  -p 16686:16686 \
  -p 4318:4318 \
  jaegertracing/all-in-one:latest
```

2. Set the endpoint:
```bash
export OTEL_ENDPOINT="http://localhost:4318/v1/traces"
```

3. View traces at http://localhost:16686

## Trace Structure

### Service Identification

All traces from Module 1 are identified by:
- `service.name`: "Agento-Module-1"
- `service.version`: "1.1.0"
- `service.namespace`: "agento"

### Span Hierarchy

```
run_module_1 (root span, linked to OpenAI trace)
├── search_agent
├── criteria_generation
├── criteria_evaluation
└── output_creation
```

### Key Attributes and Events

Each agent span includes:
- `agent.name`: Name of the agent
- `ai.prompt`: The input prompt (truncated if > 8KB)
- `ai.response_excerpt`: Response summary

Each span has events:
- `full_prompt`: Complete input text
- `full_response`: Complete output
- `capture.*`: Workflow step data

## Querying Traces in Jaeger

### Find All Module 1 Traces
```
service.name="Agento-Module-1"
```

### Find Traces with Specific Agents
```
service.name="Agento-Module-1" AND agent.name="SearchAgent"
```

### Find Failed Operations
```
service.name="Agento-Module-1" AND status.code=ERROR
```

### Find Traces by Goal
```
service.name="Agento-Module-1" AND user_goal="your search term"
```

## Migration Timeline

1. **Now - 2025-Q2**: Both OTLP and legacy file export available
   - OTLP is primary, legacy requires `TRACE_LEGACY=1`
   - Monitor both to ensure parity

2. **2025-Q3**: Legacy file export removed
   - Remove `EnhancedFileTracingProcessor` class
   - Remove `TRACE_LEGACY` environment variable check

## Troubleshooting

### No Traces Appearing

1. Verify collector is running:
```bash
curl http://localhost:4318/v1/traces
# Should return 404 (endpoint exists but needs POST)
```

2. Check environment variables:
```bash
env | grep OTEL
```

3. Enable console export for debugging:
```python
# Module already includes ConsoleSpanExporter for debugging
```

### Large Payload Issues

Payloads > 8KB are automatically handled:
- Attribute gets truncated version with `_truncated` suffix
- Full content stored in `ai.payload.large` event
- No manual intervention needed

### Linking to OpenAI Traces

OpenAI SDK trace IDs are automatically linked when available:
- Look for spans with Links in Jaeger UI
- Link has attribute `source="openai-agents-sdk"`

## Best Practices

1. **Always set OTEL_EXPORTER_OTLP_PROTOCOL**: The http/protobuf protocol is recommended
2. **Use Jaeger for development**: Provides best UI for trace exploration
3. **Monitor collector health**: Check for dropped spans in collector metrics
4. **Set resource limits**: Configure collector memory limits for production

## Next Steps

After Module 1, the same pattern will be applied to Modules 2-5:
- Shared tracing utilities in `agento_tracing.py`
- Consistent span naming: `agent.run`, `validation`, etc.
- Trace ID propagation between modules via Links

## Required Environment Variables

- `OPENAI_API_KEY`: Your API key for OpenAI services.
- `OTEL_EXPORTER_OTLP_ENDPOINT`: The URL of your OpenTelemetry collector (e.g., `http://localhost:4318/v1/traces`). Set to `disabled` to run without a collector.
- `DEPLOYMENT_ENV`: (Optional) The deployment environment, e.g., `development` or `production`.
- `OTEL_CONSOLE_EXPORT`: (Optional) Set to `true` to see trace data printed to the console for debugging.