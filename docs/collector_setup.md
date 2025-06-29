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