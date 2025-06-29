# agento_tracing.py
import os
import sys
import json
import logging
import re
import random
import hashlib
from contextlib import contextmanager
from typing import List, Dict, Any, Optional

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider, Span
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SpanExporter,
    SpanExportResult,
    SimpleSpanProcessor,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.trace import Link, SpanContext, TraceFlags, StatusCode, SpanKind

class NoOpSpanExporter(SpanExporter):
    def export(self, spans: list[Span]) -> SpanExportResult: return SpanExportResult.SUCCESS
    def shutdown(self) -> None: pass

def setup_opentelemetry(service_name: str, module_number: int) -> otel_trace.Tracer:
    """Standard OTEL setup for all Agento modules with a graceful fallback."""
    resource = Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: "1.1.0",
        "service.namespace": "agento",
        "process.pid": os.getpid(),
        "module.number": module_number,
        "deployment.environment": os.getenv("DEPLOYMENT_ENV", "development")
    })
    tracer_provider = TracerProvider(resource=resource)
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
    
    if endpoint.lower() == 'disabled':
        logging.warning("OTEL_EXPORTER_OTLP_ENDPOINT is 'disabled'. Using NoOpExporter.")
        exporter = NoOpSpanExporter()
    else:
        try:
            exporter = OTLPSpanExporter(endpoint=endpoint, timeout=5)
        except Exception as e:
            logging.error(f"Failed to initialize OTLP exporter to '{endpoint}': {e}. Spans will be dropped.")
            exporter = NoOpSpanExporter()
    
    tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
    if os.getenv("OTEL_CONSOLE_EXPORT", "true").lower() == "true":
        tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    
    otel_trace.set_tracer_provider(tracer_provider)
    return otel_trace.get_tracer(f"agento.module{module_number}")

MAX_ATTR_BYTES = 8192

def safe_set(span: Span, key: str, value: Any):
    """Safely set attributes, handling large payloads by creating events."""
    if not isinstance(value, str):
        value = json.dumps(value, default=str)
    value_bytes = value.encode("utf-8")
    if len(value_bytes) > MAX_ATTR_BYTES:
        sha256_hash = hashlib.sha256(value_bytes).hexdigest()
        span.set_attribute(f"{key}_truncated", value_bytes[:MAX_ATTR_BYTES].decode("utf-8", "ignore"))
        span.set_attribute(f"{key}_sha256", sha256_hash)
        span.add_event(name="ai.payload.large", attributes={key: value})
    else:
        span.set_attribute(key, value)

@contextmanager
def traced_span(tracer: otel_trace.Tracer, name: str, attributes: Optional[Dict] = None):
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                safe_set(span, k, v)
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(StatusCode.ERROR, str(e))
            raise

def extract_parent_context(trace_metadata: Optional[Dict]) -> Optional[SpanContext]:
    """Safely extracts a parent SpanContext from a dictionary."""
    if not trace_metadata: return None
    trace_id_str, parent_span_id_str = trace_metadata.get("trace_id"), trace_metadata.get("parent_span_id")
    if not (trace_id_str and parent_span_id_str): return None
    try:
        return SpanContext(
            trace_id=int(trace_id_str, 16),
            span_id=int(parent_span_id_str, 16),
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED)
        )
    except (ValueError, TypeError):
        logging.warning("Could not parse incoming trace_metadata.")
        return None