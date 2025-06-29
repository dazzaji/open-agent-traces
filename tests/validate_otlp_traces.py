#!/usr/bin/env python3
"""
Validate OTLP trace JSON files against OpenTelemetry specifications.
This serves as our "otel-validate" tool for the project.
"""
import json
import sys
import argparse
from typing import Dict, List, Any


def validate_trace_id(trace_id: str) -> bool:
    """Validate trace ID is 32 hex characters."""
    if not isinstance(trace_id, str):
        return False
    if len(trace_id) != 32:
        return False
    try:
        int(trace_id, 16)
        return True
    except ValueError:
        return False


def validate_span_id(span_id: str) -> bool:
    """Validate span ID is 16 hex characters."""
    if not isinstance(span_id, str):
        return False
    if len(span_id) != 16:
        return False
    try:
        int(span_id, 16)
        return True
    except ValueError:
        return False


def validate_span(span: Dict[str, Any], errors: List[str]) -> bool:
    """Validate a single span structure."""
    is_valid = True
    
    # Required fields
    if "traceId" not in span:
        errors.append("Span missing required field: traceId")
        is_valid = False
    elif not validate_trace_id(span["traceId"]):
        errors.append(f"Invalid traceId format: {span.get('traceId')}")
        is_valid = False
    
    if "spanId" not in span:
        errors.append("Span missing required field: spanId")
        is_valid = False
    elif not validate_span_id(span["spanId"]):
        errors.append(f"Invalid spanId format: {span.get('spanId')}")
        is_valid = False
    
    if "name" not in span:
        errors.append("Span missing required field: name")
        is_valid = False
    
    # Validate parentSpanId if present
    if "parentSpanId" in span and span["parentSpanId"]:
        if not validate_span_id(span["parentSpanId"]):
            errors.append(f"Invalid parentSpanId format: {span['parentSpanId']}")
            is_valid = False
    
    # Validate timestamps
    for ts_field in ["startTimeUnixNano", "endTimeUnixNano"]:
        if ts_field in span:
            if not isinstance(span[ts_field], (int, str)):
                errors.append(f"Invalid {ts_field} type: should be numeric")
                is_valid = False
    
    # Validate attributes
    if "attributes" in span:
        if not isinstance(span["attributes"], list):
            errors.append("Span attributes should be a list")
            is_valid = False
        else:
            for attr in span["attributes"]:
                if not isinstance(attr, dict):
                    errors.append("Each attribute should be a dict")
                    is_valid = False
                elif "key" not in attr:
                    errors.append("Attribute missing key")
                    is_valid = False
                elif "value" not in attr:
                    errors.append("Attribute missing value")
                    is_valid = False
    
    # Validate events
    if "events" in span:
        if not isinstance(span["events"], list):
            errors.append("Span events should be a list")
            is_valid = False
        else:
            for event in span["events"]:
                if not isinstance(event, dict):
                    errors.append("Each event should be a dict")
                    is_valid = False
                elif "name" not in event:
                    errors.append("Event missing name")
                    is_valid = False
    
    # Validate links
    if "links" in span:
        if not isinstance(span["links"], list):
            errors.append("Span links should be a list")
            is_valid = False
        else:
            for link in span["links"]:
                if "traceId" not in link:
                    errors.append("Link missing traceId")
                    is_valid = False
                elif not validate_trace_id(link["traceId"]):
                    errors.append(f"Invalid link traceId: {link['traceId']}")
                    is_valid = False
                if "spanId" not in link:
                    errors.append("Link missing spanId")
                    is_valid = False
                elif not validate_span_id(link["spanId"]):
                    errors.append(f"Invalid link spanId: {link['spanId']}")
                    is_valid = False
    
    return is_valid


def validate_otlp_json(data: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Validate OTLP JSON structure."""
    errors = []
    is_valid = True
    
    # Check for resourceSpans
    if "resourceSpans" not in data:
        errors.append("Missing required field: resourceSpans")
        return False, errors
    
    if not isinstance(data["resourceSpans"], list):
        errors.append("resourceSpans must be a list")
        return False, errors
    
    # Validate each resource span
    for i, resource_span in enumerate(data["resourceSpans"]):
        # Check resource
        if "resource" in resource_span:
            resource = resource_span["resource"]
            if "attributes" in resource:
                if not isinstance(resource["attributes"], list):
                    errors.append(f"resourceSpans[{i}].resource.attributes must be a list")
                    is_valid = False
        
        # Check scopeSpans
        if "scopeSpans" not in resource_span:
            errors.append(f"resourceSpans[{i}] missing scopeSpans")
            is_valid = False
            continue
        
        if not isinstance(resource_span["scopeSpans"], list):
            errors.append(f"resourceSpans[{i}].scopeSpans must be a list")
            is_valid = False
            continue
        
        # Validate each scope span
        for j, scope_span in enumerate(resource_span["scopeSpans"]):
            # Check spans
            if "spans" not in scope_span:
                errors.append(f"resourceSpans[{i}].scopeSpans[{j}] missing spans")
                is_valid = False
                continue
            
            if not isinstance(scope_span["spans"], list):
                errors.append(f"resourceSpans[{i}].scopeSpans[{j}].spans must be a list")
                is_valid = False
                continue
            
            # Validate each span
            for k, span in enumerate(scope_span["spans"]):
                span_errors = []
                if not validate_span(span, span_errors):
                    is_valid = False
                    for error in span_errors:
                        errors.append(f"resourceSpans[{i}].scopeSpans[{j}].spans[{k}]: {error}")
    
    # Check for our custom requirements
    if is_valid:
        # Verify at least one span has events
        has_events = False
        for rs in data["resourceSpans"]:
            for ss in rs.get("scopeSpans", []):
                for span in ss.get("spans", []):
                    if span.get("events"):
                        has_events = True
                        break
        
        if not has_events:
            errors.append("Warning: No spans contain events (expected full_prompt/full_response)")
        
        # Check for links
        has_links = False
        for rs in data["resourceSpans"]:
            for ss in rs.get("scopeSpans", []):
                for span in ss.get("spans", []):
                    if span.get("links"):
                        has_links = True
                        break
        
        if not has_links:
            errors.append("Info: No spans contain links (OpenAI trace link may be missing)")
    
    return is_valid, errors


def main():
    parser = argparse.ArgumentParser(description="Validate OTLP trace JSON files")
    parser.add_argument("files", nargs="+", help="JSON files to validate")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    all_valid = True
    
    for filename in args.files:
        print(f"\nValidating {filename}...")
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"  ❌ File not found: {filename}")
            all_valid = False
            continue
        except json.JSONDecodeError as e:
            print(f"  ❌ Invalid JSON: {e}")
            all_valid = False
            continue
        
        is_valid, errors = validate_otlp_json(data)
        
        if is_valid and not errors:
            print(f"  ✅ Valid OTLP trace file")
            if args.verbose:
                # Print summary statistics
                total_spans = 0
                total_events = 0
                total_links = 0
                for rs in data.get("resourceSpans", []):
                    for ss in rs.get("scopeSpans", []):
                        spans = ss.get("spans", [])
                        total_spans += len(spans)
                        for span in spans:
                            total_events += len(span.get("events", []))
                            total_links += len(span.get("links", []))
                
                print(f"    - Total spans: {total_spans}")
                print(f"    - Total events: {total_events}")
                print(f"    - Total links: {total_links}")
        else:
            if is_valid and errors:
                # Just warnings/info
                print(f"  ⚠️  Valid with warnings:")
            else:
                print(f"  ❌ Invalid OTLP trace file:")
                all_valid = False
            
            for error in errors:
                print(f"    - {error}")
    
    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()