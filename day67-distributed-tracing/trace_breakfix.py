"""Inject a controlled client-side delay, then compare Jaeger traces."""
import argparse
import json
import time
import urllib.request
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.trace import SpanKind
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

parser = argparse.ArgumentParser()
parser.add_argument("--delay", type=float, default=0.0)
args = parser.parse_args()
if not 0 <= args.delay <= 5:
    parser.error("Delay must be between 0 and 5 seconds")
provider = TracerProvider(resource=Resource.create({"service.name": "ai-infra-client"}))
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="http://127.0.0.1:4318/v1/traces")))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("day67.breakfix")
payload = {
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Explain GPU memory in one sentence."}],
    "max_tokens": 32, "temperature": 0.0, "stream": False,
}
start = time.perf_counter()
try:
    with tracer.start_as_current_span("client.inference_workflow") as root:
        trace_id = format(root.get_span_context().trace_id, "032x")
        with tracer.start_as_current_span("client.preprocessing") as span:
            span.set_attribute("test.injected_delay_seconds", args.delay)
            time.sleep(args.delay)
        with tracer.start_as_current_span("client.http_request", kind=SpanKind.CLIENT) as span:
            headers = {"Content-Type": "application/json"}
            TraceContextTextMapPropagator().inject(headers)
            request = urllib.request.Request(
                "http://127.0.0.1:8000/v1/chat/completions",
                data=json.dumps(payload).encode(), headers=headers, method="POST",
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.load(response)
                span.set_attribute("http.response.status_code", response.status)
    elapsed = time.perf_counter() - start
    print(f"Trace ID: {trace_id}")
    print(f"Injected delay: {args.delay:.3f} s")
    print(f"Total workflow duration: {elapsed:.3f} s")
    print("Model response:", result["choices"][0]["message"]["content"])
finally:
    print("Trace flush:", provider.force_flush(timeout_millis=10000))
    provider.shutdown()
