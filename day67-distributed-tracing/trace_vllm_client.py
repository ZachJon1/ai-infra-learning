"""Instrument one HTTP request and propagate W3C trace context to vLLM."""
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

provider = TracerProvider(resource=Resource.create({"service.name": "ai-infra-client"}))
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="http://127.0.0.1:4318/v1/traces")))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("day67.vllm-client")
payload = {
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Explain GPU memory in one sentence."}],
    "max_tokens": 32, "temperature": 0.0, "stream": False,
}
try:
    with tracer.start_as_current_span("client.chat_completion", kind=SpanKind.CLIENT) as span:
        span.set_attribute("model.name", payload["model"])
        span.set_attribute("request.max_tokens", 32)
        headers = {"Content-Type": "application/json"}
        TraceContextTextMapPropagator().inject(headers)
        trace_id = format(span.get_span_context().trace_id, "032x")
        print("Trace ID:", trace_id)
        print("traceparent:", headers.get("traceparent"))
        request = urllib.request.Request(
            "http://127.0.0.1:8000/v1/chat/completions",
            data=json.dumps(payload).encode(), headers=headers, method="POST",
        )
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
            print("HTTP status:", response.status)
        elapsed = time.perf_counter() - start
        span.set_attribute("client.duration_seconds", elapsed)
        print(f"Client elapsed time: {elapsed:.3f} seconds")
        print("Model response:", result["choices"][0]["message"]["content"])
finally:
    print("Trace flush:", provider.force_flush(timeout_millis=10000))
    provider.shutdown()
