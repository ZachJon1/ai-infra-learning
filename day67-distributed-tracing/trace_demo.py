"""Three artificial inference traces; no GPU is used."""
import time
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

provider = TracerProvider(resource=Resource.create({"service.name": "ai-infra-trace-demo", "deployment.environment.name": "dev"}))
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="http://127.0.0.1:4318/v1/traces")))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("ai-infra.day67")

def simulated_request(number: int, queue_delay: float, model_delay: float):
    with tracer.start_as_current_span("inference.request") as root:
        root.set_attribute("request.number", number)
        root.set_attribute("model.name", "simulated-model")
        trace_id = format(root.get_span_context().trace_id, "032x")
        print(f"Request {number}: trace_id={trace_id}")
        with tracer.start_as_current_span("request.validate"):
            time.sleep(0.02)
        with tracer.start_as_current_span("queue.wait") as span:
            span.set_attribute("queue.simulated", True)
            time.sleep(queue_delay)
        with tracer.start_as_current_span("model.generate.simulated") as span:
            span.set_attribute("model.gpu_used", False)
            span.add_event("generation_started")
            time.sleep(model_delay)
            span.add_event("generation_completed")
        with tracer.start_as_current_span("response.serialize"):
            time.sleep(0.01)
        root.set_attribute("request.status", "success")

for args in [(1, 0.04, 0.13), (2, 0.12, 0.36), (3, 0.07, 0.20)]:
    simulated_request(*args)
print("SDK flush completed:", provider.force_flush(timeout_millis=10000))
provider.shutdown()
