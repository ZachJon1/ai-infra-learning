"""Day 61 educational inference gateway; single process, no streaming support."""
import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import Counter, Gauge, Histogram, CONTENT_TYPE_LATEST, generate_latest

UPSTREAM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8000/v1/chat/completions")

REQUESTS = Counter(
    "app_inference_requests_total",
    "Completed inference requests",
    ["status_code"],
)
INFLIGHT = Gauge(
    "app_inference_inflight_requests",
    "Currently executing inference requests",
)
DURATION = Histogram(
    "app_inference_request_duration_seconds",
    "Inference request duration in seconds",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
        app.state.http_client = client
        yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), headers={"Content-Type": CONTENT_TYPE_LATEST})

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    start = time.perf_counter()
    status_code = 500
    INFLIGHT.inc()
    try:
        upstream = await request.app.state.http_client.post(
            UPSTREAM_URL,
            content=await request.body(),
            headers={"content-type": request.headers.get("content-type", "application/json")},
        )
        status_code = upstream.status_code
        return Response(
            content=upstream.content,
            status_code=status_code,
            headers={"content-type": upstream.headers.get("content-type", "application/json")},
        )
    except httpx.TimeoutException:
        status_code = 504
        return JSONResponse(status_code=504, content={"error": "Upstream timeout"})
    except httpx.RequestError:
        status_code = 502
        return JSONResponse(status_code=502, content={"error": "Upstream connection failure"})
    finally:
        REQUESTS.labels(status_code=str(status_code)).inc()
        DURATION.observe(time.perf_counter() - start)
        INFLIGHT.dec()
