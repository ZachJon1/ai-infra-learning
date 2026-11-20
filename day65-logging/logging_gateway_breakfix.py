"""Day 65 reference application: basic structured logging inference gateway.

Lab use only: avoid using as a public production service without authentication,
input validation, rate limiting, bounded concurrency and better error handling.
"""
import json
import os
import logging
import sys
import time
import uuid
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response


class JSONFormatter(logging.Formatter):
    def format(self, record):
        entry = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "service": "inference-gateway",
            "event": record.getMessage(),
        }
        for field in (
            "request_id", "method", "path", "status_code",
            "duration_ms", "upstream_status", "error_type",
        ):
            if hasattr(record, field):
                entry[field] = getattr(record, field)
        return json.dumps(entry)


logger = logging.getLogger("inference-gateway")
logger.setLevel(logging.INFO)
logger.propagate = False
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)

app = FastAPI()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.error("unhandled_exception", extra={"request_id": request_id, "error_type": type(exc).__name__})
        response = JSONResponse(status_code=500, content={"detail": "Internal server error"})
    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    level = logging.ERROR if response.status_code >= 500 else logging.WARNING if response.status_code >= 400 else logging.INFO
    logger.log(level, "http_request_completed", extra={
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_ms": duration_ms,
    })
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/simulate-error")
async def simulate_error():
    return JSONResponse(status_code=503, content={"detail": "Simulated service outage"})


@app.post("/v1/chat/completions")
async def inference(request: Request):
    payload = await request.json()
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            upstream = await client.post(os.environ.get("VLLM_URL", "http://127.0.0.1:8000/v1/chat/completions"), json=payload)
    except httpx.RequestError as exc:
        logger.error("upstream_connection_failed", extra={
            "request_id": request.state.request_id,
            "error_type": type(exc).__name__,
        })
        raise HTTPException(status_code=502, detail="Inference backend unavailable")
    if upstream.status_code >= 400:
        logger.warning("upstream_http_error", extra={
            "request_id": request.state.request_id,
            "upstream_status": upstream.status_code,
        })
    return Response(content=upstream.content, status_code=upstream.status_code, media_type="application/json")
