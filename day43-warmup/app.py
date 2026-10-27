import time
import torch
from torch import nn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

@asynccontextmanager
async def lifespan(app: FastAPI):
    start = time.perf_counter()
    app.state.ready = False

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024)
    ).to(device).eval()

    x = torch.randn(32, 1024, device=device)

    # Model warmup
    with torch.inference_mode():
        for _ in range(5):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    app.state.model = model
    app.state.inputs = x
    app.state.ready = True

    elapsed = time.perf_counter() - start

    print(f"Device: {device}", flush=True)
    print(f"Model ready after {elapsed:.3f}s", flush=True)

    yield

    app.state.ready = False
    del app.state.model
    del app.state.inputs


app = FastAPI(lifespan=lifespan)


@app.get("/live")
def live():
    return {"status": "alive"}


@app.get("/ready")
def ready():
    if not app.state.ready:
        raise HTTPException(
            status_code=503,
            detail="Model not ready"
        )

    return {"status": "ready"}


@app.get("/predict")
def predict():
    if not app.state.ready:
        raise HTTPException(
            status_code=503,
            detail="Model not ready"
        )

    with torch.inference_mode():
        output = app.state.model(app.state.inputs)

    return {"prediction": float(output.mean().item())}