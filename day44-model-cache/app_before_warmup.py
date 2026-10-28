import os
import time
import threading
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODE = os.getenv("LOAD_MODE", "lazy").lower()

if MODE not in ("eager", "lazy"):
    raise ValueError("LOAD_MODE must be eager or lazy")


def create_model():
    return torch.nn.Sequential(
        torch.nn.Linear(2048, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 8),
    )


class ModelCache:
    def __init__(self):
        self.model = None
        self.lock = threading.Lock()
        self.load_count = 0
        self.load_ms = None

    def get_model(self):
        if self.model is None:
            with self.lock:
                if self.model is None:
                    start = time.perf_counter()

                    weights = torch.load(
                        "model_weights.pt",
                        map_location="cpu",
                        weights_only=True,
                    )

                    model = create_model()
                    model.load_state_dict(weights)
                    model = model.to(DEVICE)
                    model.eval()

                    self.model = model
                    self.load_count += 1
                    self.load_ms = (
                        time.perf_counter() - start
                    ) * 1000

                    print(
                        f"Model loaded in "
                        f"{self.load_ms:.2f} ms",
                        flush=True,
                    )

        return self.model


cache = ModelCache()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODE == "eager":
        cache.get_model()
    yield
    cache.model = None


app = FastAPI(lifespan=lifespan)


class PredictInput(BaseModel):
    scale: float = 1.0


@app.get("/status")
def status():
    return {
        "mode": MODE,
        "device": DEVICE,
        "loaded": cache.model is not None,
        "load_count": cache.load_count,
        "load_ms": cache.load_ms,
    }


@app.post("/predict")
def predict(request: PredictInput):
    start = time.perf_counter()
    model = cache.get_model()
    loading_wait_ms = (
        time.perf_counter() - start
    ) * 1000

    start = time.perf_counter()

    x = torch.full(
        (1, 2048),
        request.scale,
        device=DEVICE,
    )

    with torch.inference_mode():
        output = model(x)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    inference_ms = (
        time.perf_counter() - start
    ) * 1000

    return {
        "loading_wait_ms": round(loading_wait_ms, 3),
        "inference_ms": round(inference_ms, 3),
        "load_count": cache.load_count,
        "prediction": output[0].tolist(),
    }
