from contextlib import asynccontextmanager
import os

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
)

from model import TinyRegressionModel

@asynccontextmanager
async def lifespan(app: FastAPI):

    require_cuda = os.getenv("REQUIRE_CUDA", "0") == "1"

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU inference required, but CUDA is unavailable"
        )

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    app.state.device = device
    app.state.model = TinyRegressionModel().to(device)
    app.state.model.eval()

    print(f"Inference device: {device}")

    # Application starts serving requests
    yield

    # Application shutdown
    del app.state.model
    print("PyTorch model released")


app = FastAPI(
    title="AI Infrastructure Inference API",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: list[FiniteFloat] = Field(
        min_length=3,
        max_length=3,
    )


class PredictionResponse(BaseModel):
    prediction_value: float
    model_version: str


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/ready")
def ready(request: Request):
    model = getattr(request.app.state, "model", None)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not ready",
        )

    return {"status": "ready"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request):

    model = request.app.state.model
    device = request.app.state.device

    features = torch.tensor(
        [payload.features],
        dtype=torch.float32,
        device=device,
    )

    with torch.inference_mode():
        prediction = model(features)

    prediction_value = round(
        float(prediction.item()), 6
    )

    return PredictionResponse(
        prediction_value=prediction_value,
        model_version="demo-linear-v1",
    )
