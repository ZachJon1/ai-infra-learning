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


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Load once per application worker
#     app.state.model = TinyRegressionModel()
#     app.state.model.eval()

#     print("PyTorch model loaded successfully")

#     yield

#     # Release model on shutdown
#     del app.state.model

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None

    if os.getenv("SIMULATE_MODEL_UNAVAILABLE") != "1":
        model = TinyRegressionModel()
        model.eval()
        app.state.model = model

        print("Model loaded successfully")
    else:
        print("SIMULATION: Model unavailable")

    yield

    app.state.model = None

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
    prediction: float
    model_version: str


@app.get("/health")
def health():
    return {"status": "healthy"}


# @app.get("/ready")
# def ready(request: Request):
#     if not hasattr(request.app.state, "model"):
#         raise HTTPException(
#             status_code=503,
#             detail="Model not loaded",
#         )

#     return {"status": "ready"}

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
    model = getattr(request.app.state, "model", None)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model unavailable",
        )

    features = torch.tensor(
        [payload.features],
        dtype=torch.float32,
    )

    with torch.inference_mode():
        prediction = model(features)

    return PredictionResponse(
        prediction=round(float(prediction.item()), 6),
        model_version="demo-linear-v1",
    )