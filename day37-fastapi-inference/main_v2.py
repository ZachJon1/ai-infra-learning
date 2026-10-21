from fastapi import FastAPI
from pydantic import (BaseModel,
                      ConfigDict,
                      Field,
                      FiniteFloat,)

app = FastAPI(title="AI infra inference APi")

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

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    "temp calc, not an ML pred"
    score = sum(request.features)
    
    return PredictionResponse(
        prediction=score,
        model_version="baseline-sum-v0",
    )