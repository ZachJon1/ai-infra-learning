from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AI infra inference APi")

class PredictionRequest(BaseModel):
    features: list[float]
    
@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(request: PredictionRequest):
    "temp calc, not an ML pred"
    score = sum(request.features)
    
    return {"prediction": score}