import os
import time
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "unknown-model")
MODEL_VERSION = os.getenv("MODEL_VERSION", "unknown")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
API_KEY = os.getenv("API_KEY", "not-set")

model_ready = False


def load_model():
    global model_ready

    print(
        f"Loading model={MODEL_NAME}, "
        f"version={MODEL_VERSION}, "
        f"batch_size={BATCH_SIZE}"
    )

    time.sleep(5)

    model_ready = True
    print("Model loaded successfully")


load_model()


@app.route("/healthz")
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/readyz")
def ready():
    if model_ready:
        return jsonify({"status": "ready"}), 200

    return jsonify({"status": "not ready"}), 503


@app.route("/predict", methods=["POST"])
def predict():

    provided_key = request.headers.get("X-API-Key")

    if provided_key != API_KEY:
        return jsonify({"error": "unauthorized"}), 401

    payload = request.get_json(silent=True) or {}
    value = payload.get("value", 0)

    # Simulated inference
    prediction = value * 2

    return jsonify(
        {
            "model": MODEL_NAME,
            "version": MODEL_VERSION,
            "prediction": prediction,
        }
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080
    )
