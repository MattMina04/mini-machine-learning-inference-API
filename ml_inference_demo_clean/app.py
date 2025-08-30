# app.py (≈85 lines) — minimal Flask inference API
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Lazy-load the model at startup
MODEL_PATH = "model.joblib"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Warning: could not load model from {MODEL_PATH}: {e}")

@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200

@app.post("/predict")
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please run train.py first."}), 500

    payload = request.get_json(silent=True)
    if not payload or "text" not in payload:
        return jsonify({"error": "Missing 'text' in JSON body"}), 400

    text = str(payload["text"]).strip()
    if not text:
        return jsonify({"error": "Empty 'text' value"}), 400

    # Predict label and probability
    proba = getattr(model, "predict_proba", None)
    if proba is None:
        pred = model.predict([text])[0]
        return jsonify({"label": pred, "probability": None}), 200

    probs = model.predict_proba([text])[0]
    idx = int(np.argmax(probs))
    label = model.classes_[idx]
    confidence = float(probs[idx])

    return jsonify({"label": label, "probability": round(confidence, 4)}), 200

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
