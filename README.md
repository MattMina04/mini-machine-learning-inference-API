# Mini ML Inference API (Flask + scikit-learn)

A compact example that integrates a **machine learning model** into a running **inference server** (Flask).
- **<lines of Python total** across `train.py` and `app.py`.
- Simple **text classifier** (spam vs ham) built with `scikit-learn` (`TfidfVectorizer` + `LogisticRegression`).
- REST endpoint: `POST /predict` that returns a JSON label and probability.
- Health check: `GET /health`.

## Why this project
- End-to-end mindset in a tiny package: **train → persist → serve → predict**.
- Clear, concise code that’s easy to review and reason about.
- Includes basic input validation and example requests.

What it is

A tiny spam vs ham text classifier (like detecting whether a message is junk or normal).

Built in Python using scikit-learn.

Served through a Flask API so you can send a request and get back a prediction.
## Project Structure
```
.
  ├─ app.py           # Flask inference server
  ├─ train.py         # Trains and saves the model pipeline
  ├─ requirements.txt # Minimal deps
  └─ README.md
```

## Quickstart
1) Create and activate a virtual environment (optional):
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2) Install dependencies:
```bash
pip install -r requirements.txt
```

3) Train the model (writes `model.joblib`):
```bash
python train.py
```

4) Start the API:
```bash
python app.py
```

5) Test a prediction:
```bash
curl -s -X POST http://127.0.0.1:5000/predict   -H "Content-Type: application/json"   -d '{"text": "WIN a free iPhone, click this link!"}'
```

**Response example:**
```json
{"label":"spam","probability":0.93}
```

## API
- `GET /health` → `{"status":"ok"}`
- `POST /predict` with payload:
```json
{"text": "your message here"}
```
Returns:
```json
{"label": "spam" | "ham", "probability": 0.0-1.0}
```

## Notes
- Data is a tiny, embedded toy set to keep the code self-contained for review.
- For a real system, you’d externalise data, add authentication, request logging, monitoring, CI/CD, and containerisation.
- Code is intentionally **short and readable**.

---

**Author:** Matthew Mina Mikhail
