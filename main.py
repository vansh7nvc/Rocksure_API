from fastapi import FastAPI, Depends, HTTPException, status, Security
from fastapi.security.api_key import APIKeyHeader
import pickle
import pandas as pd
from typing import Optional
import os

# ✅ This must exist at the top level so ASGI servers can import it
app = FastAPI(title="Rockfall Prediction API")

# --- API key setup -------------------------------------------------
# Use header name 'access_token' by convention in this project
API_KEY_NAME = "access_token"
# Prefer environment variable for secrets; fallback to a placeholder for dev
API_KEY = os.getenv("API_KEY", "changeme")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key: Optional[str] = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )

# --- Model handling ------------------------------------------------
# We'll load the model during startup to avoid import-time failures that
# prevent ASGI servers from finding `app`.
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        with open("full_rockfall_pipeline.pkl", "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        # Keep model as None; endpoints will return 503 until it's available
        model = None
    except Exception as e:
        # Log/raise as needed, but don't break import-time
        model = None

# Home endpoint
@app.get("/")
def home():
    return {"message": "Rockfall Prediction API is running!"}

# Prediction endpoint
@app.post("/predict")
def predict(data: dict, api_key: str = Depends(get_api_key)):
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Model not loaded")

    df = pd.DataFrame([data])  # Convert JSON → DataFrame
    prediction = model.predict(df)
    # Ensure JSON serializable
    return {"prediction": prediction.tolist()}


if __name__ == "__main__":
    # Allow running locally with: python main.py
    # This block won't run when an ASGI server imports `app`.
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")

