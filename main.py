import os
import joblib   # ✅ use joblib instead of pickle
import logging
import secrets
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Any, List

import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sys
import sklearn

# =============================================================================
# 1. Configuration and Setup
# =============================================================================

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("rockfall_api")

MODEL_FILENAME = "full_rockfall_pipeline.pkl"
model: Optional[Any] = None

API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "X-API-Key"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# =============================================================================
# 2. Pydantic Models
# =============================================================================

class RockfallFeatures(BaseModel):
    rock_type: str
    geological_structure: str
    joint_spacing_cm: float
    weathering_grade: str
    presence_of_clay_seams: int = Field(..., ge=0, le=1)
    slope_angle_degrees: float
    bench_height_m: float
    slope_aspect_degrees: float
    bench_width_m: float
    annual_rainfall_mm: float
    pore_water_pressure_kPa: float
    seepage_presence: int = Field(..., ge=0, le=1)
    seismic_activity: int = Field(..., ge=0, le=1)
    blast_vibration_intensity_mm_s: float
    blast_frequency_per_month: int
    time_since_last_rockfall_days: int

class PredictResponse(BaseModel):
    prediction: List[Any]

# =============================================================================
# 3. Security Dependency
# =============================================================================

async def get_api_key(api_key: str = Security(api_key_header)):
    if API_KEY is None:
        logger.error("API_KEY environment variable not set on the server.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API Key not configured on server",
        )
    if secrets.compare_digest(api_key, API_KEY):
        return api_key
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )

# =============================================================================
# 4. Application Lifespan (Model Loading)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    p = Path(__file__).resolve().parent / MODEL_FILENAME
    if not p.exists():
        logger.error("Model file not found at %s. Prediction endpoint will be disabled.", p)
        model = None
    else:
        try:
            model = joblib.load(p)   # ✅ FIXED: use joblib.load
            logger.info("Model loaded successfully from %s", p)
        except Exception as exc:
            logger.exception("Failed to load model: %s", exc)
            model = None

    if not API_KEY:
        logger.critical("CRITICAL: API_KEY environment variable is not set. API will be insecure.")

    yield

    model = None
    logger.info("Model released and resources cleaned up on shutdown.")

# =============================================================================
# 5. FastAPI App Instance and Routes
# =============================================================================

app = FastAPI(
    title="Rockfall Prediction API",
    version="1.2",
    lifespan=lifespan,
    description="An API to predict rockfall probability based on geological and environmental factors."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["General"])
def home():
    return {"status": "ok", "message": "Rockfall Prediction API is running!"}

@app.get("/health", tags=["General"])
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "python": sys.version,
        "sklearn": sklearn.__version__
    }

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(payload: RockfallFeatures, api_key: str = Depends(get_api_key)):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Cannot perform predictions.",
        )
    try:
        df = pd.DataFrame([payload.model_dump()])
        prediction = model.predict(df)
        result = prediction.tolist() if hasattr(prediction, 'tolist') else [prediction]
        return {"prediction": result}
    except Exception as exc:
        logger.exception("Prediction failed for payload: %s", payload.model_dump_json())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during prediction: {exc}",
        )

# =============================================================================
# 6. Local Server Run
# =============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8002"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
