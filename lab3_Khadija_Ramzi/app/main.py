### main.py

#Import Libraries
import os
import joblib
import pandas as pd
import xgboost as xgb
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum

# Set up Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
    handlers=[
        # Write to file
        logging.FileHandler("app/logs/app.log"),   
        # Show in console
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# File paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "model.json")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "data", "label_encoder.pkl")
COLUMNS_PATH = os.path.join(os.path.dirname(__file__), "data", "columns.pkl")

# Load model
try:
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    feature_columns = joblib.load(COLUMNS_PATH)
    logger.info("SUCCESS: Model, label encoder, and column list loaded.")
except Exception as e:
    logger.error(f"Failed to load model artifacts: {e}")
    raise RuntimeError("ERROR: Could not load model or preprocessing artifacts.")

# FastAPI app
app = FastAPI()

# Define Enums for categorical features
class Island(str, Enum):
    Torgersen = "Torgersen"
    Biscoe = "Biscoe"
    Dream = "Dream"

class Sex(str, Enum):
    Male = "male"
    Female = "female"

# Pydantic Model for input data
class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    year: int
    sex: Sex
    island: Island

# Preprocessing to match training
def preprocess_features(features: PenguinFeatures) -> pd.DataFrame:
    input_dict = features.model_dump()
    df = pd.DataFrame([input_dict])
    df = pd.get_dummies(df, columns=["sex", "island"])

    # Add missing columns and keep order
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]
    return df.astype(float)

## Define Endpoints

# Root/Index Endpoint
@app.get("/")
async def root():
    return {"message": "Penguin Predictor API is running."}

# Health Endpoint
@app.get("/health")
async def health():
    return {"status": "ok"}

# Predict Endpoint
@app.post("/predict")
async def predict(features: PenguinFeatures):
    try:
        logger.info(f"Received input: {features.model_dump()}")
        X_input = preprocess_features(features)
        pred = model.predict(X_input.values)
        species = label_encoder.inverse_transform(pred)[0]
        logger.info(f"Prediction successful: {species}")
        return {"prediction": species}
    except Exception as e:
        logger.debug(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
