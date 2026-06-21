"""
Attrition Insight — FastAPI backend
====================================

HOW TO LOAD YOUR OWN MODEL — THE ONLY THING YOU NEED TO CHANGE
----------------------------------------------------------------
1. Put your trained model file in this `backend/` folder (or update MODEL_PATH).
   Supported out of the box: anything saved with `joblib.dump(...)` or
   `pickle.dump(...)` that exposes a scikit-learn-style API
   (`.predict_proba()` or `.predict()`).

2. Set MODEL_PATH below to your filename, e.g. "model.pkl" or "attrition_model.joblib"

3. Check FEATURE_ORDER and the *_ENCODING maps further down — they must
   match the exact column order / encoding your model was TRAINED on.

4. Run it:
       pip install -r requirements.txt
       uvicorn main:app --reload --port 8000
"""

import os
import pickle
import logging
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import mlflow.pyfunc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("attrition-insight")

# ============================================================================
# 1. CONFIG — change this block for your own model
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# >>> CHANGE THIS to your model's filename <
MODEL_PATH = os.path.join(BASE_DIR, "production_model")

# Set to True once you've dropped in a real model. While False, the API
# uses a transparent placeholder scorer so the whole app works end-to-end
# even before you have a trained model.
USE_PLACEHOLDER_IF_MISSING = True

# ============================================================================
# 2. REQUEST SCHEMA — mirrors the form fields exactly
# ============================================================================

class EmployeeProfile(BaseModel):
    Age: int = Field(..., alias="Age", ge=0)
    Gender: str = Field(..., description="Male | Female")
    Years_at_Company: int = Field(..., alias="Years at Company", ge=0)
    Job_Role: str = Field(..., alias="Job Role")
    Monthly_Income: int = Field(..., alias="Monthly Income", ge=0)
    Work_Life_Balance: str = Field(..., alias="Work-Life Balance")
    Job_Satisfaction: str = Field(..., alias="Job Satisfaction")
    Performance_Rating: str = Field(..., alias="Performance Rating")
    Number_of_Promotions: int = Field(..., alias="Number of Promotions", ge=0)
    Overtime: str
    Distance_from_Home: int = Field(..., alias="Distance from Home", ge=0)
    Education_Level: str = Field(..., alias="Education Level")
    Marital_Status: str = Field(..., alias="Marital Status")
    Number_of_Dependents: int = Field(..., alias="Number of Dependents", ge=0)
    Job_Level: str = Field(..., alias="Job Level")
    Company_Size: str = Field(..., alias="Company Size")
    Company_Tenure: int = Field(..., alias="Company Tenure", ge=0)
    Remote_Work: str = Field(..., alias="Remote Work")
    Leadership_Opportunities: str = Field(..., alias="Leadership Opportunities")
    Innovation_Opportunities: str = Field(..., alias="Innovation Opportunities")
    Company_Reputation: str = Field(..., alias="Company Reputation")
    Employee_Recognition: str = Field(..., alias="Employee Recognition")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Age": 22,
                "Gender": "Female",
                "Years at Company": 3,
                "Job Role": "Technology",
                "Monthly Income": 4200,
                "Work-Life Balance": "Fair",
                "Job Satisfaction": "Low",
                "Performance Rating": "Average",
                "Number of Promotions": 0,
                "Overtime": "Yes",
                "Distance from Home": 28,
                "Education Level": "Bachelor's Degree",
                "Marital Status": "Single",
                "Number of Dependents": 0,
                "Job Level": "Entry",
                "Company Size": "Large",
                "Company Tenure": 3,
                "Remote Work": "No",
                "Leadership Opportunities": "No",
                "Innovation Opportunities": "No",
                "Company Reputation": "Fair",
                "Employee Recognition": "Low",
            }
        }


class PredictionResponse(BaseModel):
    prediction: str            # "Stayed" or "Left"
    probability: float         # 0.0 - 1.0, probability of "Left"
    top_factors: Optional[List[Dict[str, Any]]] = None


# ============================================================================
# 3. ENCODING — must match how your model was trained
# ============================================================================

GENDER_MAP = {"Male": 0, "Female": 1}

JOB_ROLE_MAP = {"Education": 0, "Media": 1, "HealthCare": 2, "Technology": 3, "Finance": 4}

WORK_LIFE_BALANCE_MAP = {"Poor": 0, "Fair": 1, "Good": 2, "Excellent": 3}

JOB_SATISFACTION_MAP = {"Low": 0, "Medium": 1, "High": 2, "Very High": 3}

PERFORMANCE_RATING_MAP = {"Low": 0, "Below Average": 1, "Average": 2, "High": 3}

YES_NO_MAP = {"No": 0, "Yes": 1}  # used for Overtime, Remote Work, Leadership/Innovation Opportunities

EDUCATION_LEVEL_MAP = {
    "High School": 0,
    "Associate Degree": 1,
    "Bachelor's Degree": 2,
    "Master's Degree": 3,
    "PhD": 4,
}

MARITAL_STATUS_MAP = {"Single": 0, "Married": 1, "Divorced": 2}

JOB_LEVEL_MAP = {"Entry": 0, "Mid": 1, "Senior": 2}

COMPANY_SIZE_MAP = {"Small": 0, "Medium": 1, "Large": 2}

COMPANY_REPUTATION_MAP = {"Poor": 0, "Fair": 1, "Good": 2, "Excellent": 3}

EMPLOYEE_RECOGNITION_MAP = {"Low": 0, "Medium": 1, "High": 2, "Very High": 3}

# Final column order fed into the model. If your model was trained on a
# pandas DataFrame, this MUST match `X_train.columns` order exactly.
FEATURE_ORDER = [
    "Age",
    "Gender",
    "Years at Company",
    "Job Role",
    "Monthly Income",
    "Work-Life Balance",
    "Job Satisfaction",
    "Performance Rating",
    "Number of Promotions",
    "Overtime",
    "Distance from Home",
    "Education Level",
    "Marital Status",
    "Number of Dependents",
    "Job Level",
    "Company Size",
    "Company Tenure",
    "Remote Work",
    "Leadership Opportunities",
    "Innovation Opportunities",
    "Company Reputation",
    "Employee Recognition",
]


def encode_features(profile: EmployeeProfile) -> pd.DataFrame:
    """
    Converts a validated EmployeeProfile into a single-row DataFrame with
    numerically encoded categorical features, in FEATURE_ORDER.

    If your model expects one-hot encoded columns instead of ordinal codes,
    replace the body of this function with e.g.:

        raw = pd.DataFrame([{...raw string values...}])
        encoded = pd.get_dummies(raw)
        encoded = encoded.reindex(columns=TRAINING_COLUMNS, fill_value=0)
        return encoded

    ...where TRAINING_COLUMNS is the saved list of columns from training.
    """
    row = {
        "Age":profile.Age,
        "Gender": GENDER_MAP[profile.Gender],
        "Years at Company": profile.Years_at_Company,
        "Job Role": JOB_ROLE_MAP[profile.Job_Role],
        "Monthly Income": profile.Monthly_Income,
        "Work-Life Balance": WORK_LIFE_BALANCE_MAP[profile.Work_Life_Balance],
        "Job Satisfaction": JOB_SATISFACTION_MAP[profile.Job_Satisfaction],
        "Performance Rating": PERFORMANCE_RATING_MAP[profile.Performance_Rating],
        "Number of Promotions": profile.Number_of_Promotions,
        "Overtime": YES_NO_MAP[profile.Overtime],
        "Distance from Home": profile.Distance_from_Home,
        "Education Level": EDUCATION_LEVEL_MAP[profile.Education_Level],
        "Marital Status": MARITAL_STATUS_MAP[profile.Marital_Status],
        "Number of Dependents": profile.Number_of_Dependents,
        "Job Level": JOB_LEVEL_MAP[profile.Job_Level],
        "Company Size": COMPANY_SIZE_MAP[profile.Company_Size],
        "Company Tenure": profile.Company_Tenure,
        "Remote Work": YES_NO_MAP[profile.Remote_Work],
        "Leadership Opportunities": YES_NO_MAP[profile.Leadership_Opportunities],
        "Innovation Opportunities": YES_NO_MAP[profile.Innovation_Opportunities],
        "Company Reputation": COMPANY_REPUTATION_MAP[profile.Company_Reputation],
        "Employee Recognition": EMPLOYEE_RECOGNITION_MAP[profile.Employee_Recognition],
    }
    return pd.DataFrame([row], columns=FEATURE_ORDER)


def validate_categoricals(profile: EmployeeProfile) -> None:
    """Raise a clear 422 error if a category isn't one the model knows."""
    checks = [
        ("Gender", profile.Gender, GENDER_MAP),
        ("Job Role", profile.Job_Role, JOB_ROLE_MAP),
        ("Work-Life Balance", profile.Work_Life_Balance, WORK_LIFE_BALANCE_MAP),
        ("Job Satisfaction", profile.Job_Satisfaction, JOB_SATISFACTION_MAP),
        ("Performance Rating", profile.Performance_Rating, PERFORMANCE_RATING_MAP),
        ("Overtime", profile.Overtime, YES_NO_MAP),
        ("Education Level", profile.Education_Level, EDUCATION_LEVEL_MAP),
        ("Marital Status", profile.Marital_Status, MARITAL_STATUS_MAP),
        ("Job Level", profile.Job_Level, JOB_LEVEL_MAP),
        ("Company Size", profile.Company_Size, COMPANY_SIZE_MAP),
        ("Remote Work", profile.Remote_Work, YES_NO_MAP),
        ("Leadership Opportunities", profile.Leadership_Opportunities, YES_NO_MAP),
        ("Innovation Opportunities", profile.Innovation_Opportunities, YES_NO_MAP),
        ("Company Reputation", profile.Company_Reputation, COMPANY_REPUTATION_MAP),
        ("Employee Recognition", profile.Employee_Recognition, EMPLOYEE_RECOGNITION_MAP),
    ]
    for field_name, value, mapping in checks:
        if value not in mapping:
            allowed = ", ".join(mapping.keys())
            raise HTTPException(
                status_code=422,
                detail=f"Invalid value '{value}' for '{field_name}'. Allowed values: {allowed}",
            )


# ============================================================================
# 4. MODEL LOADING
# ============================================================================

model = None
model_load_error: Optional[str] = None

def load_model():
    global model, model_load_error
    try:
        model = mlflow.sklearn.load_model(MODEL_PATH)
        logger.info(f"MLflow model loaded from {MODEL_PATH}")
        return model
    except Exception as e:
        model_load_error = str(e)
        logger.error(model_load_error)
        return None


model = load_model()


def placeholder_score(features: pd.DataFrame) -> float:
    """
    Transparent, deterministic stand-in scorer used ONLY when no real model
    file is present yet, so the full app works end-to-end out of the box.
    This is NOT a trained model — replace by adding your model.pkl file.
    """
    row = features.iloc[0]
    score = 0.30
    if row["Job Satisfaction"] <= 1: score += 0.14
    if row["Work-Life Balance"] <= 1: score += 0.12
    if row["Overtime"] == 1: score += 0.10
    if row["Employee Recognition"] <= 1: score += 0.08
    if row["Company Reputation"] <= 1: score += 0.06
    if row["Number of Promotions"] == 0 and row["Years at Company"] >= 3: score += 0.08
    if row["Leadership Opportunities"] == 0: score += 0.04
    if row["Innovation Opportunities"] == 0: score += 0.04
    if row["Monthly Income"] < 3000: score += 0.06
    if row["Distance from Home"] > 25: score += 0.04
    return float(min(max(score, 0.02), 0.97))


def predict_probability(features: pd.DataFrame) -> float:
    """
    Returns the probability of attrition (class "Left" / 1) as a float
    between 0 and 1, using the loaded model if available.
    """
    if model is None:
        if USE_PLACEHOLDER_IF_MISSING:
            return placeholder_score(features)
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. {model_load_error or ''} Add a model file at {MODEL_PATH}.",
        )

    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)
            # assume column 1 = positive class ("Left" / attrition = 1)
            proba_arr = np.asarray(proba)
            if proba_arr.ndim == 2 and proba_arr.shape[1] >= 2:
                return float(proba_arr[0][1])
            return float(proba_arr[0])
        elif hasattr(model, "predict"):
            pred = model.predict(features)
            return float(np.asarray(pred).flatten()[0])
        else:
            raise RuntimeError("Loaded model has neither predict_proba nor predict.")
    except Exception as e:
        logger.exception("Model inference failed")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")


def get_top_factors(features: pd.DataFrame, top_n: int = 5) -> Optional[List[Dict[str, Any]]]:
    """
    Best-effort feature importance for the response's 'top_factors' field.
    Works for tree-based scikit-learn models exposing `feature_importances_`.
    Returns None if unavailable (the frontend handles that gracefully).
    """
    if model is None or not hasattr(model, "feature_importances_"):
        return None
    try:
        importances = np.asarray(model.feature_importances_)
        order = np.argsort(importances)[::-1][:top_n]
        total = importances.sum() or 1.0
        return [
            {"name": FEATURE_ORDER[i], "weight": round(float(importances[i] / total) * 100, 1)}
            for i in order
        ]
    except Exception:
        return None


# ============================================================================
# 5. APP SETUP
# ============================================================================

app = FastAPI(
    title="Attrition Insight API",
    description="Predicts employee attrition risk from an HR profile using a trained ML model.",
    version="1.0.0",
)

# CORS — open by default so the static frontend (served from anywhere,
# including file:// during local testing) can call this API. Lock this
# down to your real domain(s) before deploying publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_index():
    path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse({"message": "Attrition Insight API is running. See /docs for the API."})


@app.get("/predict.html")
def serve_predict_page():
    path = os.path.join(FRONTEND_DIR, "predict.html")
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="predict.html not found")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "model_load_error": model_load_error,
        "using_placeholder_scorer": model is None and USE_PLACEHOLDER_IF_MISSING,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(profile: EmployeeProfile):

    validate_categoricals(profile)

    # 🔥 CRITICAL FIX: send RAW data only
    features = pd.DataFrame([profile.model_dump(by_alias=True)])

    probability = predict_probability(features)
    label = "Left" if probability >= 0.5 else "Stayed"

    top_factors = get_top_factors(features)

    return PredictionResponse(
        prediction=label,
        probability=round(probability, 4),
        top_factors=top_factors,
    )


@app.get("/predict/schema")
def predict_schema():
    """Returns the allowed categorical values — handy for debugging the form."""
    return {
        "Gender": list(GENDER_MAP.keys()),
        "Job Role": list(JOB_ROLE_MAP.keys()),
        "Work-Life Balance": list(WORK_LIFE_BALANCE_MAP.keys()),
        "Job Satisfaction": list(JOB_SATISFACTION_MAP.keys()),
        "Performance Rating": list(PERFORMANCE_RATING_MAP.keys()),
        "Overtime": list(YES_NO_MAP.keys()),
        "Education Level": list(EDUCATION_LEVEL_MAP.keys()),
        "Marital Status": list(MARITAL_STATUS_MAP.keys()),
        "Job Level": list(JOB_LEVEL_MAP.keys()),
        "Company Size": list(COMPANY_SIZE_MAP.keys()),
        "Remote Work": list(YES_NO_MAP.keys()),
        "Leadership Opportunities": list(YES_NO_MAP.keys()),
        "Innovation Opportunities": list(YES_NO_MAP.keys()),
        "Company Reputation": list(COMPANY_REPUTATION_MAP.keys()),
        "Employee Recognition": list(EMPLOYEE_RECOGNITION_MAP.keys()),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)