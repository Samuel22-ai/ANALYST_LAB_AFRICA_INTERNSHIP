from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import set_config
import __main__

# ==========================================
# 0. GLOBAL ENVIRONMENT CONFIGURATION
# ==========================================
# CRITICAL FIX: We must force Scikit-Learn to output Pandas DataFrames 
# so the custom transformers don't crash looking for column strings.
set_config(transform_output="pandas")

# ==========================================
# 1. THE BLUEPRINTS (Custom Transformers)
# ==========================================
class OutlierTreatmentTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_treat=None, lower_percentile=0.25, upper_percentile=0.75, factor=1.5):
        self.columns_to_treat = columns_to_treat
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.factor = factor
        self.outlier_bounds_ = {} 

    def fit(self, X, y=None):
        if self.columns_to_treat is None:
            raise ValueError("columns_to_treat is not defined.")
        X_copy = X.copy()
        cols = self.columns_to_treat
        for col in cols:
            Q1 = X_copy[col].quantile(self.lower_percentile)
            Q3 = X_copy[col].quantile(self.upper_percentile)
            IQR = Q3 - Q1
            lower_bound = Q1 - (self.factor * IQR)
            upper_bound = Q3 + (self.factor * IQR)
            self.outlier_bounds_[col] = {'lower' : lower_bound, 'upper': upper_bound}      
        return self

    def transform(self, X):
        if self.columns_to_treat is None:
            raise ValueError("columns_to_treat is not defined.")
        X_copy = X.copy()
        cols = self.columns_to_treat
        for col in cols:
            lower = self.outlier_bounds_[col]['lower']
            upper = self.outlier_bounds_[col]['upper']
            X_copy[col] = np.clip(X_copy[col], lower, upper)
        return X_copy

class ConvertZerotoNanTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_fix=None):
        self.columns_to_fix = columns_to_fix

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.columns_to_fix is not None:
            for column in self.columns_to_fix:
                X_copy[column] = X_copy[column].replace(0, np.nan)
        else:
            raise ValueError("columns_to_fix is not defined. Please provide a list of columns to fix.")
        return X_copy 

# ==========================================
# 2. THE NAMESPACE HACK
# ==========================================
__main__.ConvertZerotoNanTransformer = ConvertZerotoNanTransformer
__main__.OutlierTreatmentTransformer = OutlierTreatmentTransformer # Mapped to your actual class name!

# ==========================================
# 3. SERVER & MODEL INITIALIZATION
# ==========================================
app = FastAPI(
    title="Pima Diabetes Diagnostic API", 
    description="A Zero-Leakage XGBoost/SMOTE pipeline for extreme edge anomaly detection.",
    version="1.0.0"
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'pima_xgboost_pipeline.joblib')
try:
    model = joblib.load(MODEL_PATH)
    
    # THE MLOPS SURGICAL BRUTE-FORCE:
    # We iterate through the pipeline and ONLY flip the Pandas switch on 
    # machines that natively support it (like StandardScaler and ColumnTransformer).
    if hasattr(model, 'steps'):
        for step_name, step_obj in model.steps:
            try:
                if hasattr(step_obj, 'set_output'):
                    step_obj.set_output(transform="pandas")
            except ValueError:
                # If it's a custom machine without the switch, we safely ignore it!
                pass
except FileNotFoundError:
    raise RuntimeError(f"CRITICAL ERROR: Model file not found at {MODEL_PATH}")

# ==========================================
# 4. API SCHEMAS & ENDPOINTS
# ==========================================
class PatientData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.post("/predict")
def predict_diabetes(patient: PatientData):
    try:
        input_data = pd.DataFrame([patient.model_dump()])
        prediction_array = model.predict(input_data)
        prediction = int(prediction_array[0]) 
        
        return {
            "prediction_code": prediction,
            "diagnosis": "Diabetic" if prediction == 1 else "Healthy",
            "status": "Success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction Engine Failure: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Pima Diagnostic API is live. Navigate to /docs to test endpoints."}