"""
Module: Pima Diabetes Diagnostic FastAPI Microservice
Purpose: Deploys a trained XGBoost classifier wrapped in a custom Scikit-Learn pipeline 
         to predict the onset of diabetes based on 8 clinical diagnostic measurements. 
         Provides a RESTful API endpoint for applications to send JSON data and receive 
         automated medical predictions.
Time Complexity: O(1) for inference (model prediction on a single record is instantaneous).
Space Complexity: O(M) where M is the memory footprint of the loaded XGBoost pipeline.
Dependencies: fastapi, pydantic, joblib, pandas, numpy, scikit-learn
"""

# Import FastAPI for building the API, and HTTPException for handling server errors
from fastapi import FastAPI, HTTPException

# Import CORSMiddleware to allow cross-origin requests from frontend applications (like Streamlit)
from fastapi.middleware.cors import CORSMiddleware

# Import BaseModel from Pydantic to create strict data validation schemas
from pydantic import BaseModel

# Import joblib to deserialize (load) our pre-trained machine learning pipeline
import joblib

# Import data manipulation and math libraries
import pandas as pd
import numpy as np
import os

# Import base Scikit-Learn classes required to rebuild our custom transformers
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import set_config

# Import the special __main__ namespace to resolve model pickling errors
import __main__

# ==========================================
# 0. GLOBAL ENVIRONMENT CONFIGURATION
# ==========================================
# By default, Scikit-Learn outputs raw NumPy arrays after transformations.
# We globally force it to output Pandas DataFrames. This ensures that when data 
# passes through the pipeline, the column names survive, preventing the custom 
# transformers from crashing when they look for specific string column names.
set_config(transform_output="pandas")

# ==========================================
# 1. THE BLUEPRINTS (Custom Transformers)
# ==========================================
# When pickled models was saved, we saved custom Python classes inside it.
# To load the model successfully, the exact "blueprints" of those classes must exist 
# in this deployment script, otherwise, Python will not know how to rebuild the pipeline.

class OutlierTreatmentTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to calculate Interquartile Range (IQR) boundaries 
    and clip extreme physiological outliers to prevent model skew.
    
    Args:
        columns_to_treat (list, optional): Column names to apply capping to.
        lower_percentile (float): The lower quartile limit (default 0.25).
        upper_percentile (float): The upper quartile limit (default 0.75).
        factor (float): The multiplier for the IQR to determine outlier bounds (default 1.5).
    """
    def __init__(self, columns_to_treat=None, lower_percentile=0.25, upper_percentile=0.75, factor=1.5):
        # Initialize the transformer with user-defined boundaries and columns
        self.columns_to_treat = columns_to_treat
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.factor = factor
        self.outlier_bounds_ = {} 

    def fit(self, X, y=None):
        """
        Learns the upper and lower statistical boundaries from the dataset.
        """
        if self.columns_to_treat is None:
            raise ValueError("columns_to_treat is not defined.")
        
        # Always work on a copy to prevent mutating the original memory reference
        X_copy = X.copy()
        cols = self.columns_to_treat
        
        # Loop through each column to learn its specific statistical spread
        for col in cols:
            # Calculate the 25th (Q1) and 75th (Q3) percentiles
            Q1 = X_copy[col].quantile(self.lower_percentile)
            Q3 = X_copy[col].quantile(self.upper_percentile)
            
            # The IQR is the "middle 50%" of the data
            IQR = Q3 - Q1
            
            # Define the fences: anything outside these fences is an extreme outlier
            lower_bound = Q1 - (self.factor * IQR)
            upper_bound = Q3 + (self.factor * IQR)
            
            # Memorize these fences in a dictionary to use during the transform phase
            self.outlier_bounds_[col] = {'lower' : lower_bound, 'upper': upper_bound}      
        
        return self

    def transform(self, X):
        """
        Applies the memorized boundaries to clip extreme values.
        """
        if self.columns_to_treat is None:
            raise ValueError("columns_to_treat is not defined.")
        
        X_copy = X.copy()
        cols = self.columns_to_treat
        
        # Apply the clipping rules to the live data passing through the API
        for col in cols:
            lower = self.outlier_bounds_[col]['lower']
            upper = self.outlier_bounds_[col]['upper']
            
            # np.clip forces any number lower than 'lower' to equal 'lower', 
            # and any number higher than 'upper' to equal 'upper'.
            X_copy[col] = np.clip(X_copy[col], lower, upper)
            
        return X_copy

class ConvertZerotoNanTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to replace biologically impossible '0' values 
    (indicating hardware sensor failures) with formal missing values (NaN).
    
    Args:
        columns_to_fix (list, optional): Specific columns known to have hidden zero anomalies.
    """
    def __init__(self, columns_to_fix=None):
        self.columns_to_fix = columns_to_fix

    def fit(self, X, y=None):
        # This transformer does not learn any statistical boundaries, so fit just returns itself
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.columns_to_fix is not None:
            for column in self.columns_to_fix:
                # Target the specific column, replacing the integer 0 with Numpy's NaN
                # This ensures the next step in the pipeline (the Imputer) recognizes it as missing
                X_copy[column] = X_copy[column].replace(0, np.nan)
        else:
            raise ValueError("columns_to_fix is not defined. Please provide a list of columns to fix.")
        return X_copy 

# ==========================================
# 2. THE NAMESPACE HACK
# ==========================================
# When Joblib saved the model in the Jupyter notebook, it saved these custom transformers 
# under the "__main__" namespace of that specific notebook environment. 
# In this new file, Joblib will look for "__main__.ConvertZerotoNanTransformer".
# Explicitly mapped our local classes to the __main__ namespace to prevent a FileNotFoundError / ModuleNotFoundError.
__main__.ConvertZerotoNanTransformer = ConvertZerotoNanTransformer
__main__.OutlierTreatmentTransformer = OutlierTreatmentTransformer 

# ==========================================
# 3. SERVER & MODEL INITIALIZATION
# ==========================================
# Instantiate the FastAPI server framework, providing metadata for the automated /docs page.
app = FastAPI(
    title="Pima Diabetes Diagnostic API", 
    description="A Zero-Leakage XGBoost/SMOTE pipeline for extreme edge anomaly detection.",
    version="1.0.0"
)

# --- SECURITY / MIDDLEWARE CONFIGURATION ---
# Inject Cross-Origin Resource Sharing (CORS) middleware into the application.
# This prevents modern web browsers from blocking API requests originating from different domains 
# (e.g., future Streamlit frontend hosted on a different URL).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any origin. For strict production, replace "*" with specific URLs.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- ROBUST PATH RESOLUTION ---
# Dynamically construct the file path to the saved machine learning model.
# Using os.path.abspath ensures that even if Docker initiates Uvicorn from a different root directory,
# the path is resolved absolutely, preventing relative directory crashes.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'models', 'pima_xgboost_pipeline.joblib'))

try:
    # Load the entire serialized pipeline (cleaning + scaling + XGBoost model) into memory
    model = joblib.load(MODEL_PATH)
    
    # THE MLOPS SURGICAL BRUTE-FORCE:
    # We iterate through the pipeline and ONLY flip the Pandas switch on 
    # machines that natively support it (like StandardScaler and ColumnTransformer).
    if hasattr(model, 'steps'):
        for step_name, step_obj in model.steps:
            try:
                # If the specific transformer supports the 'set_output' method,
                # forcefully override it to output Pandas DataFrames instead of Numpy arrays.
                # This prevents "Environment Amnesia" between the notebook and the API.
                if hasattr(step_obj, 'set_output'):
                    step_obj.set_output(transform="pandas")
            except ValueError:
                # If the step is an algorithm or custom transformer that doesn't support 'set_output', safely ignore it.
                pass
except FileNotFoundError:
    # If the model file is missing or the path is wrong, crash the server immediately with a clear error.
    raise RuntimeError(f"CRITICAL ERROR: Model file not found at {MODEL_PATH}")

# ==========================================
# 4. API SCHEMAS & ENDPOINTS
# ==========================================
class PatientData(BaseModel):
    """
    Pydantic Schema for incoming JSON data validation.
    This acts as a strict "data bouncer". If an API request sends a string where 
    a float is expected, Pydantic will instantly reject it before it hits our model.
    """
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
    """
    The primary POST endpoint. Receives clinical vitals, passes them through 
    the pre-trained pipeline, and returns a diabetes diagnosis.
    
    Args:
        patient (PatientData): The validated JSON payload parsed into a Python object.
        
    Returns:
        dict: A dictionary containing the prediction code (0 or 1), the text diagnosis, and the server status.
        
    Raises:
        HTTPException: Returns a 400 Bad Request if the prediction engine fails mathematically.
    """
    try:
        # Convert the Pydantic object into a Python dictionary, wrap it in a list, 
        # and convert it into a Pandas DataFrame. Models require 2D matrix inputs!
        input_data = pd.DataFrame([patient.model_dump()])
        
        # Pass the DataFrame into the pipeline. 
        # It automatically handles the zero-to-NaN conversions, IQR capping, scaling, and XGBoost inference.
        prediction_array = model.predict(input_data)
        
        # Extract the exact prediction integer from the resulting 1D Numpy array
        prediction = int(prediction_array[0]) 
        
        # Construct and return the JSON response
        return {
            "prediction_code": prediction,
            "diagnosis": "Diabetic" if prediction == 1 else "Healthy",
            "status": "Success"
        }
    except Exception as e:
        # If anything goes wrong inside the Pandas or Scikit-Learn logic, 
        # safely catch the error and return a formatted HTTP response to the client.
        raise HTTPException(status_code=400, detail=f"Prediction Engine Failure: {str(e)}")

@app.get("/")
def read_root():
    """
    Root endpoint serving as a simple health check to verify the server is running.
    """
    return {"message": "Pima Diagnostic API is live. Navigate to /docs to test endpoints."}