# Pima Diabetes Diagnostic API

## Project Overview
This project is an end-to-end Machine Learning microservice built for the AnalystLab Africa Internship. It deploys a trained XGBoost classifier wrapped in a custom Scikit-Learn pipeline to predict the onset of diabetes based on clinical diagnostic measurements.

## Model Description
The core engine is an XGBoost algorithm optimized via `GridSearchCV` and balanced using SMOTE to handle class imbalances. To prevent data leakage during deployment, custom transformers (`OutlierTreatmentTransformer` and `ConvertZerotoNanTransformer`) are natively integrated into the pipeline. The model is served via a FastAPI backend and containerized using Docker.

## Input Features
The API expects a JSON payload containing the following 8 clinical vitals:
* `Pregnancies` (int): Number of times pregnant
* `Glucose` (float): Plasma glucose concentration
* `BloodPressure` (float): Diastolic blood pressure (mm Hg)
* `SkinThickness` (float): Triceps skin fold thickness (mm)
* `Insulin` (float): 2-Hour serum insulin (mu U/ml)
* `BMI` (float): Body mass index
* `DiabetesPedigreeFunction` (float): Diabetes pedigree function
* `Age` (int): Age in years

## Example Request
**POST /predict**
json
{
"Pregnancies": 1,
"Glucose": 180.0,
"BloodPressure": 70.0,
"SkinThickness": 20.0,
"Insulin": 79.0,
"BMI": 32.0,
"DiabetesPedigreeFunction": 0.5,
"Age": 33
}

## Example Response
**200 OK**
json
{
"prediction_code": 1,
"diagnosis": "Diabetic",
"status": "Success"
}

## Instructions on How to Run the API

### Option 1: Live Cloud Testing
The API is actively deployed on Render.
1. Navigate to: `https://analyst-lab-africa-internship.onrender.com/docs`
2. Expand the `POST /predict` endpoint and click "Try it out".
3. Enter the Example Request JSON above and click "Execute".

### Option 2: Local Docker Container
1. Clone this repository to your local machine.
2. Open a terminal in the root directory.
3. Build the container image:
   `docker build -t pima-api .`
4. Run the container:
   `docker run -p 8000:8000 pima-api`
5. Navigate to `http://localhost:8000/docs` in your browser.