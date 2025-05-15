from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Define input model
class PatientInput(BaseModel):
    patient_id: str
    height: float = Field(..., gt=0)
    weight: float = Field(..., gt=0)

# Create FastAPI app
app = FastAPI(title="BMI Health Verdict API")

# In-memory patient database
patient_db: Dict[str, dict] = {}

# Dataset for training
data = [
    {"height": 175, "weight": 70, "verdict": "normal"},
    {"height": 160, "weight": 50, "verdict": "underweight"},
    {"height": 180, "weight": 90, "verdict": "overweight"},
    {"height": 165, "weight": 95, "verdict": "obese"},
    {"height": 172, "weight": 65, "verdict": "normal"},
    {"height": 158, "weight": 45, "verdict": "underweight"},
    {"height": 182, "weight": 78, "verdict": "normal"},
    {"height": 170, "weight": 85, "verdict": "overweight"},
    {"height": 168, "weight": 100, "verdict": "obese"},
    {"height": 160, "weight": 60, "verdict": "normal"},
]

df = pd.DataFrame(data)
df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)

# Train model
X = df[["bmi"]]
y = df["verdict"]
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to the BMI Verdict API"}

# Predict and store by patient_id
@app.post("/predict")
def predict_verdict(patient: PatientInput):
    bmi = patient.weight / ((patient.height / 100) ** 2)
    pred = model.predict([[bmi]])[0]

    result = {
        "patient_id": patient.patient_id,
        "height": patient.height,
        "weight": patient.weight,
        "bmi": round(bmi, 2),
        "predicted_verdict": pred
    }

    # Save result
    patient_db[patient.patient_id] = result

    return result

# Retrieve prediction by ID
@app.get("/patient/{patient_id}")
def get_patient_prediction(patient_id: str):
    if patient_id not in patient_db:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient_db[patient_id]
