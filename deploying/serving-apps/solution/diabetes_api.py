import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from joblib import load

from dto.patient_dto import Patient

app = FastAPI()

# Load the model from a file
model = load('sklearn_diabetes_model.joblib')


@app.post("/patient/diagnose")
def diagnose_diabetes(patient: Patient):
    try:
        # Return the model prediction
        return predict(patient)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


classes = ('No diabetes', 'Diabetes')


def predict(patients: Patient):
    inputs = pd.DataFrame([patients.dict()])
    return classes[model.predict(inputs)[0]]


if __name__ == "__main__":
    uvicorn.run("diabetes_api:app", host="0.0.0.0", port=8000, reload=True)
