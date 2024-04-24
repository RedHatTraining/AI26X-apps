import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from joblib import load

from dto.patient_dto import Patient

app = FastAPI()

# Load the model from a file
# model = ...


@app.post("/patient/diagnose")
def diagnose_diabetes(patient: Patient):
    try:
        # Return the model prediction
        return "TODO: implement prediction"
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


classes = ('No diabetes', 'Diabetes')


def predict(patients: Patient):
    # Add the model prediction
    ...


if __name__ == "__main__":
    uvicorn.run("diabetes_api:app", host="0.0.0.0", port=8000, reload=True)
