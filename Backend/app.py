from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model.predictor import predict_diagnosis

app = FastAPI()

class SymptomInput(BaseModel):
    symptom: str

@app.post("/predict")
def get_prediction(input: SymptomInput):
    result = predict_diagnosis(input.symptom)
    return result
