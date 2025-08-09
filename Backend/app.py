from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from model.gemma_lora import generate_response

app = FastAPI()

class SymptomInput(BaseModel):
    symptom: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def get_prediction(input: SymptomInput):
    text = input.symptom.lower().strip()

    try:
        response = generate_response(text)
        return {"type": "llm", "response": response}
    except Exception as e:
        print("Error during prediction:", e)
        return {"type": "error", "response": "Internal server error."}
