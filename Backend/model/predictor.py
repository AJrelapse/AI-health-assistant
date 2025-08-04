import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
import csv

with open("model/label_mapping.json") as f:
    label_map = json.load(f)

id2label = {int(v): k for k, v in label_map.items()}

model_path = "model/trained_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

diagnosis_data = {}
with open("model/diagnosis_info.csv", newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        diagnosis_data[row["diagnosis"]] = {
            "cause": row["cause"],
            "department": row["department"],
            "home_remedies": row["home_remedies"].split(";")
        }

def predict_diagnosis(symptom: str) -> dict:
    inputs = tokenizer(symptom, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_label_id = torch.argmax(outputs.logits, dim=1).item()

    diagnosis = id2label.get(predicted_label_id, "Unknown Diagnosis")
    info = diagnosis_data.get(diagnosis, {
        "cause": "Information not available.",
        "department": "General Physician",
        "home_remedies": ["Consult a doctor."]
    })

    return {
        "diagnosis": diagnosis,
        "cause": info["cause"],
        "department": info["department"],
        "home_remedies": info["home_remedies"]
    }
