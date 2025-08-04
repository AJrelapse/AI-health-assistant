from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
import torch
import transformers
import json

print("Transformers version path:", transformers.__file__)

raw_dataset = load_dataset('csv', data_files='backend/data/symptoms.csv')
dataset = raw_dataset['train']

label_encoder = LabelEncoder()
label_encoder.fit(dataset['diagnosis'])

dataset = dataset.map(lambda x: {'label': int(label_encoder.transform([x['diagnosis']])[0])})

label_map = {
    str(label): int(code)
    for label, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
}
with open("backend/model/label_mapping.json", "w") as f:
    json.dump(label_map, f)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_fn(example):
    return tokenizer(example['symptom'], padding='max_length', truncation=True, max_length=64)

tokenized_dataset = dataset.map(tokenize_fn)

tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_encoder.classes_)
)

training_args = TrainingArguments(
    output_dir="./backend/model/trained_model",
    evaluation_strategy="no",
    save_strategy="epoch",
    logging_strategy="epoch",
    logging_dir="./backend/model/logs",
    per_device_train_batch_size=8,
    num_train_epochs=5,
    learning_rate=2e-5
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()
trainer.save_model("./backend/model/trained_model")
tokenizer.save_pretrained("./backend/model/trained_model")
