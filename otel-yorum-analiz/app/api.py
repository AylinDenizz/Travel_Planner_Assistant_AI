# app/api.py

from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

model = BertForSequenceClassification.from_pretrained("saved_model")
tokenizer = BertTokenizer.from_pretrained("saved_model")

# Sınıf isimlerini sözlükle eşleştir
label_map = {
    0: "Negatif",
    1: "Nötr",
    2: "Pozitif"
}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits).item()

    label = label_map.get(predicted_class, "Bilinmiyor")
    return jsonify({"prediction": predicted_class, "label": label})

@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    data = request.json
    texts = data.get("texts", [])
    results = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class = torch.argmax(outputs.logits).item()
            label = label_map.get(predicted_class, "Bilinmiyor")
            results.append({"text": text, "prediction": predicted_class, "label": label})

    return jsonify(results)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)
