from transformers import BertTokenizer, BertForSequenceClassification
import torch
from config import SAVE_DIR

def predict_sample(text):
    tokenizer = BertTokenizer.from_pretrained(SAVE_DIR)
    model = BertForSequenceClassification.from_pretrained(SAVE_DIR)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()
    print(f"Text: {text}\nPredicted class: {pred}")
