import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Model ve tokenizer'ı yükle
model = BertForSequenceClassification.from_pretrained("saved_model")
tokenizer = BertTokenizer.from_pretrained("saved_model")

# Örnek yorum
text = "The hotel room was clean and the staff were friendly."

# Tokenize et
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# Tahmin yap
with torch.no_grad():
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits).item()

# Sonucu yazdır
print(f"Tahmin edilen sınıf: {predicted_class}")
