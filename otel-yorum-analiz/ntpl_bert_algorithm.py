import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import evaluate
import re
import torch

df=pd.read_csv(r'C:\Users\aylin\OneDrive\Masaüstü\travelplanner\otel-yorum-analiz\data\Hotel_Reviews.csv')

df['Negative_Review'] = df['Negative_Review'].replace("No Negative", "")
df['Positive_Review'] = df['Positive_Review'].replace("No Positive", "")

df['Full_Review'] = df['Positive_Review'] + " " + df['Negative_Review']

df_cleaned = df[['Hotel_Name', 'Hotel_Address', 'Reviewer_Score', 'Full_Review', 'lat', 'lng']].copy()


# 4. Text normalize: küçük harf, noktalama temizliği
def clean_text(text):
    text = str(text).lower()                        # küçük harfe çevir
    text = re.sub(r'[^\w\s]', '', text)             # noktalama işaretlerini kaldır
    text = re.sub(r'\s+', ' ', text).strip()        # fazla boşlukları temizle
    return text

def classify_score(score):
    if score <= 4.0:
        return 0   # Olumsuz
    elif score <= 7.0:
        return 1   # Nötr
    else:
        return 2   # Olumlu

df_cleaned['Full_Review'] = df_cleaned['Full_Review'].apply(clean_text)

df_cleaned['Label'] = df_cleaned['Reviewer_Score'].apply(classify_score)

df_cleaned.to_csv("cleaned_reviews.csv", index=False)

# 🧪 Adım 2: Veriyi HuggingFace Dataset formatına çevir
dataset = Dataset.from_pandas(df_cleaned[['Full_Review', 'Label']])

# küçük bir subset ile deneyebilirsin
dataset = dataset.select(range(20000))  # İlk 20K örnekle çalış

# ✂️ Adım 3: Train/Validation ayır
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset['train']
val_dataset = dataset['test']

# Tokenizer yükle
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["Full_Review"], padding="max_length", truncation=True, max_length=256)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Label'ı modelin anlayacağı şekilde ayarla
train_dataset = train_dataset.rename_column("Label", "labels")
val_dataset = val_dataset.rename_column("Label", "labels")

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 🧠 Adım 5: Modeli yükle
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ⚙️ Adım 6: Eğitim parametreleri
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",  # 🔧 Kayıt kriteri
    fp16=torch.cuda.is_available()      # 🔧 GPU varsa mixed precision kullan
)



# metric hesaplama
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted"),
    }



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# 🚀 Adım 9: Eğitimi başlat
trainer.train()

# 💾 Modeli kaydet
model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")


# 📊 Sonuçları göster
eval_result = trainer.evaluate()
print("Evaluation Results:", eval_result)


from transformers import BertTokenizer, BertForSequenceClassification
import torch

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
print(f"Predicted class: {predicted_class}")


from sklearn.metrics import classification_report

#Tüm test verisi üzerinde tahmin yap, gerçek ve tahmin edilenleri karşılaştır
#classification_report(y_true, y_pred, target_names=["neg", "neutral", "pos"])
#🎉 Hangisini yapalım? 🔍 Tahmin örneği mi?

#📊 Test ve metrik analizi mi?

#🌐 Web servisi mi?

#🧪 Test verisiyle batch değerlendirme mi?

#Sen karar ver, ben buradayım 💪