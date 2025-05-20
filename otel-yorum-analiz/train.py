from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from config import *
from preprocessing import preprocess_and_save
import torch
import pandas as pd

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted"),
    }

def train_model():
    preprocess_and_save()
    df = pd.read_csv("data/cleaned_reviews.csv")
    dataset = Dataset.from_pandas(df).train_test_split(test_size=0.2)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    def tokenize_fn(example):
        return tokenizer(example["Full_Review"], padding="max_length", truncation=True, max_length=MAX_LEN)

    dataset = dataset.map(tokenize_fn, batched=True)
    dataset = dataset.rename_column("Label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    args = TrainingArguments(
        output_dir="output",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        logging_dir="logs",
        logging_steps=10,
        report_to="none",
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

