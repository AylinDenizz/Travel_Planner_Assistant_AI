import pandas as pd
import re
from config import DATA_PATH

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def classify_score(score):
    if score <= 4.0:
        return 0
    elif score <= 7.0:
        return 1
    return 2

def preprocess_and_save():
    df = pd.read_csv(DATA_PATH)
    df['Negative_Review'] = df['Negative_Review'].replace("No Negative", "")
    df['Positive_Review'] = df['Positive_Review'].replace("No Positive", "")
    df['Full_Review'] = (df['Positive_Review'] + " " + df['Negative_Review']).apply(clean_text)
    df['Label'] = df['Reviewer_Score'].apply(classify_score)
    df[['Full_Review', 'Label']].to_csv("data/cleaned_reviews.csv", index=False)

