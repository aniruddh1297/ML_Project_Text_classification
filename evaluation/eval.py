import os
import json
import pickle
from sklearn.metrics import accuracy_score, f1_score
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
NEWS_MODEL_PATH = os.getenv("NEWS_MODEL_PATH", "news_model.pkl")
EVAL_FILE_PATH = "evaluation/news_eval_dataset.json"

# Load Model
with open(NEWS_MODEL_PATH, "rb") as f:
    vectorizer, clf = pickle.load(f)

# Load Evaluation Dataset
data = []
labels = []
texts = []

with open(EVAL_FILE_PATH, "r", encoding="utf-8") as f:
    records = json.load(f) 


for item in records:
    text = item.get("text") or item.get("headline")
    label = item.get("label") or item.get("true_label")

    if not text or not label:
        continue

    texts.append(text)
    labels.append(label)

# Make Predictions
X = vectorizer.transform(texts)
preds = clf.predict(X)

# Evaluate
accuracy = accuracy_score(labels, preds)
f1 = f1_score(labels, preds, average="weighted")

print("======================")
print("Model Evaluation Results")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score (Weighted): {f1:.4f}")
print("======================")
