import os
import pandas as pd
from pymongo import MongoClient
import json
import logging
from app.model import predict_news
from dotenv import load_dotenv
import time

# Load env vars
load_dotenv()

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# MongoDB connection
client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
db = client.ml_db


# IRIS Functions
def load_iris_csv_to_mongo():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(base_dir, "data", "iris.csv")

    logger.info(f"Loading Iris CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    df.rename(columns={"species": "label"}, inplace=True)
    records = df.to_dict(orient="records")

    db.iris_samples.drop()
    db.iris_samples.insert_many(records)

    logger.info(f"Inserted {len(records)} Iris records.")


def get_iris_samples_df():
    data = list(db.iris_samples.find({}, {"_id": 0}))
    return pd.DataFrame(data)


# NEWS Functions
def load_news_dataset_to_mongo(path="data/CPS_use_case_classification_training.json"):
    cleaned = []
    total = 0
    skipped = 0

    logger.info(f"Loading news data from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            try:
                item = json.loads(line)
                text = item.get("headline")
                label = item.get("category")

                if text and label:
                    cleaned.append({"text": text, "label": label})
                else:
                    skipped += 1
            except json.JSONDecodeError:
                skipped += 1

    if not cleaned:
        raise Exception(
            "❌ No valid labeled records found — check your field names or content!"
        )

    db.news_articles.drop()
    db.news_articles.insert_many(cleaned)
    logger.info(f"Processed {total} | Inserted {len(cleaned)} | Skipped {skipped}")


def get_news_articles():
    return list(db.news_articles.find({}, {"_id": 0}))


def predict_response_file(
    input_path="data/CPS_use_case_classification_response.json",
    output_path="data/CPS_response_with_predictions.json",
    verbose=True,
):
    start = time.time()
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    result = []
    skipped = 0

    logger.info(f"Starting predictions on {len(lines)} headlines...")
    for i, line in enumerate(lines):
        try:
            item = json.loads(line)
            headline = item.get("headline")
            if not headline:
                skipped += 1
                continue

            predicted = predict_news(headline)
            item["predicted_category"] = predicted
            result.append(item)

            if verbose and i % 1000 == 0:
                logger.info(f"Processed {i}/{len(lines)} headlines...")

        except json.JSONDecodeError:
            skipped += 1

    with open(output_path, "w", encoding="utf-8") as out:
        for record in result:
            out.write(json.dumps(record) + "\n")

    logger.info(f"✅ Wrote {len(result)} predictions to {output_path}")
    logger.info(f"🕒 Time taken: {time.time() - start:.2f} seconds")
    logger.info(f"Skipped {skipped} malformed or empty records")
