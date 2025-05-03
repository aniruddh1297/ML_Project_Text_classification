from fastapi import FastAPI, HTTPException, Body
from app import model, mongodb
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class NewsInput(BaseModel):
    text: str


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


app = FastAPI()

IRIS_MODEL_PATH = os.getenv("IRIS_MODEL_PATH", "iris_model.pkl")


@app.get("/health")
def health():
    return {"status": "ok"}


# ======== IRIS Endpoints ========


@app.post("/iris/load")
def load_iris_data():
    mongodb.load_iris_csv_to_mongo()
    return {"message": "Iris data loaded into MongoDB"}


@app.post("/iris/train")
def train_iris_model():
    df = mongodb.get_iris_samples_df()
    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    joblib.dump(clf, IRIS_MODEL_PATH)
    return {"message": "Iris model trained"}


@app.post("/iris/predict")
def predict_iris(input: IrisInput):
    if not os.path.exists(IRIS_MODEL_PATH):
        raise HTTPException(status_code=400, detail="Model not trained yet")

    model = joblib.load(IRIS_MODEL_PATH)
    features = [
        [input.sepal_length, input.sepal_width, input.petal_length, input.petal_width]
    ]
    prediction = model.predict(features)[0]
    return {"prediction": prediction}


# ======== NEWS Endpoints ========


@app.post("/news/load")
def load_news():
    mongodb.load_news_dataset_to_mongo()
    return {"message": "News dataset loaded into MongoDB"}


@app.post("/news/train")
def train_news():
    data = mongodb.get_news_articles()
    if not data:
        raise HTTPException(status_code=400, detail="No news data found to train")
    model.train_news_classifier(data)
    return {"message": "News classifier trained successfully"}


@app.post("/news/predict")
def predict_news(input: NewsInput):
    prediction = model.predict_news(input.text)
    return {"label": prediction}


@app.post("/news/predict-unlabeled")
def predict_unlabeled_news():
    mongodb.predict_response_file(verbose=True)
    return {"message": "Predictions written to CPS_response_with_predictions.json"}
