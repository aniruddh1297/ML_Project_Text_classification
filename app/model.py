import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

_news_vec = None
_news_clf = None

NEWS_MODEL_PATH = os.getenv("NEWS_MODEL_PATH", "news_model.pkl")


def train_news_classifier(samples):
    texts = [s["text"] for s in samples]
    labels = [s["label"] for s in samples]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, labels)

    with open(NEWS_MODEL_PATH, "wb") as f:
        pickle.dump((vectorizer, clf), f)


def load_news_model():
    global _news_vec, _news_clf
    if _news_vec is None or _news_clf is None:
        print("[INFO] Loading news_model.pkl into memory...")
        with open(NEWS_MODEL_PATH, "rb") as f:
            _news_vec, _news_clf = pickle.load(f)
    return _news_vec, _news_clf


def predict_news(text):
    vec, clf = load_news_model()
    X = vec.transform([text])
    return clf.predict(X)[0]
