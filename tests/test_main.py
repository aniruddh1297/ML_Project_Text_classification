# ========================
# tests/test_main.py
# ========================

import logging
import pytest
from fastapi.testclient import TestClient
from app.main import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def client():
    return TestClient(app)

# ========== FAST TESTS ==========

@pytest.mark.fast
def test_health(client):
    logger.info("Testing /health endpoint")
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@pytest.mark.fast
def test_iris_load(client):
    logger.info("Testing /iris/load endpoint")
    response = client.post("/iris/load")
    assert response.status_code == 200

@pytest.mark.fast
def test_iris_train(client):
    logger.info("Testing /iris/train endpoint")
    response = client.post("/iris/train")
    assert response.status_code == 200

@pytest.mark.fast
def test_iris_predict(client):
    logger.info("Testing /iris/predict endpoint")
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/iris/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()

@pytest.mark.fast
def test_news_load(client):
    logger.info("Testing /news/load endpoint")
    response = client.post("/news/load")
    assert response.status_code == 200

# ========== SLOW TESTS ==========

@pytest.mark.slow
def test_news_train(client):
    logger.info("Testing /news/train endpoint (slow)")
    response = client.post("/news/train")
    assert response.status_code == 200

@pytest.mark.slow
def test_news_predict(client):
    logger.info("Testing /news/predict endpoint")
    payload = {"text": "Breaking news in technology today!"}
    response = client.post("/news/predict", json=payload)
    assert response.status_code == 200
    assert "label" in response.json()

@pytest.mark.slow
def test_news_predict_unlabeled(client):
    logger.info("Testing /news/predict-unlabeled endpoint")
    response = client.post("/news/predict-unlabeled")
    assert response.status_code == 200
