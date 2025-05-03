**MLE Project - FastAPI ML Service**********

Welcome to the MLE Assignment Project! This project demonstrates building, training, and serving simple machine learning models using FastAPI with MongoDB as storage. It also includes Dockerization and a GitLab CI/CD setup.

**Project Features**

- Train a model on Iris dataset (classic classification)
- Load and classify real-world news articles
- MongoDB integration for storage

**RESTful API endpoints for:**

- Data loading
- Model training
- Single and batch predictions
- Fully containerized with Docker + Docker Compose
- Automated Linting and Build via GitLab CI/CD

**Tech Stack**

- FastAPI (Python 3.12)
- scikit-learn (SVM, Logistic Regression)
- MongoDB (NoSQL database)
- Docker (Containerization)
- GitLab CI/CD (Pipeline Automation)

**How to Run Locally**

1. Install dependencies

```
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```


2. Start MongoDB

Install Mongo locally 

OR

Run via Docker:
`docker run -d -p 27017:27017 --name mongo mongo`

3. Run FastAPI server

`uvicorn app.main:app --reload`

App will be available at:

http://localhost:8000/docs


**Docker Setup**

1. Build and Run with Docker Compose

```
docker-compose up --build

FastAPI app: http://localhost:8000

MongoDB: mongodb://localhost:27017
```



**API Endpoints**

**✨ Health Check**

`GET /health`

**🌿 Iris Dataset**


```
POST /iris/load-data : Load Iris data into Mongo

POST /iris/train : Train model on Iris data

POST /iris/predict : Predict Iris species (based on text input)
```


**📰 News Classification (Bonus)**

```
POST /news/load : Load labeled news dataset

POST /news/train : Train classifier on news headlines

POST /news/predict : Predict category for one news headline

POST /news/predict-batch : Predict for multiple news headlines

POST /news/predict-unlabeled : Predict categories for unlabeled data (CPS responses)
```

**GitLab CI/CD**

Automated pipeline:

- Lint Python files with black
- Build Docker image
- Located in .gitlab-ci.yml

**Model Evaluation**
We evaluated the News Classification Model on a custom sample evaluation dataset consisting of real-world headlines across multiple categories (e.g., Politics, Tech, Sports, Health).


**Metric	Score**
- Accuracy	80%
- F1 Score (Weighted)	80%
- Evaluation was performed using sklearn metrics: accuracy_score and f1_score.

Dataset contained 10 news headlines manually labeled with ground truth.

Model generalizes well and shows strong performance on unseen data.

**💜 Acknowledgements**

- scikit-learn for easy ML prototyping
- FastAPI for beautiful API development
- GitLab for robust CI/CD pipelines

**🌟 Author**
by Aniruddh Sahukar Srinvas

