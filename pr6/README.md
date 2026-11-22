# Iris Classifier API: FastAPI + Docker

A simple machine learning model (Random Forest trained on the Iris dataset)
served as a REST API with **FastAPI** and packaged with **Docker**.

## Project structure

```
pr6/
├── train.py           # trains the model and saves model.joblib
├── model.joblib       # saved trained model (artifact)
├── main.py            # FastAPI application
├── requirements.txt   # Python dependencies (pinned versions)
├── Dockerfile         # container instructions
├── .dockerignore      # files excluded from the Docker image
└── README.md
```

## Model

- Dataset: Iris (150 flowers, 4 features, 3 classes: setosa, versicolor, virginica)
- Algorithm: `RandomForestClassifier` (100 trees)
- Train/test split: 80/20, stratified
- Test accuracy: **0.90**

## API endpoints

| Method | Path       | Description                          |
|--------|------------|--------------------------------------|
| GET    | `/`        | Health check: confirms the API is running |
| POST   | `/predict` | Returns the predicted iris species   |
| GET    | `/docs`    | Interactive Swagger documentation    |

**Example request:**

```json
POST /predict
{
  "sepal_length": 6.7,
  "sepal_width": 3.0,
  "petal_length": 5.2,
  "petal_width": 2.3
}
```

**Example response:**

```json
{
  "prediction": 2,
  "class_name": "virginica",
  "probabilities": {"setosa": 0.0, "versicolor": 0.0, "virginica": 1.0}
}
```

## How to run

### 1. Train the model (optional, `model.joblib` is already included)

```bash
pip install -r requirements.txt
python train.py
```

### 2. Run locally with Uvicorn

```bash
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs

### 3. Run with Docker

```bash
docker build -t iris-api .
docker run -d --name iris-api -p 8000:8000 iris-api
```

Open http://127.0.0.1:8000/docs

Test from the terminal:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 6.0, "sepal_width": 2.7, "petal_length": 4.5, "petal_width": 1.4}'
```

Stop and remove the container:

```bash
docker stop iris-api && docker rm iris-api
```
