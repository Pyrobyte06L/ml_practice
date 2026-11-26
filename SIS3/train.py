import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

mlflow.set_tracking_uri("sqlite:///mlflow.db") 
EXPERIMENT_NAME = "iris-classification"
MODEL_NAME = "IrisClassifier"

# Internal sklearn types that skops doesn't trust by default.
# Safe here because we create these models ourselves.
TRUSTED_TYPES = [
    "sklearn.tree._tree.Tree",                       # RandomForest
    "sklearn.neighbors._kd_tree.KDTree",             # KNN
    "sklearn.metrics._dist_metrics.EuclideanDistance64",  # KNN
]

mlflow.set_experiment(EXPERIMENT_NAME)           

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

candidates = [
    ("logistic_regression", LogisticRegression,
     {"C": 1.0, "max_iter": 200}),
    ("random_forest_small", RandomForestClassifier,
     {"n_estimators": 50, "max_depth": 3, "random_state": 42}),
    ("random_forest_large", RandomForestClassifier,
     {"n_estimators": 200, "max_depth": None, "random_state": 42}),
    ("knn", KNeighborsClassifier,
     {"n_neighbors": 5}),
]

results = []

for run_name, model_class, params in candidates:
    with mlflow.start_run(run_name=run_name) as run:
        model = model_class(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "precision_macro": precision_score(y_test, y_pred, average="macro"),
            "recall_macro": recall_score(y_test, y_pred, average="macro"),
        }

        mlflow.log_param("model_type", model_class.__name__)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_train, model.predict(X_train))
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=X_train[:2],
            skops_trusted_types=TRUSTED_TYPES,
        )

        joblib.dump(
            {"model": model, "target_names": iris.target_names.tolist()},
            "model.joblib",
        )
        mlflow.log_artifact("model.joblib")

        results.append({
            "run_name": run_name,
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
            "model": model,
            **metrics,
        })
        print(f"{run_name:22s} accuracy={metrics['accuracy']:.3f}  f1={metrics['f1_macro']:.3f}")

best = max(results, key=lambda r: r["f1_macro"])
print(f"\nBest model: {best['run_name']} (f1={best['f1_macro']:.3f})")

model_version = mlflow.register_model(model_uri=best["model_uri"], name=MODEL_NAME)
version = int(model_version.version)
print(f"Registered '{MODEL_NAME}' version {version}")

client = MlflowClient()
client.set_registered_model_alias(MODEL_NAME, "champion", version)
client.update_model_version(
    MODEL_NAME, version,
    description=f"{best['run_name']}, f1_macro={best['f1_macro']:.3f}",
)
client.set_model_version_tag(MODEL_NAME, version, "algorithm", best["run_name"])
print(f"Alias 'champion' -> version {version}")

metric_names = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
joblib.dump(
    {
        "model": best["model"],
        "target_names": iris.target_names.tolist(),
        "model_name": MODEL_NAME,
        "model_version": version,
        "algorithm": best["run_name"],
        "run_id": best["run_id"],
        "metrics": {m: round(best[m], 3) for m in metric_names},
    },
    "model.joblib",
)
print("Exported champion to model.joblib")

champion = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
pred = champion.predict([[5.1, 3.5, 1.4, 0.2]])[0]
print("Registry champion predicts:", iris.target_names[pred])