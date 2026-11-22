import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X = iris.data           
y = iris.target          

print("Features:", iris.feature_names)
print("Classes: ", list(iris.target_names))
print("X shape: ", X.shape)
print("First row:", X[0], "->", iris.target_names[y[0]])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

artifact = {
    "model": model,
    "feature_names": iris.feature_names,
    "target_names": iris.target_names.tolist(),
}
joblib.dump(artifact, "model.joblib")
print("Saved model.joblib")

loaded = joblib.load("model.joblib")
sample = [[5.1, 3.5, 1.4, 0.2]]
pred = loaded["model"].predict(sample)[0]
print("Prediction for", sample, "->", loaded["target_names"][pred])