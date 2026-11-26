import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="centered")

st.sidebar.header("System status")

try:
    requests.get(f"{API_URL}/", timeout=3).raise_for_status()
    info = requests.get(f"{API_URL}/model-info", timeout=3).json()
    api_ok = True
except requests.exceptions.RequestException:
    api_ok = False

if api_ok:
    st.sidebar.success(f"API online: {API_URL}")
    st.sidebar.subheader("Model in production")
    st.sidebar.write(f"**{info['model_name']}** v{info['model_version']}")
    st.sidebar.write(f"Algorithm: `{info['algorithm']}`")
    st.sidebar.caption(f"MLflow run: {info['run_id']}")
    st.sidebar.subheader("Test metrics")
    for name, value in info["metrics"].items():
        st.sidebar.metric(name, f"{value:.3f}")
else:
    st.sidebar.error(f"API not reachable at {API_URL}")

st.title("Iris Species Classifier")
st.write(
    "Set the flower measurements and click **Predict**. "
    "The prediction comes from the FastAPI service, which serves the "
    "champion model from the MLflow Model Registry."
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Sepal")
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8, 0.1)
    sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0, 0.1)
with col2:
    st.subheader("Petal")
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.3, 0.1)
    petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.3, 0.1)

# ---------- Prediction ----------
if st.button("Predict", type="primary", disabled=not api_ok):
    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    confidence = result["probabilities"][result["class_name"]]
    st.success(f"Predicted species: **{result['class_name']}**")

    c1, c2 = st.columns(2)
    c1.metric("Confidence", f"{confidence:.0%}")
    c2.metric("Model version", f"v{result['model_version']}")

    st.subheader("Class probabilities")
    probs = pd.DataFrame({"probability": result["probabilities"]})
    st.bar_chart(probs)

    with st.expander("Raw API request and response"):
        st.write("Request sent to `/predict`:")
        st.json(payload)
        st.write("Response:")
        st.json(result)