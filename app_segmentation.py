"""
app_segmentation.py
---------------------
Streamlit frontend for the Customer Persona Segmentation app.
Talks to the FastAPI backend (main_segmentation.py) over HTTP.

Run with:
    streamlit run app_segmentation.py

Make sure the backend is running first:
    uvicorn main_segmentation:app --reload --port 8000
"""

import pandas as pd
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Customer Persona Segmentation", page_icon="🧩", layout="centered")

st.title("🧩 Customer Persona Segmentation")
st.write(
    "Enter a customer's profile to predict which **persona segment** they "
    "belong to. This app calls a FastAPI backend serving a trained "
    "KMeans clustering model."
)

# ---- Backend health check ----
with st.sidebar:
    st.header("Backend status")
    try:
        health = requests.get(f"{API_URL}/", timeout=3).json()
        if health.get("model_loaded"):
            st.success("Backend connected ✅\nModel loaded.")
        else:
            st.warning("Backend connected, but model artifacts are missing.\n"
                        "Run `train_segmentation.py` first.")
    except requests.exceptions.RequestException:
        st.error("Cannot reach backend at " + API_URL +
                  "\nStart it with:\n`uvicorn main_segmentation:app --reload --port 8000`")

    st.divider()
    st.header("Known personas")
    try:
        personas = requests.get(f"{API_URL}/personas", timeout=3).json()["personas"]
        for p in personas:
            st.markdown(f"**{p['name']}**  \n{p['description']}")
    except Exception:
        st.caption("Persona list unavailable until backend is running.")

st.divider()

# ---- Input form ----
st.subheader("Customer profile")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 16, 90, 35)
    annual_income_k = st.number_input("Annual income ($k)", 0.0, 500.0, 65.0, step=1.0)
    spending_score = st.slider("Spending score (0-100)", 0, 100, 55)
with col2:
    purchase_frequency = st.number_input("Purchases per year", 0.0, 100.0, 12.0, step=1.0)
    avg_transaction_value = st.number_input("Avg transaction value ($)", 0.0, 5000.0, 95.0, step=5.0)

if st.button("Predict persona", type="primary", use_container_width=True):
    payload = {
        "age": age,
        "annual_income_k": annual_income_k,
        "spending_score": spending_score,
        "purchase_frequency": purchase_frequency,
        "avg_transaction_value": avg_transaction_value,
    }
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        resp.raise_for_status()
        result = resp.json()

        st.success(f"### Predicted persona: **{result['persona']}**")
        st.write(result["persona_description"])
        st.caption(f"Cluster ID: {result['cluster']}")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach backend: {e}")

st.divider()

# ---- Batch upload ----
st.subheader("Batch segmentation (optional)")
st.write("Upload a CSV with columns: `age, annual_income_k, spending_score, "
         "purchase_frequency, avg_transaction_value`")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head())

    if st.button("Segment all customers"):
        records = df.to_dict(orient="records")
        try:
            resp = requests.post(f"{API_URL}/predict_batch", json=records, timeout=15)
            resp.raise_for_status()
            results = pd.DataFrame(resp.json())
            df_out = pd.concat([df.reset_index(drop=True), results], axis=1)
            st.dataframe(df_out)

            st.bar_chart(df_out["persona"].value_counts())

            csv = df_out.to_csv(index=False).encode("utf-8")
            st.download_button("Download results as CSV", csv, "segmented_customers.csv", "text/csv")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach backend: {e}")
