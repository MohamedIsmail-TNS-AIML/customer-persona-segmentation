# Customer Persona Segmentation

A simple end-to-end ML web app that segments customers into personas
(**Budget Shopper**, **Occasional Buyer**, **Loyal Regular**, **High Roller**)
using unsupervised KMeans clustering — structured the same way as the
car-price-estimator example (dataset script, train script, FastAPI backend,
Streamlit frontend, saved `.pkl` artifacts).

## Files

| File                        | Purpose                                              |
|-----------------------------|-------------------------------------------------------|
| `dataset_segmentation.py`   | Generates synthetic customer data (`customers.csv`)   |
| `train_segmentation.py`     | Trains KMeans model, saves `.pkl` artifacts            |
| `main_segmentation.py`      | FastAPI backend serving predictions                    |
| `app_segmentation.py`       | Streamlit frontend UI                                  |
| `segmentation_model.pkl`    | Trained KMeans model                                   |
| `preprocessor.pkl`          | Fitted StandardScaler                                  |
| `persona_labels.pkl`        | Cluster ID -> persona name mapping                     |
| `customers.csv`             | Raw synthetic dataset                                  |
| `customers_labeled.csv`     | Dataset with cluster/persona labels attached            |
| `requirements.txt`          | Python dependencies                                     |

## Setup

```bash
pip install -r requirements.txt
```

## 1. Generate data + train the model

```bash
python dataset_segmentation.py   # creates customers.csv
python train_segmentation.py     # trains KMeans, saves .pkl files
```

## 2. Start the backend (FastAPI)

```bash
uvicorn main_segmentation:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive API docs.

## 3. Start the frontend (Streamlit)

In a separate terminal:

```bash
streamlit run app_segmentation.py
```

Visit `http://localhost:8501` in your browser.

## How it works

- Each customer is described by 5 features: `age`, `annual_income_k`,
  `spending_score`, `purchase_frequency`, `avg_transaction_value`.
- Features are scaled with `StandardScaler`, then clustered into 4 groups
  with `KMeans`.
- Cluster centroids are ranked by income/spend/frequency and mapped to
  human-readable persona names.
- The Streamlit app sends a customer's profile to the FastAPI `/predict`
  endpoint and displays the predicted persona, or lets you upload a CSV
  for batch segmentation.

## Extend it

- Swap the synthetic data for a real customer/CRM export (keep the same
  column names, or update `FEATURES` in `train_segmentation.py` and
  `main_segmentation.py`).
- Try a different number of clusters (`N_CLUSTERS` in `train_segmentation.py`)
  or use the elbow method / silhouette score to pick it automatically.
- Add RFM (Recency, Frequency, Monetary) features for a more standard
  marketing segmentation.
