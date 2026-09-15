"""
train_segmentation.py
----------------------
Trains a KMeans clustering model on customer data to discover customer
personas/segments, then saves:
    - segmentation_model.pkl  (trained KMeans model)
    - preprocessor.pkl        (fitted StandardScaler)
    - persona_labels.pkl      (mapping from cluster id -> human label)

Run this file directly:
    python train_segmentation.py
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from dataset_segmentation import generate_customer_data

FEATURES = [
    "age",
    "annual_income_k",
    "spending_score",
    "purchase_frequency",
    "avg_transaction_value",
]

N_CLUSTERS = 4


def label_clusters(centroids_df: pd.DataFrame) -> dict:
    """
    Assign a human-readable persona name to each cluster based on the
    characteristics of its centroid (income + spending score + frequency).
    """
    labels = {}
    # Rank clusters by a simple "value score" combining spend + frequency + income
    value_score = (
        centroids_df["spending_score"].rank()
        + centroids_df["purchase_frequency"].rank()
        + centroids_df["annual_income_k"].rank()
    )
    order = value_score.sort_values().index.tolist()  # low -> high value

    names = ["Budget Shopper", "Occasional Buyer", "Loyal Regular", "High Roller"]
    for rank, cluster_id in enumerate(order):
        labels[int(cluster_id)] = names[min(rank, len(names) - 1)]
    return labels


def main():
    # 1. Load / generate data
    try:
        df = pd.read_csv("customers.csv")
    except FileNotFoundError:
        df = generate_customer_data(1000)
        df.to_csv("customers.csv", index=False)

    X = df[FEATURES].values

    # 2. Preprocess
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Train KMeans
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(X_scaled)
    df["cluster"] = cluster_ids

    # 4. Build human-readable persona labels from centroids (in original units)
    centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
    centroids_df = pd.DataFrame(centroids_original, columns=FEATURES)
    persona_labels = label_clusters(centroids_df)
    df["persona"] = df["cluster"].map(persona_labels)

    # 5. Save artifacts
    joblib.dump(kmeans, "segmentation_model.pkl")
    joblib.dump(scaler, "preprocessor.pkl")
    joblib.dump(persona_labels, "persona_labels.pkl")
    df.to_csv("customers_labeled.csv", index=False)

    print("Training complete.")
    print("Cluster centroids (original scale):")
    print(centroids_df.round(2))
    print("\nPersona label mapping:", persona_labels)
    print("\nPersona distribution:")
    print(df["persona"].value_counts())


if __name__ == "__main__":
    main()
