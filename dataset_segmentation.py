"""
dataset_segmentation.py
------------------------
Generates a synthetic (but realistic) customer dataset for persona
segmentation, similar in spirit to the classic "Mall Customers" dataset
used for clustering demos.

Run this file directly to produce customers.csv:
    python dataset_segmentation.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)


def generate_customer_data(n_customers: int = 1000) -> pd.DataFrame:
    """Generate synthetic customer data with 4 latent personas baked in."""

    records = []

    # Persona archetypes -> (age range, income range($k), spending score,
    # purchase frequency/year, avg transaction value($))
    personas = [
        {"name": "Budget Shopper",   "age": (22, 40), "income": (15, 40),
         "spend_score": (10, 35), "freq": (1, 8),  "txn": (10, 40)},
        {"name": "High Roller",      "age": (30, 55), "income": (80, 150),
         "spend_score": (70, 100), "freq": (15, 40), "txn": (150, 400)},
        {"name": "Occasional Buyer", "age": (25, 65), "income": (30, 70),
         "spend_score": (30, 55), "freq": (2, 10), "txn": (30, 90)},
        {"name": "Loyal Regular",    "age": (28, 50), "income": (40, 90),
         "spend_score": (55, 80), "freq": (12, 30), "txn": (60, 150)},
    ]

    n_per_persona = n_customers // len(personas)

    for persona in personas:
        for _ in range(n_per_persona):
            age = np.random.randint(*persona["age"])
            income = np.random.uniform(*persona["income"])
            spend_score = np.random.uniform(*persona["spend_score"])
            freq = np.random.uniform(*persona["freq"])
            txn = np.random.uniform(*persona["txn"])

            records.append({
                "age": age,
                "annual_income_k": round(income, 1),
                "spending_score": round(spend_score, 1),
                "purchase_frequency": round(freq, 1),
                "avg_transaction_value": round(txn, 1),
            })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    df.insert(0, "customer_id", range(1, len(df) + 1))
    return df


if __name__ == "__main__":
    df = generate_customer_data(1000)
    df.to_csv("customers.csv", index=False)
    print(f"Saved {len(df)} rows to customers.csv")
    print(df.head())
