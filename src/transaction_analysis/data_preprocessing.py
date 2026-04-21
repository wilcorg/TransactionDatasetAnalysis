import json

import numpy as np
import pandas as pd
import paths

transactions = pd.read_csv(paths.RAW_DATASET_DIR / "transactions_data.csv")
cards = pd.read_csv(paths.RAW_DATASET_DIR / "cards_data.csv")
users = pd.read_csv(paths.RAW_DATASET_DIR / "users_data.csv")


def parse_currency(series: pd.Series) -> pd.Series:
    """Remove leading '$', strip commas, cast to float."""
    return series.astype(str).str.replace(r"[\$,]", "", regex=True).str.strip().replace("nan", np.nan).astype(float)


transactions["amount"] = parse_currency(transactions["amount"])
transactions["date"] = pd.to_datetime(transactions["date"], errors="coerce")


cat_cols_t = ["use_chip", "merchant_city", "merchant_state", "errors"]
for col in cat_cols_t:
    transactions[col] = transactions[col].fillna("Unknown")


transactions["zip"] = transactions["zip"].astype(str).str.zfill(5)
transactions["zip"] = transactions["zip"].replace("nan", "Unknown")


transactions["mcc"] = transactions["mcc"].fillna(0).astype(int)


cards["credit_limit"] = parse_currency(cards["credit_limit"])

# -- boolean-like column: map Yes/No → 1/0
for col in ["has_chip", "card_on_dark_web"]:
    cards[col] = cards[col].map({"Yes": 1, "No": 0, True: 1, False: 0})
    cards[col] = cards[col].fillna(0).astype(int)

# -- date columns
cards["acct_open_date"] = pd.to_datetime(cards["acct_open_date"], errors="coerce")
cards["expires"] = pd.to_datetime(cards["expires"], format="%m/%Y", errors="coerce")
cards["year_pin_last_changed"] = pd.to_numeric(cards["year_pin_last_changed"], errors="coerce")
cards["year_pin_last_changed"] = (
    cards["year_pin_last_changed"].fillna(cards["year_pin_last_changed"].median()).astype(int)
)

# -- numeric columns
cards["cvv"] = pd.to_numeric(cards["cvv"], errors="coerce")
cards["num_cards_issued"] = cards["num_cards_issued"].fillna(1).astype(int)

# -- categorical
cards["card_brand"] = cards["card_brand"].fillna("Unknown")
cards["card_type"] = cards["card_type"].fillna("Unknown")


currency_cols_u = ["per_capita_income", "yearly_income", "total_debt"]
for col in currency_cols_u:
    users[col] = parse_currency(users[col])

# -- numeric columns: fill with median
num_cols_u = [
    "current_age",
    "retirement_age",
    "birth_year",
    "birth_month",
    "latitude",
    "longitude",
    "credit_score",
    "num_credit_cards",
]
for col in num_cols_u:
    users[col] = pd.to_numeric(users[col], errors="coerce")
    users[col] = users[col].fillna(users[col].median())

# -- currency: fill with median
for col in currency_cols_u:
    users[col] = users[col].fillna(users[col].median())

# -- categorical
users["gender"] = users["gender"].fillna("Unknown")
users["address"] = users["address"].fillna("Unknown")

with open(paths.DA / "train_fraud_labels.json") as f:
    labels_raw = json.load(f)

labels_df = (
    pd.DataFrame.from_dict(labels_raw["target"], orient="index", columns=["fraud"])
    .reset_index()
    .rename(columns={"index": "id"})
)

labels_df["id"] = labels_df["id"].astype(transactions["id"].dtype)
labels_df["fraud"] = labels_df["fraud"].map({"Yes": 1, "No": 0}).astype(int)


# Merge labels onto transactions (left join keeps all transactions)
transactions = transactions.merge(labels_df, on="id", how="left")

# Rows with no label are test-set transactions – leave fraud as NaN
print(f"Labelled transactions : {transactions['fraud'].notna().sum():,}")
print(f"Unlabelled (test set) : {transactions['fraud'].isna().sum():,}")


print("\n── transactions ──")
print(transactions.dtypes)
print(transactions.isnull().sum())

print("\n── cards ──")
print(cards.dtypes)
print(cards.isnull().sum())

print("\n── users ──")
print(users.dtypes)
print(users.isnull().sum())

# Uncomment if you want a single modelling-ready DataFrame.

dataset = transactions.merge(cards.add_prefix("card_"), left_on="card_id", right_on="card_id", how="left").merge(
    users.add_prefix("user_"), left_on="client_id", right_on="user_id", how="left"
)
print(f"\nFull dataset shape: {dataset.shape}")

dataset.to_csv(paths.DATASET_DIR / "transactions_cleaned.csv", index=False)

print("\nPreprocessing complete.")
