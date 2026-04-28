import json
import os

import numpy as np
import pandas as pd

from transaction_analysis.paths import FRAUD_DATASET_DIR


def parse_currency(series: pd.Series) -> pd.Series:
    """Remove leading '$', strip commas, cast to float."""
    return series.astype(str).str.replace(r"[\$,]", "", regex=True).str.strip().replace("nan", np.nan).astype(float)


def fill_na_with_unknown(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        df[col] = df[col].fillna("Unknown")
    return df


def map_binary_to_int(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        df[col] = df[col].map({"Yes": 1, "No": 0, True: 1, False: 0})
        df[col] = df[col].fillna(0).astype(int)
    return df


def impute_with_median(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median()).astype(int)
    return df


def run(force: bool = False) -> None:
    out_transactions = FRAUD_DATASET_DIR / "cleaned" / "transactions.csv"
    if out_transactions.exists() and not force:
        print("Dataset already cleaned, skipping. Use `force=True` to re-run.")
        return
    os.makedirs(out_transactions.parent, exist_ok=True)

    transactions = pd.read_csv(FRAUD_DATASET_DIR / "raw" / "transactions_data.csv")
    cards = pd.read_csv(FRAUD_DATASET_DIR / "raw" / "cards_data.csv")
    users = pd.read_csv(FRAUD_DATASET_DIR / "raw" / "users_data.csv")

    with open(FRAUD_DATASET_DIR / "raw" / "train_fraud_labels.json") as f:
        labels_raw = json.load(f)

    fraud_labels = (
        pd.DataFrame.from_dict(labels_raw["target"], orient="index", columns=["fraud"])
        .reset_index()
        .rename(columns={"index": "id"})
    )

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
    currency_cols_u = ["per_capita_income", "yearly_income", "total_debt"]

    transactions = (
        transactions.assign(amount=lambda df: parse_currency(df["amount"]))
        .assign(date=lambda df: pd.to_datetime(df["date"], errors="coerce"))
        .pipe(fill_na_with_unknown, ["use_chip", "merchant_city", "merchant_state", "errors"])
        .assign(zip=lambda df: df["zip"].astype(str).str.zfill(5).replace("nan", "Unknown"))
        .assign(mcc=lambda df: df["mcc"].fillna(0).astype(int))
    )

    cards = (
        cards.assign(credit_limit=lambda df: parse_currency(df["credit_limit"]))
        .pipe(map_binary_to_int, ["has_chip", "card_on_dark_web"])
        .assign(acct_open_date=lambda df: pd.to_datetime(df["acct_open_date"], errors="coerce"))
        .assign(expires=lambda df: pd.to_datetime(df["expires"], format="%m/%Y", errors="coerce"))
        .pipe(impute_with_median, ["year_pin_last_changed"])
        .assign(cvv=lambda df: pd.to_numeric(df["cvv"], errors="coerce"))
        .assign(num_cards_issued=lambda df: df["num_cards_issued"].fillna(1).astype(int))
        .pipe(fill_na_with_unknown, ["card_brand", "card_type"])
    )

    users = (
        users.assign(per_capita_income=lambda df: parse_currency(df["per_capita_income"]))
        .assign(yearly_income=lambda df: parse_currency(df["yearly_income"]))
        .assign(total_debt=lambda df: parse_currency(df["total_debt"]))
        .pipe(impute_with_median, num_cols_u)
        .pipe(impute_with_median, currency_cols_u)
        .pipe(fill_na_with_unknown, ["gender", "address"])
    )

    # fmt: off
    fraud_labels = (
        fraud_labels
        .assign(id=lambda df: df["id"].astype(transactions["id"].dtype))
        .assign(fraud=lambda df: df["fraud"].map({"Yes": 1, "No": 0}).astype(int))
    )
    # fmt: on

    # Merge labels onto transactions (left join keeps all transactions)
    transactions = transactions.merge(fraud_labels, on="id", how="left")

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

    dataset.to_csv(out_transactions, index=False)

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    run(force=True)
