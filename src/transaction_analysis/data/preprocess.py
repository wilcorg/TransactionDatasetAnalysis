import os
from pathlib import Path

import pandas as pd
import pyarrow as pa

from transaction_analysis.paths import FRAUD_DATASET_DIR


def currency_to_decimal(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = df[col].replace(r"[\$,]", "", regex=True).str.strip().astype(pd.ArrowDtype(pa.decimal32(9, 2)))
    return df


def str_to_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def int64_downcast(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = pd.to_numeric(df[col], downcast="unsigned")
    return df


def str_to_category(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = df[col].astype("category")
    return df


def zip_to_category(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = df[col].astype("string").replace(r"\.\d+$", "", regex=True).str.zfill(5).astype("category")
    return df


def month_year_to_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col], format="%m/%Y", errors="coerce")
    return df


def yes_no_to_bool(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = df[col].str.strip().str.upper().map({"YES": True, "NO": False}).astype("boolean")
    return df


def run(force: bool = False) -> None:
    os.makedirs(FRAUD_DATASET_DIR / "cleaned", exist_ok=True)

    def process_transactions(input_dataset: Path, output_dataset: Path) -> None:
        if output_dataset.exists() and not force:
            print("Transactions already preprocessed, skipping. Use `force=True` to re-run.")
            return

        transactions = pd.read_csv(input_dataset, engine="pyarrow", dtype_backend="pyarrow")

        transactions = (
            transactions.pipe(int64_downcast, "id")
            .pipe(str_to_datetime, "date")
            .pipe(int64_downcast, "client_id")
            .pipe(int64_downcast, "card_id")
            .pipe(currency_to_decimal, "amount")
            .pipe(str_to_category, "use_chip")
            .pipe(int64_downcast, "merchant_id")
            .pipe(str_to_category, "merchant_city")
            .pipe(str_to_category, "merchant_state")
            .pipe(zip_to_category, "zip")
            .pipe(int64_downcast, "mcc")
            .pipe(str_to_category, "errors")
        )

        transactions.to_parquet(output_dataset, compression="zstd")

    def process_users(input_dataset: Path, output_dataset: Path) -> None:
        if output_dataset.exists() and not force:
            print("Users already preprocessed, skipping. Use `force=True` to re-run.")
            return

        users = pd.read_csv(input_dataset, engine="pyarrow", dtype_backend="pyarrow")
        users = (
            users.pipe(int64_downcast, "id")
            .pipe(int64_downcast, "current_age")
            .pipe(int64_downcast, "retirement_age")
            .pipe(int64_downcast, "birth_year")
            .pipe(int64_downcast, "birth_month")
            .pipe(str_to_category, "gender")
            .pipe(currency_to_decimal, "per_capita_income")
            .pipe(currency_to_decimal, "yearly_income")
            .pipe(currency_to_decimal, "total_debt")
            .pipe(int64_downcast, "credit_score")
            .pipe(int64_downcast, "num_credit_cards")
        )

        users.to_parquet(output_dataset, compression="zstd")

    def process_cards(input_dataset: Path, output_dataset: Path) -> None:
        if output_dataset.exists() and not force:
            print("Cards already preprocessed, skipping. Use `force=True` to re-run.")
            return

        cards = pd.read_csv(input_dataset, engine="pyarrow", dtype_backend="pyarrow")
        cards = (
            cards.pipe(int64_downcast, "id")
            .pipe(int64_downcast, "client_id")
            .pipe(str_to_category, "card_brand")
            .pipe(str_to_category, "card_type")
            .pipe(month_year_to_datetime, "expires")
            .pipe(int64_downcast, "cvv")
            .pipe(yes_no_to_bool, "has_chip")
            .pipe(int64_downcast, "num_cards_issued")
            .pipe(currency_to_decimal, "credit_limit")
            .pipe(month_year_to_datetime, "acct_open_date")
            .pipe(int64_downcast, "year_pin_last_changed")
            .pipe(yes_no_to_bool, "card_on_dark_web")
        )

        cards.to_parquet(output_dataset, compression="zstd")

    def process_mcc_codes(input_dataset: Path, output_dataset: Path) -> None:
        if output_dataset.exists() and not force:
            print("MCC codes already preprocessed, skipping. Use `force=True` to re-run.")
            return

        mcc_codes: pd.DataFrame = (
            pd.read_json(input_dataset, typ="series", dtype_backend="pyarrow")
            .rename_axis("mcc")
            .reset_index(name="description")
        )
        mcc_codes = mcc_codes.pipe(int64_downcast, "mcc")

        mcc_codes.to_parquet(output_dataset, compression="zstd")

    def process_fraud_labels(input_dataset: Path, output_dataset: Path) -> None:
        if output_dataset.exists() and not force:
            print("Fraud labels already preprocessed, skipping. Use `force=True` to re-run.")
            return

        # fmt: off
        fraud_labels = (
            pd.read_json(input_dataset, dtype_backend="pyarrow")
            .rename_axis("id")
            .reset_index(name="fraud")
        )
        # fmt: on
        fraud_labels = fraud_labels.pipe(int64_downcast, "id")
        fraud_labels.to_parquet(output_dataset, compression="zstd")

    # fmt: off
    process_transactions(
        FRAUD_DATASET_DIR / "raw" / "transactions_data.csv",
        FRAUD_DATASET_DIR / "cleaned" / "transactions.parquet"
    )
    process_users(
        FRAUD_DATASET_DIR / "raw" / "users_data.csv",
        FRAUD_DATASET_DIR / "cleaned" / "users.parquet"
    )
    process_cards(
        FRAUD_DATASET_DIR / "raw" / "cards_data.csv",
        FRAUD_DATASET_DIR / "cleaned" / "cards.parquet"
    )
    process_mcc_codes(
        FRAUD_DATASET_DIR / "raw" / "mcc_codes.json",
        FRAUD_DATASET_DIR / "cleaned" / "mcc_codes.parquet"
    )
    process_fraud_labels(
        FRAUD_DATASET_DIR / "raw" / "train_fraud_labels.json",
        FRAUD_DATASET_DIR / "cleaned" / "fraud_labels.parquet"
    )
    # fmt: on


if __name__ == "__main__":
    run(force=True)
