import os
from pathlib import Path

import pandas as pd
import pyarrow as pa

from transaction_analysis.data import io
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


def run(dataset_in_dir: Path, dataset_out_dir: Path, force: bool = False) -> None:
    os.makedirs(dataset_out_dir, exist_ok=True)

    def process_transactions(in_file: Path, out_file: Path) -> None:
        if out_file.exists() and not force:
            print("Transactions already preprocessed, skipping. Use `force=True` to re-run.")
            return

        (
            io.read_csv(in_file)
            .pipe(int64_downcast, "id")
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
            .set_index("id")
            .pipe(io.to_parquet, out_file)
        )

    def process_users(in_file: Path, out_file: Path) -> None:
        if out_file.exists() and not force:
            print("Users already preprocessed, skipping. Use `force=True` to re-run.")
            return

        (
            io.read_csv(in_file)
            .pipe(int64_downcast, "id")
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
            .set_index("id")
            .pipe(io.to_parquet, out_file)
        )

    def process_cards(in_file: Path, out_file: Path) -> None:
        if out_file.exists() and not force:
            print("Cards already preprocessed, skipping. Use `force=True` to re-run.")
            return

        (
            io.read_csv(in_file)
            .pipe(int64_downcast, "id")
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
            .set_index("id")
            .pipe(io.to_parquet, out_file)
        )

    def process_mcc_codes(in_file: Path, out_file: Path) -> None:
        if out_file.exists() and not force:
            print("MCC codes already preprocessed, skipping. Use `force=True` to re-run.")
            return

        (
            io.read_json(in_file, typ="series")
            .rename_axis("id")
            .rename("description")
            .reset_index()
            .pipe(int64_downcast, "id")
            .set_index("id")
            .pipe(io.to_parquet, out_file)
        )

    def process_fraud_labels(in_file: Path, out_file: Path) -> None:
        if out_file.exists() and not force:
            print("Fraud labels already preprocessed, skipping. Use `force=True` to re-run.")
            return

        # fmt: off
        (
            io.read_json(in_file)
            .rename_axis("id")
            .rename(columns={"target": "fraud"})
            .reset_index()
            .pipe(int64_downcast, "id")
            .pipe(yes_no_to_bool, "fraud")
            .set_index("id")
            .pipe(io.to_parquet, out_file)
        )
        # fmt: on

    # fmt: off
    process_transactions(
        dataset_in_dir / "transactions_data.csv",
        dataset_out_dir / "transactions.parquet",
    )
    process_users(
        dataset_in_dir / "users_data.csv",
        dataset_out_dir / "users.parquet",
    )
    process_cards(
        dataset_in_dir / "cards_data.csv",
        dataset_out_dir / "cards.parquet",
    )
    process_mcc_codes(
        dataset_in_dir / "mcc_codes.json",
        dataset_out_dir / "mcc_codes.parquet",
    )
    process_fraud_labels(
        dataset_in_dir / "train_fraud_labels.json",
        dataset_out_dir / "fraud_labels.parquet",
    )
    # fmt: on


if __name__ == "__main__":
    run(
        dataset_in_dir=FRAUD_DATASET_DIR / "raw",
        dataset_out_dir=FRAUD_DATASET_DIR / "preprocessed",
        force=True,
    )
