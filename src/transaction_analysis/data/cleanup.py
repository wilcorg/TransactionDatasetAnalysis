import os
from pathlib import Path

import pandas as pd

from transaction_analysis.data import io
from transaction_analysis.paths import FRAUD_DATASET_DIR


def datetime_to_date(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = df[col].dt.date
    return df


def assert_no_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        missing = df[col].isna().sum()
        assert missing == 0, f"Column '{col}' has {missing} missing values"
    return df


def impute_online(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df.loc[df["merchant_city"] == "ONLINE", col] = "ONLINE"
    return df


def impute_errors(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df.loc[df[col].isna(), col] = "No error"
    return df


def join_year_month_to_datetime(df: pd.DataFrame, year_col: str, month_col: str, out_col: str) -> pd.DataFrame:
    df[year_col] = pd.to_datetime(
        {
            "year": df[year_col].astype("int64"),
            "month": df[month_col].astype("int64"),
            "day": 1,
        }
    ).dt.date
    df.rename(columns={year_col: out_col}, inplace=True)
    return df.drop(columns=[month_col])


def run(dataset_in_dir: Path, dataset_out_dir: Path, force: bool = False) -> None:
    os.makedirs(dataset_out_dir, exist_ok=True)

    def cleanup_cards(in_file: Path, out_file: Path) -> None:
        if out_file.exists() and not force:
            print("Cards already cleaned, skipping. Use `force=True` to re-run.")
            return

        (
            io.read_parquet(in_file)
            .rename(columns={"id": "card_id"})
            .pipe(datetime_to_date, "expires")
            .pipe(datetime_to_date, "acct_open_date")
            .rename(columns={"credit_limit": "credit_limit_usd"})
            .pipe(assert_no_missing_values)
            .pipe(io.to_parquet, out_file)
        )

    def cleanup_mcc(in_file: Path, out_file: Path) -> None:
        if out_file.exists() and not force:
            print("MCC codes already cleaned, skipping. Use `force=True` to re-run.")
            return

        (io.read_parquet(in_file).pipe(assert_no_missing_values).pipe(io.to_parquet, out_file))

    def cleanup_fraud_labels(in_file: Path, out_file: Path) -> None:
        if out_file.exists() and not force:
            print("Fraud labels already cleaned, skipping. Use `force=True` to re-run.")
            return

        (io.read_parquet(in_file).pipe(assert_no_missing_values).pipe(io.to_parquet, out_file))

    def cleanup_transactions(in_file: Path, out_file: Path) -> None:
        if out_file.exists() and not force:
            print("Transactions already cleaned, skipping. Use `force=True` to re-run.")
            return

        (
            io.read_parquet(in_file)
            .rename(columns={"id": "transaction_id"})
            .rename(columns={"amount": "amount_usd"})
            .rename(columns={"use_chip": "trasaction_type"})
            .pipe(impute_online, "merchant_state")
            .pipe(impute_online, "zip")
            .pipe(impute_errors, "errors")
            .pipe(assert_no_missing_values)
            .pipe(io.to_parquet, out_file)
        )

    def cleanup_users(in_file: Path, out_file: Path) -> None:
        if out_file.exists() and not force:
            print("Users already cleaned, skipping. Use `force=True` to re-run.")
            return

        (
            io.read_parquet(in_file)
            .drop(columns=["current_age"])
            .pipe(join_year_month_to_datetime, "birth_year", "birth_month", "birth_date")
            .drop(columns=["address", "latitude", "longitude"])
            .rename(columns={"per_capita_income": "per_capita_income_usd"})
            .rename(columns={"yearly_income": "yearly_income_usd"})
            .rename(columns={"total_debt": "total_debt_usd"})
            .pipe(assert_no_missing_values)
            .pipe(io.to_parquet, out_file)
        )

    cleanup_cards(dataset_in_dir / "cards.parquet", dataset_out_dir / "cards.parquet")
    cleanup_mcc(dataset_in_dir / "mcc_codes.parquet", dataset_out_dir / "mcc_codes.parquet")
    cleanup_fraud_labels(dataset_in_dir / "fraud_labels.parquet", dataset_out_dir / "fraud_labels.parquet")
    cleanup_transactions(dataset_in_dir / "transactions.parquet", dataset_out_dir / "transactions.parquet")
    cleanup_users(dataset_in_dir / "users.parquet", dataset_out_dir / "users.parquet")


if __name__ == "__main__":
    run(Path(FRAUD_DATASET_DIR / "preprocessed"), Path(FRAUD_DATASET_DIR / "cleaned"), force=True)
