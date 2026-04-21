import os
import duckdb
import pandas as pd

USERS_CSV        = "data/users_data.csv"
CARDS_CSV        = "data/cards_data.csv"
TRANSACTIONS_CSV = "data/transactions_data.csv"

USERS_PARQUET        = "data/users.parquet"
CARDS_PARQUET        = "data/cards.parquet"
TRANSACTIONS_PARQUET = "data/transactions.parquet"

def csv_to_parquet(con: duckdb.DuckDBPyConnection, csv_path: str, parquet_path: str, extra_read_opts: str="") -> None:
    if os.path.exists(parquet_path):
        print(f"{parquet_path} already exists — skipping conversion")
        return
    print(f"Converting {csv_path} → {parquet_path} …")
    con.execute(f"""
        COPY (SELECT * FROM read_csv('{csv_path}', quote='"' {extra_read_opts}))
        TO '{parquet_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    size_mb = os.path.getsize(parquet_path) / 1_048_576
    print(f"Done — {size_mb:.1f} MB")

def q(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).df()

def test() -> None:
    con = duckdb.connect()

    con.execute(f"""
    CREATE OR REPLACE VIEW users AS
    SELECT
        id AS user_id,
        current_age,
        retirement_age,
        gender,
        credit_score,
        num_credit_cards,
        CAST(REPLACE(REPLACE(per_capita_income, '$', ''), ',', '') AS DOUBLE) AS per_capita_income,
        CAST(REPLACE(REPLACE(yearly_income,     '$', ''), ',', '') AS DOUBLE) AS yearly_income,
        CAST(REPLACE(REPLACE(total_debt,        '$', ''), ',', '') AS DOUBLE) AS total_debt
        -- dropped: birth_year, birth_month, address, latitude, longitude
    FROM read_parquet('{USERS_PARQUET}')
    """)

    con.execute(f"""
    CREATE OR REPLACE VIEW cards AS
    SELECT
        id                                                              AS card_id,
        client_id                                                       AS user_id,
        card_brand,
        card_type,
        has_chip,
        card_on_dark_web,
        CAST(REPLACE(REPLACE(credit_limit, '$', ''), ',', '') AS DOUBLE) AS credit_limit
        -- dropped: card_number, cvv, expires, acct_open_date,
        --          num_cards_issued, year_pin_last_changed
    FROM read_parquet('{CARDS_PARQUET}')
    """)

    con.execute(f"""
    CREATE OR REPLACE VIEW transactions AS
    SELECT
        id                                                              AS txn_id,
        CAST(date AS TIMESTAMP)                                         AS txn_date,
        EXTRACT(year  FROM CAST(date AS TIMESTAMP))::INT               AS txn_year,
        EXTRACT(month FROM CAST(date AS TIMESTAMP))::INT               AS txn_month,
        EXTRACT(hour  FROM CAST(date AS TIMESTAMP))::INT               AS txn_hour,
        EXTRACT(dow   FROM CAST(date AS TIMESTAMP))::INT               AS txn_dow,
        client_id                                                       AS user_id,
        card_id,
        CAST(REPLACE(REPLACE(amount, '$', ''), ',', '') AS DOUBLE)     AS amount,
        CASE WHEN LOWER(use_chip) LIKE '%chip%' THEN 1 ELSE 0 END      AS used_chip,
        merchant_id,
        merchant_city,
        merchant_state,
        mcc,
        CASE WHEN errors IS NOT NULL THEN 1 ELSE 0 END                 AS has_error,
        errors
        -- dropped: zip (mostly redundant with city/state)
    FROM read_parquet('{TRANSACTIONS_PARQUET}')
    WHERE client_id IS NOT NULL
    AND card_id   IS NOT NULL
    AND amount    IS NOT NULL
    AND date      IS NOT NULL
    """)

    print(q(con, "SELECT COUNT(*) AS total_transactions FROM transactions").to_string(index=False))

    con.close()