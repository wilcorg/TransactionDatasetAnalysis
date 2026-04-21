import duckdb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

from _old.utils import CARDS_PARQUET, TRANSACTIONS_PARQUET, USERS_PARQUET, q

class EDA:
    """Exploratory Data Analysis (EDA) for Credit Card Transactions Dataset"""
    def __init__(self):
        self.figsize = (12, 5)
        self.figsq = (8, 6)
        self.con = duckdb.connect()
        self._setup_database()
        self.user_agg = self._get_user_agg()
        sns.set_theme()

    def __del__(self):
        self.con.close()
        
    def preprocessing(user_agg):
        feature_columns = ['credit_score', 'yearly_income', '']
        X = user_agg[feature_columns].copy()

        numeric_features = X.select_dtypes(include='number').columns.tolist()
        categorical_features = X.select_dtypes(exclude='number').columns.tolist()

        preprocessor = ColumnTransformer([
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features),
        ])

        X_prepared = preprocessor.fit_transform(X)
        X_prepared.shape

    def amount_sample(self, bins: int = 80) -> None:
        """Visualize the distribution of transaction amounts (raw and log-transformed)"""
        amount_sample = q(self.con, """
            SELECT amount
            FROM transactions
            WHERE amount > 0
            """)
        _, axes = plt.subplots(1, 2, figsize=self.figsize)
        sns.histplot(amount_sample["amount"], bins=bins, ax=axes[0], color="steelblue")
        axes[0].set_title("Transaction Amount Distribution")
        axes[0].set_xlabel("Amount ($)")
        sns.histplot(np.log1p(amount_sample["amount"]), bins=bins, ax=axes[1], color="teal")
        axes[1].set_title("Log-transformed Amount")
        axes[1].set_xlabel("log(1 + Amount)")
        plt.tight_layout()
        plt.savefig("plot_01_amount_distribution.png", dpi=150)

    def monthly_transactions(self) -> None:
        """Visualize monthly transaction volume and average amount over time"""
        monthly = q(self.con, """
            SELECT txn_year, txn_month,
                COUNT(*)        AS txn_count,
                AVG(amount)     AS avg_amount,
                MEDIAN(amount)  AS median_amount
            FROM transactions
            GROUP BY txn_year, txn_month
            ORDER BY txn_year, txn_month
        """)
        monthly["period"] = pd.to_datetime(
            monthly["txn_year"].astype(str) + "-" + monthly["txn_month"].astype(str).str.zfill(2)
        )
        _, ax = plt.subplots(figsize=self.figsize)
        ax.plot(monthly["period"], monthly["txn_count"], marker="o", linewidth=1.5)
        ax.set_title("Monthly Transaction Volume")
        ax.set_xlabel("Month")
        ax.set_ylabel("Transactions")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("plot_02_volume_over_time.png", dpi=150)

    def patterns(self) -> None:
        """Visualize transaction patterns by hour of day and day of week"""
        time_stats = q(self.con, """
            SELECT txn_hour, txn_dow,
                COUNT(*)    AS txn_count,
                AVG(amount) AS avg_amount,
                SUM(has_error) AS error_count
            FROM transactions
            GROUP BY txn_hour, txn_dow
        """)
        hourly = time_stats.groupby("txn_hour")["txn_count"].sum().sort_index()
        daily  = time_stats.groupby("txn_dow")["txn_count"].sum().sort_index()
        dow_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        
        _, axes = plt.subplots(1, 2, figsize=self.figsize)
        hourly.plot(kind="bar", ax=axes[0], color="steelblue")
        axes[0].set_title("Transactions by Hour of Day")
        axes[0].set_xlabel("Hour")
        daily.rename(index=dict(enumerate(dow_labels))).plot(kind="bar", ax=axes[1], color="coral")
        axes[1].set_title("Transactions by Day of Week")
        axes[1].set_xlabel("Day of Week")
        plt.tight_layout()
        plt.savefig("plot_03_time_patterns.png", dpi=150)

    def errors(self) -> None:
        """Visualize the most common transaction error types and the prevalence of cards on the dark web"""
        error_counts = q(self.con, """
            SELECT errors, COUNT(*) AS n
            FROM transactions
            WHERE has_error = 1
            GROUP BY errors
            ORDER BY n DESC
            LIMIT 10
        """)
        dark_web = q(self.con, """
            SELECT card_on_dark_web, COUNT(*) AS n
            FROM cards
            GROUP BY card_on_dark_web
        """)
        _, axes = plt.subplots(1, 2, figsize=self.figsize)
        axes[0].barh(error_counts["errors"], error_counts["n"], color="salmon")
        axes[0].set_title("Top Transaction Error Types")
        axes[0].invert_yaxis()
        dark_web.set_index("card_on_dark_web")["n"].plot(
            kind="pie", ax=axes[1], autopct="%1.1f%%",
            colors=["#69b3a2","#ff6b6b"], startangle=90)
        axes[1].set_title("Card on Dark Web")
        axes[1].set_ylabel("")
        plt.tight_layout()
        plt.savefig("plot_04_errors_darkweb.png", dpi=150)

    def score_by_gender(self) -> None:
        """Visualize credit score distributions by gender using KDE plots"""
        user_scores = q(self.con, "SELECT gender, credit_score FROM users WHERE credit_score IS NOT NULL")
        _, ax = plt.subplots(figsize=self.figsize)
        for gender, grp in user_scores.groupby("gender"):
            sns.kdeplot(grp["credit_score"], ax=ax, label=gender, fill=True, alpha=0.4)
        ax.set_title("Credit Score Distribution by Gender")
        ax.set_xlabel("Credit Score")
        ax.legend()
        plt.tight_layout()
        plt.savefig("plot_05_credit_score_gender.png", dpi=150)

    def correlations(self) -> None:
        """Visualize correlations between key user-level features using a heatmap"""
        self.user_agg["debt_to_income"] = \
            self.user_agg["total_debt"] / self.user_agg["yearly_income"].replace(0, np.nan)
        self.user_agg["income_per_card"] = \
            self.user_agg["yearly_income"] / self.user_agg["num_credit_cards"].replace(0, np.nan)

        corr_cols = [
            "avg_amount","credit_score","yearly_income","total_debt","per_capita_income",
            "avg_credit_limit","num_credit_cards","debt_to_income","current_age",
            "txn_count","total_errors","chip_ratio","has_dark_web_card"
        ]
        _, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(self.user_agg[corr_cols].corr(), annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, linewidths=0.5, ax=ax)
        ax.set_title("Correlation Matrix  (per-user aggregates)")
        plt.tight_layout()
        plt.savefig("plot_07_correlation_heatmap.png", dpi=150)

    def anomalies(self) -> None:
        """Detect anomalous users based on aggregated transaction and user features using Isolation Forest"""
        iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=42, n_jobs=-1)
        self.user_agg = self.user_agg.loc[self.user_agg.index].copy()

        feature_cols = [c for c in self.user_agg.columns if c not in ["anomaly", "anomaly_score"]]
        X_scaled = StandardScaler().fit_transform(self.user_agg[feature_cols])

        self.user_agg["anomaly"] = iso.fit_predict(X_scaled)
        self.user_agg["anomaly_score"] = iso.decision_function(X_scaled)

        n_anomalies = (self.user_agg["anomaly"] == -1).sum()
        print(f"Flagged {n_anomalies} anomalous users "
            f"({n_anomalies / len(self.user_agg) * 100:.1f}%)")

        _, ax = plt.subplots(figsize=self.figsize)
        sns.histplot(self.user_agg["anomaly_score"], bins=60, color="steelblue", ax=ax)
        ax.axvline(0, color="red", linestyle="--", label="Decision boundary")
        ax.set_title("Isolation Forest Anomaly Score Distribution")
        ax.set_xlabel("Score  (lower = more anomalous)")
        ax.legend()
        plt.tight_layout()
        plt.savefig("plot_11_anomaly_scores.png", dpi=150)

        _, ax = plt.subplots(figsize=self.figsize)
        sns.violinplot(data=self.user_agg, x="anomaly", y="avg_amount",
                    order=[1, -1], ax=ax,
                    palette={'1': "#69b3a2", '-1': "#ff6b6b"})
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Normal", "Anomaly"])
        ax.set_title("Avg Transaction Amount: Normal vs Anomalous Users")
        plt.tight_layout()
        plt.savefig("plot_12_anomaly_amount.png", dpi=150)

    def summary(self) -> None:
        """Print a summary of key statistics and findings from the EDA"""
        totals = q(self.con, "SELECT COUNT(*) AS n, AVG(amount) AS avg, MEDIAN(amount) AS med FROM transactions")
        errors = q(self.con, "SELECT SUM(has_error) AS n, AVG(has_error)*100 AS pct FROM transactions")
        dark   = q(self.con, "SELECT COUNT(*) AS n FROM cards WHERE card_on_dark_web = 'Yes'")

        n_anomalies = (self.user_agg["anomaly"] == -1).sum()

        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Total transactions   : {int(totals['n'][0]):>12,}")
        print(f"Unique users         : {self.user_agg['user_id'].nunique():>12,}")
        print(f"Avg txn amount       : {totals['avg'][0]:>12,.2f} $")
        print(f"Median txn amount    : {totals['med'][0]:>12,.2f} $")
        print(f"Transactions w/ error: {int(errors['n'][0]):>12,}  ({errors['pct'][0]:.2f}%)")
        print(f"Cards on dark web    : {int(dark['n'][0]):>12,}")
        print(f"Anomalous users      : {n_anomalies:>12,}  ({n_anomalies/len(self.user_agg)*100:.2f}%)")
        print("="*60)

    def _get_user_agg(self) -> pd.DataFrame:
        return q(self.con, """
            SELECT
                t.user_id,
                COUNT(*)            AS txn_count,
                AVG(t.amount)       AS avg_amount,
                SUM(t.has_error)    AS total_errors,
                AVG(t.used_chip)    AS chip_ratio,
                u.credit_score,
                u.yearly_income,
                u.total_debt,
                u.per_capita_income,
                u.num_credit_cards,
                u.current_age,
                u.retirement_age,
                AVG(c.credit_limit) AS avg_credit_limit,
                MAX(CASE WHEN c.card_on_dark_web = 'Yes' THEN 1 ELSE 0 END) AS has_dark_web_card
            FROM transactions t
            JOIN users u ON t.user_id = u.user_id
            JOIN cards c ON t.card_id = c.card_id AND t.user_id = c.user_id
            GROUP BY t.user_id, u.credit_score, u.yearly_income, u.total_debt,
                    u.per_capita_income, u.num_credit_cards, u.current_age, u.retirement_age
        """)

    def _setup_database(self) -> None:
        self.con.execute(f"""
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

        self.con.execute(f"""
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

        self.con.execute(f"""
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

        print(q(self.con, "SELECT COUNT(*) AS total_transactions FROM transactions").to_string(index=False))