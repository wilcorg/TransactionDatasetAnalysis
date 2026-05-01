from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow", dtype_backend="pyarrow")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, engine="pyarrow", dtype_backend="pyarrow")


def read_json(path: Path, **kwargs: Any) -> pd.api.typing.JsonReader:
    # pandas developers are retarded
    # parameter `typ` builds different objects depending on the value
    return pd.read_json(path, dtype_backend="pyarrow", **kwargs)


def to_parquet(df: pd.DataFrame, out_file: Path) -> None:
    df.to_parquet(out_file, compression="zstd")
    del df

    # Workaround for compatibility between numpy and pyarrow
    table: pq.Table = pq.read_table(out_file).replace_schema_metadata(None)
    pq.write_table(table, out_file)
