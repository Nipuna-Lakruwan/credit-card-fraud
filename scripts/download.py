"""Download the ULB credit-card fraud dataset from OpenML (dataset 1597) and save it as parquet.

The original data: Dal Pozzolo et al., ULB Machine Learning Group, licensed ODbL / DbCL.
OpenML mirrors it with no login, so no Kaggle account is needed.
"""
import hashlib
from pathlib import Path

import openml
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "creditcard.parquet"


def main() -> None:
    openml.config.set_root_cache_directory(str(OUT.parent / ".openml"))
    ds = openml.datasets.get_dataset(1597, download_data=True)
    # Time is marked as the row-id attribute on OpenML, so ask for it explicitly.
    X, y, _, _ = ds.get_data(target=ds.default_target_attribute, include_row_id=True)
    df = X.copy()
    df["Class"] = y.astype(int)
    OUT.parent.mkdir(exist_ok=True)
    df.to_parquet(OUT, index=False)
    digest = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]
    print(f"{len(df):,} rows, {int(df['Class'].sum())} fraud, columns={list(df.columns)[:4]}..., content hash {digest}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
