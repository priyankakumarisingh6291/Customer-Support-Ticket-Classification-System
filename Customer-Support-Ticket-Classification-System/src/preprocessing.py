"""Data loading, cleaning and text preprocessing (Task 1)."""
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "tickets.csv"


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation/digits/symbols, collapse extra spaces.

    Stop-words are NOT removed on purpose: words like "not" and "cannot" carry
    meaning for tickets ("password reset is NOT working").
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)   
    text = re.sub(r"\s+", " ", text).strip()  
    return text


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Full cleaning pipeline. Returns a new cleaned DataFrame."""
    df = df.copy()
    if verbose:
        print("Rows before cleaning:", len(df))
        print("\nMissing values per column:\n", df.isnull().sum())
        print("\nExact duplicate rows (all columns):", df.duplicated().sum())


    df = df.dropna(subset=["ticket_description", "category"])

    # 2. Other columns: fill missing values with a clear placeholder
    df["priority"] = df["priority"].fillna("Unknown")
    df["status"] = df["status"].fillna("Unknown")
    df["customer_name"] = df["customer_name"].fillna("Unknown")


    
    df["clean_description"] = df["ticket_description"].apply(clean_text)
    before = len(df)
    df = df.drop_duplicates(subset=["clean_description", "category"]).reset_index(drop=True)
    if verbose:
        print(f"Duplicates removed (after text cleaning): {before - len(df)}")
        print("Rows after cleaning:", len(df))
    return df


def load_clean(verbose: bool = False) -> pd.DataFrame:
    return prepare(load_raw(), verbose=verbose)


if __name__ == "__main__":
    raw = load_raw()
    print("=== DATASET (first 10 rows) ===")
    print(raw.head(10).to_string(index=False))
    print("\n=== INFO ===")
    raw.info()
    clean = prepare(raw, verbose=True)
    print("\n=== CLEANED SAMPLE ===")
    print(clean[["ticket_description", "clean_description", "category"]].head(8).to_string(index=False))
