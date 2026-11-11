"""
Data cleaning and preprocessing pipeline for the Sephora skincare review sample.
Run from the project folder:
    python data_cleaning.py

The original input CSV is never overwritten. The script creates:
    sephora_skincare_reviews_cleaned.csv
    cleaning_summary.csv

Note: rating_sentiment is derived from star ratings; it is not NLP sentiment.
"""
from pathlib import Path
import re
import html
import numpy as np
import pandas as pd

INPUT_FILE = "sephora_skincare_reviews_sample.csv"
OUTPUT_FILE = "sephora_skincare_reviews_cleaned.csv"
SUMMARY_FILE = "cleaning_summary.csv"

def clean_text(value):
    if pd.isna(value):
        return ""
    value = html.unescape(str(value)).strip()
    return re.sub(r"\\s+", " ", value)

def rating_to_sentiment(value):
    if pd.isna(value):
        return "Unknown"
    if value >= 4:
        return "Positive"
    if value == 3:
        return "Neutral"
    if value >= 1:
        return "Negative"
    return "Invalid"

def main():
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} not found. Place this script beside the dataset CSV."
        )

    raw = pd.read_csv(input_path)
    df = raw.copy()
    rows_before, cols_before = df.shape
    missing_before = df.isna().sum()

    required = ["rating", "review_text", "product_id", "product_name", "brand_name"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(f"Required columns missing: {missing_required}")

    
    df["review_text_missing"] = (
        df["review_text"].isna()
        | (df["review_text"].astype("string").str.strip() == "")
    )
    df["review_text_clean"] = df["review_text"].apply(clean_text)

    
    for col in ["review_title", "product_name", "brand_name"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().replace("", pd.NA)

    
    numeric_cols = [
        "rating", "helpfulness", "total_feedback_count",
        "total_neg_feedback_count", "total_pos_feedback_count", "price_usd"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    invalid_rating = df["rating"].notna() & ~df["rating"].between(1, 5)
    invalid_rating_count = int(invalid_rating.sum())
    df.loc[invalid_rating, "rating"] = np.nan

    negative_counts = {}
    for col in [
        "total_feedback_count", "total_neg_feedback_count",
        "total_pos_feedback_count", "price_usd"
    ]:
        if col in df.columns:
            mask = df[col] < 0
            negative_counts[col] = int(mask.sum())
            df.loc[mask, col] = np.nan

    
    date_parse_failures = 0
    if "submission_time" in df.columns:
        parsed = pd.to_datetime(df["submission_time"], errors="coerce", utc=True)
        date_parse_failures = int(parsed.isna().sum())
        df["submission_time_clean"] = parsed

    
    for col in ["skin_tone", "eye_color", "skin_type", "hair_color"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
            df[col] = df[col].fillna("Unknown")

    
    exact_duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    
    df["rating_sentiment"] = df["rating"].apply(rating_to_sentiment)

    df.to_csv(OUTPUT_FILE, index=False)

    missing_after = df.isna().sum()
    summary = []
    for col in raw.columns:
        summary.append({
            "metric": "missing_values",
            "column": col,
            "before": int(missing_before.get(col, 0)),
            "after": int(missing_after.get(col, 0)),
            "notes": "After field-specific processing; Unknown is not counted as null."
        })

    summary.extend([
        {"metric": "dataset_rows", "column": "all", "before": rows_before,
         "after": len(df), "notes": "After exact duplicate removal."},
        {"metric": "dataset_columns", "column": "all", "before": cols_before,
         "after": len(df.columns), "notes": "Processed fields added."},
        {"metric": "exact_duplicate_rows", "column": "all", "before": exact_duplicates,
         "after": 0, "notes": "Exact duplicate rows removed."},
        {"metric": "invalid_rating_values", "column": "rating", "before": invalid_rating_count,
         "after": 0, "notes": "Out-of-range ratings changed to missing."},
        {"metric": "date_parse_failures", "column": "submission_time", "before": date_parse_failures,
         "after": date_parse_failures, "notes": "Unparseable timestamps are NaT in standardized field."}
    ])
    for col, count in negative_counts.items():
        summary.append({
            "metric": "negative_values_flagged", "column": col,
            "before": count, "after": 0,
            "notes": "Negative values changed to missing."
        })

    pd.DataFrame(summary).to_csv(SUMMARY_FILE, index=False)
    print("Cleaning completed.")
    print(f"Rows: {rows_before} -> {len(df)}")
    print(f"Exact duplicate rows removed: {exact_duplicates}")
    print(f"Invalid ratings flagged: {invalid_rating_count}")
    print(f"Date parsing failures: {date_parse_failures}")
    print(f"Saved: {OUTPUT_FILE} and {SUMMARY_FILE}")

if __name__ == "__main__":
    main()
