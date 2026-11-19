"""
Week 4: Exploratory Data Analysis and Feature Engineering
Project: Skincare Customer Sentiment and Review Theme Analysis

Run from this folder:
    pip install -r requirements.txt
    python week4_eda.py

The script prefers sephora_skincare_reviews_cleaned.csv if present.
Otherwise it uses sephora_skincare_reviews_sample.csv and applies light,
documented preparation without modifying the original CSV.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent
CHART_DIR = BASE / "week4_eda_visualizations"
CHART_DIR.mkdir(exist_ok=True)

cleaned_file = BASE / "sephora_skincare_reviews_cleaned.csv"
raw_file = BASE / "sephora_skincare_reviews_sample.csv"
input_file = cleaned_file if cleaned_file.exists() else raw_file
if not input_file.exists():
    raise FileNotFoundError(
        "Place sephora_skincare_reviews_cleaned.csv or "
        "sephora_skincare_reviews_sample.csv in this folder."
    )

df = pd.read_csv(input_file)
initial_rows = len(df)

# Light fallback preparation when the Week 3 cleaned file is not present.
for col in ["rating", "helpfulness", "total_feedback_count",
            "total_neg_feedback_count", "total_pos_feedback_count", "price_usd"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "skin_type" in df.columns:
    df["skin_type_clean"] = df["skin_type"].fillna("Unknown").astype(str).str.strip()
    df.loc[df["skin_type_clean"].isin(["", "nan", "None"]), "skin_type_clean"] = "Unknown"

if "review_text" in df.columns:
    text = df["review_text"].fillna("").astype(str)
    df["review_text_clean"] = text.str.replace(r"\s+", " ", regex=True).str.strip()
    df["review_word_count"] = df["review_text_clean"].str.findall(r"\b\w+\b").str.len()
    df["review_char_count"] = df["review_text_clean"].str.len()

if "submission_time" in df.columns:
    df["review_date"] = pd.to_datetime(df["submission_time"], errors="coerce", utc=True)
    df["review_month"] = df["review_date"].dt.month
    df["review_year"] = df["review_date"].dt.year

if "rating" in df.columns:
    df["rating_sentiment"] = np.select(
        [df["rating"].between(4, 5), df["rating"].eq(3), df["rating"].between(1, 2)],
        ["Positive", "Neutral", "Negative"], default="Unknown"
    )

# Summary metrics
summary_rows = [
    ("input_file", input_file.name),
    ("rows", len(df)),
    ("columns_after_feature_engineering", len(df.columns)),
    ("exact_duplicate_rows", int(df.duplicated().sum())),
]
if "rating" in df.columns:
    summary_rows += [
        ("mean_rating", round(float(df["rating"].mean()), 4)),
        ("median_rating", round(float(df["rating"].median()), 4)),
        ("rating_missing", int(df["rating"].isna().sum())),
    ]
summary = pd.DataFrame(summary_rows, columns=["metric", "value"])
summary.to_csv(BASE / "week4_eda_summary.csv", index=False)

# Missing-value profile
missing = df.isna().sum().rename("missing_count").to_frame()
missing["missing_percent"] = (missing["missing_count"] / max(len(df), 1) * 100).round(2)
missing.sort_values("missing_count", ascending=False).to_csv(BASE / "week4_missing_value_profile.csv")

sns.set_theme(style="whitegrid")

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(CHART_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close()

# 1. Rating distribution
if "rating" in df.columns:
    plt.figure(figsize=(7, 4))
    order = sorted(df["rating"].dropna().unique())
    df["rating"].value_counts().sort_index().plot(kind="bar")
    plt.title("Customer Rating Distribution")
    plt.xlabel("Rating (stars)")
    plt.ylabel("Number of reviews")
    save_plot("01_rating_distribution.png")

# 2. Rating-derived sentiment
if "rating_sentiment" in df.columns:
    plt.figure(figsize=(7, 4))
    order = ["Positive", "Neutral", "Negative", "Unknown"]
    df["rating_sentiment"].value_counts().reindex(order, fill_value=0).plot(kind="bar")
    plt.title("Rating-derived Sentiment Distribution")
    plt.xlabel("Sentiment group")
    plt.ylabel("Number of reviews")
    save_plot("02_rating_sentiment.png")

# 3. Top brands by number of reviews
if "brand_name" in df.columns:
    top_brands = df["brand_name"].fillna("Unknown").value_counts().head(15).index
    brand_data = df[df["brand_name"].fillna("Unknown").isin(top_brands)].copy()
    plt.figure(figsize=(10, 6))
    brand_data["brand_name"].value_counts().plot(kind="barh")
    plt.title("Top 15 Brands by Review Count")
    plt.xlabel("Number of reviews")
    plt.ylabel("Brand")
    save_plot("03_top_brands_by_reviews.png")

# 4. Review length by rating-derived sentiment
if {"review_word_count", "rating_sentiment"}.issubset(df.columns):
    plot_df = df[df["rating_sentiment"].isin(["Positive", "Neutral", "Negative"])].copy()
    plt.figure(figsize=(8, 4))
    [plt.boxplot([plot_df.loc[plot_df["rating_sentiment"] == g, "review_word_count"].dropna().values
                 for g in ["Positive", "Neutral", "Negative"]],
                labels=["Positive", "Neutral", "Negative"], showfliers=False)]
    plt.title("Review Word Count by Rating-derived Sentiment")
    plt.xlabel("Rating-derived sentiment")
    plt.ylabel("Words per review")
    save_plot("04_review_length_by_sentiment.png")

# 5. Average rating for brands with at least 30 reviews
if {"brand_name", "rating"}.issubset(df.columns):
    brand_stats = df.groupby("brand_name", dropna=False).agg(
        review_count=("rating", "count"), mean_rating=("rating", "mean")
    )
    brand_stats = brand_stats[brand_stats["review_count"] >= 30].sort_values("mean_rating", ascending=False).head(15)
    brand_stats.to_csv(BASE / "week4_brand_rating_summary.csv")
    if not brand_stats.empty:
        plt.figure(figsize=(10, 6))
        plt.barh(brand_stats.index.astype(str), brand_stats["mean_rating"])
        plt.xlim(0, 5)
        plt.title("Highest Mean Ratings: Brands with at Least 30 Rated Reviews")
        plt.xlabel("Mean rating (1–5)")
        plt.ylabel("Brand")
        save_plot("05_brand_mean_rating_min30.png")

# 6. Skin type versus rating
if {"skin_type_clean", "rating"}.issubset(df.columns):
    skin = df.groupby("skin_type_clean").agg(
        review_count=("rating", "count"), mean_rating=("rating", "mean")
    ).sort_values("review_count", ascending=False).head(12)
    skin.to_csv(BASE / "week4_skin_type_rating_summary.csv")
    if not skin.empty:
        plt.figure(figsize=(9, 5))
        plt.barh(skin.index.astype(str), skin["mean_rating"])
        plt.xlim(0, 5)
        plt.title("Mean Rating by Skin Type (Top 12 by Review Count)")
        plt.xlabel("Mean rating (1–5)")
        plt.ylabel("Skin type")
        save_plot("06_skin_type_mean_rating.png")

# 7. Monthly review counts
if "review_date" in df.columns:
    dated = df.dropna(subset=["review_date"]).copy()
    if not dated.empty:
        monthly = dated.set_index("review_date").resample("MS").size().rename("review_count")
        monthly.to_csv(BASE / "week4_monthly_review_counts.csv")
        plt.figure(figsize=(10, 4))
        monthly.plot()
        plt.title("Monthly Review Volume")
        plt.xlabel("Month")
        plt.ylabel("Number of reviews")
        save_plot("07_monthly_review_volume.png")

# 8. Spearman correlation for selected numeric fields
numeric_cols = [c for c in ["rating", "helpfulness", "total_feedback_count",
                            "total_neg_feedback_count", "total_pos_feedback_count",
                            "price_usd", "review_word_count"] if c in df.columns]
if len(numeric_cols) >= 2:
    corr = df[numeric_cols].corr(method="spearman")
    corr.to_csv(BASE / "week4_spearman_correlation.csv")
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, aspect="auto", vmin=-1, vmax=1); plt.colorbar(); plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right"); plt.yticks(range(len(corr.index)), corr.index); [plt.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center") for i in range(len(corr.index)) for j in range(len(corr.columns))]
    plt.title("Spearman Correlation of Selected Numeric Features")
    save_plot("08_spearman_correlation_heatmap.png")

# Save feature-engineered data separately; original input remains unchanged.
df.to_csv(BASE / "week4_feature_engineered_dataset.csv", index=False)

print(f"Input used: {input_file.name}")
print(f"Rows analysed: {initial_rows}")
print(f"Columns after feature engineering: {len(df.columns)}")
print(f"Charts saved in: {CHART_DIR}")
print("Summary and feature-engineered dataset saved in the project folder.")
