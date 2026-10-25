"""Starter analysis. The included CSV contains fictional demo data only."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "sample_reviews.csv")

print("Dataset preview:\n", df.head())
print("\nRows and columns:", df.shape)
print("\nMissing values:\n", df.isna().sum())
print("\nRating summary:\n", df["rating"].describe())
print("\nAverage rating by product:\n",
      df.groupby("product_name")["rating"].mean().sort_values(ascending=False))

def rating_label(rating):
    if rating >= 4:
        return "Positive"
    if rating == 3:
        return "Neutral"
    return "Negative"

df["rating_sentiment"] = df["rating"].apply(rating_label)
print("\nRating-derived sentiment counts:\n", df["rating_sentiment"].value_counts())

df["rating"].value_counts().sort_index().plot(kind="bar")
plt.title("Rating Counts (Fictional Demo Data)")
plt.xlabel("Star rating")
plt.ylabel("Number of reviews")
plt.tight_layout()
plt.savefig(BASE_DIR / "rating_counts.png", dpi=150)
plt.close()

df["rating_sentiment"].value_counts().plot(kind="bar")
plt.title("Rating-Derived Sentiment (Fictional Demo Data)")
plt.xlabel("Sentiment label")
plt.ylabel("Number of reviews")
plt.tight_layout()
plt.savefig(BASE_DIR / "sentiment_counts.png", dpi=150)
plt.close()

print("\nSaved rating_counts.png and sentiment_counts.png")
print("Reminder: this is a starter workflow, not analysis of real customer data.")
