
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter

# Load the Sephora dataset
df = pd.read_csv("sephora_skincare_reviews_sample.csv")

print("SEPHORA CUSTOMER REVIEW ANALYSIS")
print("--------------------------------")

# Basic dataset information
print("\nDataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicate reviews
df = df.drop_duplicates(subset=["review_text"])

# Clean review text
df["review_text"] = df["review_text"].fillna("").astype(str)
df["review_text"] = df["review_text"].str.lower()
df["review_text"] = df["review_text"].apply(
    lambda x: re.sub(r"[^a-z\s]", " ", x)
)
df["review_text"] = df["review_text"].str.replace(
    r"\s+", " ", regex=True
).str.strip()

# Rating analysis
print("\nRating Statistics:")
print(df["rating"].describe())

print("\nRating Counts:")
print(df["rating"].value_counts().sort_index())

# Sentiment based on star ratings
def get_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

df["sentiment"] = df["rating"].apply(get_sentiment)

print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())

# Top reviewed brands
print("\nTop 10 Brands:")
print(df["brand_name"].value_counts().head(10))

# Plot rating distribution
df["rating"].value_counts().sort_index().plot(
    kind="bar", title="Customer Rating Distribution"
)
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig("rating_distribution.png")
plt.close()

# Plot sentiment distribution
df["sentiment"].value_counts().plot(
    kind="bar", title="Customer Sentiment Distribution"
)
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig("sentiment_distribution.png")
plt.close()

# Find common words in reviews
all_words = " ".join(df["review_text"]).split()
stop_words = {
    "the", "and", "is", "it", "to", "of", "a", "in",
    "for", "this", "that", "my", "with", "on", "was",
    "but", "i", "have", "very", "as", "are", "be"
}

words = [
    word for word in all_words
    if len(word) > 2 and word not in stop_words
]

common_words = Counter(words).most_common(15)

print("\nTop 15 Common Words:")
for word, count in common_words:
    print(word, ":", count)

print("\nAnalysis completed successfully!")
