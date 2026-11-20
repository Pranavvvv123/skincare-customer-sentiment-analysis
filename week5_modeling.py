"""
YuvaIntern Week 5 - Predictive Modeling and Algorithm Selection
Project: Skincare Customer Sentiment and Review Theme Analysis

Run:
    pip install pandas numpy scikit-learn matplotlib
    python week5_modeling.py

Place this script in the same folder as sephora_skincare_reviews_cleaned.csv.
It creates a week5_outputs folder containing metrics, predictions, dataset summary,
and confusion-matrix images.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

DATA_FILE = "sephora_skincare_reviews_cleaned.csv"
OUTPUT_DIR = "week5_outputs"
RANDOM_STATE = 42

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_FILE)

    df["is_recommended"] = pd.to_numeric(df["is_recommended"], errors="coerce")
    df = df[df["is_recommended"].isin([0, 1])].copy()
    df["is_recommended"] = df["is_recommended"].astype(int)

    text_col = "review_text_clean" if "review_text_clean" in df.columns else "review_text"
    df[text_col] = df[text_col].fillna("").astype(str)
    for col in ["skin_type", "brand_name"]:
        df[col] = df[col].fillna("Unknown").astype(str)
    if "price_usd" not in df.columns:
        df["price_usd"] = np.nan

    # Rating and feedback fields are excluded to avoid target leakage.
    feature_cols = [text_col, "skin_type", "brand_name", "price_usd"]
    X = df[feature_cols]
    y = df["is_recommended"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    preprocessing = ColumnTransformer([
        ("review_text", TfidfVectorizer(
            max_features=3000, ngram_range=(1, 2), min_df=3,
            max_df=0.95, stop_words="english"
        ), text_col),
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         ["skin_type", "brand_name"]),
        ("price", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False))
        ]), ["price_usd"])
    ])

    model_specs = {
        "Majority-class baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced",
            solver="liblinear", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, min_samples_split=30, min_samples_leaf=15,
            class_weight="balanced", random_state=RANDOM_STATE
        )
    }

    results, predictions, fitted = [], {}, {}
    for name, estimator in model_specs.items():
        pipe = Pipeline([("preprocessing", preprocessing), ("model", estimator)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        fitted[name] = pipe
        predictions[name] = y_pred

        auc = np.nan
        if hasattr(pipe, "predict_proba"):
            try:
                auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
            except ValueError:
                pass

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Macro Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "Macro Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "Macro F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "ROC-AUC": auc
        })

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    for name in ["Logistic Regression", "Decision Tree"]:
        pipe = Pipeline([("preprocessing", preprocessing), ("model", model_specs[name])])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
        for row in results:
            if row["Model"] == name:
                row["3-fold CV Macro F1 mean"] = scores.mean()
                row["3-fold CV Macro F1 std"] = scores.std()

    metrics = pd.DataFrame(results)
    metrics.to_csv(os.path.join(OUTPUT_DIR, "week5_model_comparison.csv"), index=False)

    prediction_frame = pd.DataFrame({"actual_is_recommended": y_test.to_numpy()})
    for name, pred in predictions.items():
        prediction_frame[name.lower().replace(" ", "_") + "_prediction"] = pred
    prediction_frame.to_csv(os.path.join(OUTPUT_DIR, "week5_test_predictions.csv"), index=False)

    summary = pd.DataFrame([{
        "original_rows": pd.read_csv(DATA_FILE).shape[0],
        "valid_target_rows": len(df),
        "missing_or_invalid_target_rows": pd.read_csv(DATA_FILE).shape[0] - len(df),
        "recommended_count": int((y == 1).sum()),
        "not_recommended_count": int((y == 0).sum()),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features_used": ", ".join(feature_cols),
        "excluded_to_avoid_leakage": "rating, rating_sentiment, helpfulness and feedback counts"
    }])
    summary.to_csv(os.path.join(OUTPUT_DIR, "week5_dataset_summary.csv"), index=False)

    for name in ["Logistic Regression", "Decision Tree"]:
        cm = confusion_matrix(y_test, predictions[name], labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm)
        ax.set_title(name + " - Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1], ["Not recommended", "Recommended"], rotation=15)
        ax.set_yticks([0, 1], ["Not recommended", "Recommended"])
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, str(value), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(os.path.join(
            OUTPUT_DIR, name.lower().replace(" ", "_") + "_confusion_matrix.png"
        ), dpi=160)
        plt.close(fig)

    print(metrics.to_string(index=False))
    print("Saved all Week 5 outputs in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
