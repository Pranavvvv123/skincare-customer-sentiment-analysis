
# Skincare Customer Sentiment and Review Theme Analysis

## Project Overview
This project analyses customer reviews of skincare products using Python. It aims to understand customer satisfaction, ratings, and feedback through exploratory data analysis.

**Status:** Updated prototype using a real Sephora customer review dataset.

## Business Problem
Star ratings show how customers rate skincare products, but they do not fully explain why customers like or dislike them. This project explores ratings and written reviews to identify customer opinions and common feedback patterns.

## Research Questions
1. What is the distribution of customer ratings?
2. Which brands receive the most reviews?
3. What are the most common words in customer reviews?
4. What proportion of reviews are positive, neutral, and negative based on ratings?
5. What patterns can be identified from customer feedback?

## Dataset
The project uses a 10,000-review sample from the Sephora Products and Skincare Reviews dataset.

**Dataset file:** `sephora_skincare_reviews_sample.csv`

### Important Features
- `review_text`: Written customer feedback
- `rating`: Product rating
- `is_recommended`: Recommendation status
- `submission_time`: Review date
- `skin_type`: Customer skin type
- `product_name`: Product name
- `brand_name`: Brand name

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Analysis Performed
- Dataset overview
- Missing value analysis
- Duplicate review removal
- Rating distribution
- Rating-based sentiment classification
- Brand-wise review analysis
- Common word frequency
- Data visualization

## Sentiment Classification
Sentiment is initially classified using customer ratings:
- 4–5 stars: Positive
- 3 stars: Neutral
- 1–2 stars: Negative

These are rating-based sentiment labels, not predictions from an AI text classifier.

## Project Structure
```text
skincare-customer-sentiment-analysis/
├── README.md
├── analysis.py
├── requirements.txt
└── sephora_skincare_reviews_sample.csv
```

## How to Run
Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Python script:

```bash
python analysis.py
```

## Limitations and Ethics
- The analysis uses a 10,000-review sample, not the complete dataset.
- Rating-based sentiment may not always match the actual meaning of review text.
- Customer reviews may contain personal opinions and biases.
- Reviews are opinions and not clinical evidence of product safety or effectiveness.
- Dataset usage must follow its applicable licence and terms.

## Future Scope
- Advanced text preprocessing
- Positive and negative review theme extraction
- Text-based sentiment classification
- Product-wise customer feedback comparison
- Improved data visualizations
