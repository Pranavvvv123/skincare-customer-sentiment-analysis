# Skincare Customer Sentiment and Review Theme Analysis

## Project overview
This beginner-friendly project proposes a data science workflow for analysing online skincare reviews. The goal is to explore ratings and review text to understand customer feedback.

**Status:** Starter prototype. `sample_reviews.csv` contains fictional demonstration data only, not real customer reviews. Do not use it to make claims about actual products or customers.

## Business problem
Star ratings show how a customer rated a product, but do not fully explain why. This project will explore ratings and written reviews to identify common themes such as texture, packaging, price, and perceived results.

## Research questions
1. What is the rating distribution?
2. Which themes appear in positive and negative reviews?
3. Do star ratings agree with sentiment expressed in review text?
4. What further analysis is possible with a properly sourced dataset?

## Files
- `analysis.py`: starter exploratory analysis
- `sample_reviews.csv`: fictional demo data
- `requirements.txt`: required Python libraries

## Run the project
Install Python 3.10+, open a terminal in this folder, then run:
```bash
pip install -r requirements.txt
python analysis.py
```
The script prints basic summaries and saves two charts.

## Next steps
1. Find a public skincare/beauty review dataset and check its licence and terms.
2. Replace the fictional demo data with permitted real data.
3. Add data cleaning, text preprocessing, and review-theme analysis.
4. Evaluate any sentiment classifier and document limitations.

## Limitations and ethics
Reviews may be biased and may not represent all buyers. Rating-derived sentiment is only a proxy. Reviews are opinions, not clinical evidence of safety or effectiveness. Use only data you are permitted to access and reuse, and do not publish personal information.
