# Task 4: Advanced Sentiment Analysis

## Overview
Analyse customer reviews using the VADER sentiment analysis model.

## Features
- Positive, neutral and negative classification
- Compound sentiment scores
- Sentiment percentages and summary statistics
- Three visualizations
- Exported review-level results and summary report

## Technologies
Python, Pandas, Matplotlib, Seaborn and VADER.

## How to Run
```bash
python -m pip install -r requirements.txt
python sentiment_analysis.py
```

## Input Data
Optionally add a `reviews.csv` file with a column named `Review`.
If no input file exists, the project creates illustrative sample reviews.

## Outputs
Generated CSV files, charts and a summary report are saved in `output/`.
Sentiment labels are automated estimates and may not always be correct.