from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Advanced Customer Review Sentiment Analysis
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
CHART_DIR = OUTPUT_DIR / "charts"
OUTPUT_DIR.mkdir(exist_ok=True)
CHART_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")
analyzer = SentimentIntensityAnalyzer()

# Load reviews.csv if provided; otherwise create sample reviews
input_file = BASE_DIR / "reviews.csv"

if input_file.exists():
    df = pd.read_csv(input_file)
    print(f"Loaded reviews from: {input_file.name}")
else:
    sample_reviews = [
        ("R001", "This product is amazing, I absolutely love it!", 5),
        ("R002", "Excellent quality and fast delivery.", 5),
        ("R003", "Very happy with my purchase.", 5),
        ("R004", "The product is okay, nothing special.", 3),
        ("R005", "It works as expected.", 3),
        ("R006", "Average quality for the price.", 3),
        ("R007", "Terrible quality, very disappointed.", 1),
        ("R008", "The item broke after one day.", 1),
        ("R009", "Poor service and late delivery.", 1),
        ("R010", "Good value and easy to use.", 4),
        ("R011", "Not bad, but could be better.", 3),
        ("R012", "I would not recommend this product.", 1),
        ("R013", "Fantastic experience, highly recommended!", 5),
        ("R014", "The package arrived late.", 2),
        ("R015", "It is fine, neither good nor bad.", 3),
    ]
    df = pd.DataFrame(
        sample_reviews, columns=["ReviewID", "Review", "Rating"]
    )
    df.to_csv(input_file, index=False)
    print("Created sample reviews.csv")

# Validate and clean input
if "Review" not in df.columns:
    raise ValueError(
        "Input CSV must contain a column named 'Review'."
    )

df = df.drop_duplicates().copy()
df["Review"] = df["Review"].fillna("").astype(str).str.strip()
df = df[df["Review"] != ""].copy()

if df.empty:
    raise ValueError("No non-empty reviews found in the dataset.")

# Calculate sentiment scores and labels
def analyze_review(review):
    scores = analyzer.polarity_scores(review)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return pd.Series({
        "PositiveScore": scores["pos"],
        "NeutralScore": scores["neu"],
        "NegativeScore": scores["neg"],
        "CompoundScore": compound,
        "Sentiment": label
    })

sentiment_columns = df["Review"].apply(analyze_review)
df = pd.concat([df.reset_index(drop=True),
                sentiment_columns.reset_index(drop=True)], axis=1)

# Summary metrics
counts = df["Sentiment"].value_counts().reindex(
    ["Positive", "Neutral", "Negative"], fill_value=0
)
percentages = (counts / len(df) * 100).round(2)

print("\n" + "=" * 45)
print("CUSTOMER REVIEW SENTIMENT ANALYSIS")
print("=" * 45)
print(f"Total reviews analysed: {len(df)}")
print("\nSentiment counts:")
print(counts)
print("\nSentiment percentages:")
print(percentages.astype(str) + "%")
print(f"\nAverage compound score: {df['CompoundScore'].mean():.3f}")

# Export results
df.to_csv(OUTPUT_DIR / "sentiment_results.csv", index=False)
summary = pd.DataFrame({
    "Sentiment": counts.index,
    "Count": counts.values,
    "Percentage": percentages.values
})
summary.to_csv(OUTPUT_DIR / "sentiment_summary.csv", index=False)

# Chart 1: Sentiment counts
plt.figure(figsize=(8, 5))
sns.barplot(data=summary, x="Sentiment", y="Count", hue="Sentiment",
            legend=False)
plt.title("Customer Review Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig(CHART_DIR / "sentiment_distribution.png", dpi=150)
plt.show()
plt.close()

# Chart 2: Sentiment percentages
plt.figure(figsize=(8, 5))
plt.pie(
    summary["Count"],
    labels=summary["Sentiment"],
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Share of Positive, Neutral and Negative Reviews")
plt.tight_layout()
plt.savefig(CHART_DIR / "sentiment_share.png", dpi=150)
plt.show()
plt.close()

# Chart 3: Compound score distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="CompoundScore", bins=15, kde=True)
plt.axvline(0, linestyle="--")
plt.title("Distribution of Sentiment Scores")
plt.xlabel("Compound Score (-1 to +1)")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig(CHART_DIR / "sentiment_scores.png", dpi=150)
plt.show()
plt.close()

# Report
report = f"""
CUSTOMER SENTIMENT ANALYSIS REPORT
==================================
Reviews analysed: {len(df)}
Average compound score: {df['CompoundScore'].mean():.3f}

SENTIMENT COUNTS:
{counts.to_string()}

SENTIMENT PERCENTAGES:
{percentages.to_string()}%

Interpretation:
Positive scores indicate more positive language; negative scores
indicate more negative language. Neutral scores are near zero.
Automated sentiment can misinterpret sarcasm, context and language.
Review labels are model estimates, not guaranteed facts.

Files saved:
- output/sentiment_results.csv
- output/sentiment_summary.csv
- output/charts/
"""

(OUTPUT_DIR / "summary_report.txt").write_text(
    report, encoding="utf-8"
)

print(report)
print("Task 4 completed successfully!")
print(f"Results saved in: {OUTPUT_DIR}")