from pathlib import Path
import time

import requests
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"
OUTPUT_DIR = Path("output")
CHART_DIR = OUTPUT_DIR / "charts"
OUTPUT_DIR.mkdir(exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; EducationalDataScraper/1.0)"
}

RATING_MAP = {
    "One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5
}


def scrape_page(page_number):
    url = BASE_URL.format(page_number)

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.RequestException as error:
        print(f"Page {page_number} failed: {error}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    books = []

    for item in soup.select("article.product_pod"):
        title_tag = item.select_one("h3 a")
        price_tag = item.select_one(".price_color")
        availability_tag = item.select_one(".availability")
        rating_tag = item.select_one(".star-rating")

        if not all([title_tag, price_tag, availability_tag, rating_tag]):
            continue

        try:
            price = float(
                price_tag.get_text(strip=True).replace("£", "").replace("Â", "")
            )
        except ValueError:
            continue

        rating_name = next(
            (name for name in RATING_MAP if name in rating_tag.get("class", [])),
            "Unknown"
        )

        books.append({
            "title": title_tag.get("title", title_tag.get_text(strip=True)),
            "price_gbp": price,
            "rating": RATING_MAP.get(rating_name),
            "availability": availability_tag.get_text(" ", strip=True),
            "product_url": "https://books.toscrape.com/catalogue/"
                           + title_tag.get("href", "").replace("../", "")
        })

    return books


def create_charts(df):
    if df.empty:
        print("No data available for charts.")
        return

    plt.figure(figsize=(9, 5))
    df["rating"].dropna().value_counts().sort_index().plot(
        kind="bar", title="Books by Star Rating"
    )
    plt.xlabel("Star Rating")
    plt.ylabel("Number of Books")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "ratings_distribution.png", dpi=150)
    plt.show()
    plt.close()

    plt.figure(figsize=(9, 5))
    df["price_gbp"].plot(kind="hist", bins=15, edgecolor="black",
                         title="Book Price Distribution")
    plt.xlabel("Price (£)")
    plt.ylabel("Number of Books")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "price_distribution.png", dpi=150)
    plt.show()
    plt.close()


def main():
    all_books = []

    # Scrape first 3 pages; each page has up to 20 books.
    for page in range(1, 4):
        print(f"Scraping page {page}...")
        all_books.extend(scrape_page(page))
        time.sleep(1)  # Be polite to the practice website.

    if not all_books:
        print("No books collected. Check your internet connection and try again.")
        return

    df = pd.DataFrame(all_books).drop_duplicates(subset="product_url")
    df.to_csv(OUTPUT_DIR / "books_data.csv", index=False)

    create_charts(df)

    report = (
        "WEB SCRAPING SUMMARY\n"
        "====================\n"
        f"Books collected: {len(df)}\n"
        f"Average price: £{df['price_gbp'].mean():.2f}\n"
        f"Average rating: {df['rating'].mean():.2f}/5\n"
        f"CSV file: {OUTPUT_DIR / 'books_data.csv'}\n"
    )
    (OUTPUT_DIR / "summary_report.txt").write_text(report, encoding="utf-8")

    print("\n" + report)
    print("Task 1 completed successfully!")


if __name__ == "__main__":
    main()