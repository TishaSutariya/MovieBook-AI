import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"


# ============================================================
# MOVIE EDA
# ============================================================

def movie_eda():

    file_path = PROCESSED_DIR / "movies_cleaned.csv"

    df = pd.read_csv(file_path)

    print("\n")
    print("=" * 60)
    print("MOVIE EDA")
    print("=" * 60)

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumns:")
    print(list(df.columns))

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\nMissing Values:")

    print(
        df.isnull()
        .sum()
        .sort_values(ascending=False)
    )

    # --------------------------------------------------------
    # Source distribution
    # --------------------------------------------------------

    if "source" in df.columns:

        print("\nMovie Source:")

        print(
            df["source"]
            .value_counts()
        )

    # --------------------------------------------------------
    # Top genres
    # --------------------------------------------------------

    if "genre" in df.columns:

        genres = (
            df["genre"]
            .dropna()
            .astype(str)
            .str.split(",")
            .explode()
            .str.strip()
            .value_counts()
            .head(10)
        )

        print("\nTop Movie Genres:")

        print(genres)

    # --------------------------------------------------------
    # Release years
    # --------------------------------------------------------

    if "year" in df.columns:

        print("\nTop Movie Release Years:")

        print(
            df["year"]
            .value_counts()
            .head(10)
        )

    # --------------------------------------------------------
    # Directors
    # --------------------------------------------------------

    if "director" in df.columns:

        print("\nTop Directors:")

        print(
            df["director"]
            .replace("Unknown", pd.NA)
            .dropna()
            .value_counts()
            .head(10)
        )

    return df


# ============================================================
# BOOK EDA
# ============================================================

def book_eda():

    file_path = PROCESSED_DIR / "books_cleaned.csv"

    df = pd.read_csv(file_path)

    print("\n")
    print("=" * 60)
    print("BOOK EDA")
    print("=" * 60)

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumns:")
    print(list(df.columns))

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\nMissing Values:")

    print(
        df.isnull()
        .sum()
        .sort_values(ascending=False)
        .head(15)
    )

    # --------------------------------------------------------
    # Categories
    # --------------------------------------------------------

    if "categories" in df.columns:

        categories = (
            df["categories"]
            .dropna()
            .astype(str)
            .str.split(",")
            .explode()
            .str.strip()
            .value_counts()
            .head(10)
        )

        print("\nTop Book Categories:")

        print(categories)

    # --------------------------------------------------------
    # Authors
    # --------------------------------------------------------

    if "authors" in df.columns:

        print("\nTop Authors:")

        print(
            df["authors"]
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .head(10)
        )

    # --------------------------------------------------------
    # Rating analysis
    # --------------------------------------------------------

    if "average_rating" in df.columns:

        print("\nAverage Rating Statistics:")

        print(
            df["average_rating"]
            .describe()
        )

    # --------------------------------------------------------
    # Most rated books
    # --------------------------------------------------------

    if "ratings_count" in df.columns:

        print("\nMost Rated Books:")

        columns = [
            "title",
            "authors",
            "average_rating",
            "ratings_count"
        ]

        available_columns = [
            col for col in columns
            if col in df.columns
        ]

        print(
            df[
                available_columns
            ]
            .sort_values(
                "ratings_count",
                ascending=False
            )
            .head(10)
        )

    return df


# ============================================================
# RUN ALL EDA
# ============================================================

def run_eda():

    movie_eda()
    book_eda()


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":
    run_eda()