from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MOVIE PREPROCESSING
# ============================================================

def preprocess_movies():

    print("\n" + "=" * 70)
    print("MOVIE PREPROCESSING")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD NETFLIX DATA
    # --------------------------------------------------------

    netflix_path = (
        RAW_DIR / "movies_on_netflix.csv"
    )

    netflix = pd.read_csv(
        netflix_path
    )

    print(
        f"\nNetflix raw shape: {netflix.shape}"
    )

    # --------------------------------------------------------
    # NORMALIZE TYPE
    # --------------------------------------------------------

    if "type" in netflix.columns:

        netflix["type"] = (
            netflix["type"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    print(
        "\nNetflix content types BEFORE filtering:"
    )

    if "type" in netflix.columns:

        print(
            netflix["type"].value_counts()
        )

    # --------------------------------------------------------
    # KEEP MOVIES ONLY
    # --------------------------------------------------------

    if "type" in netflix.columns:

        netflix = netflix[
            netflix["type"] == "movie"
        ].copy()

    print(
        f"\nNetflix movies AFTER type filtering: "
        f"{len(netflix)}"
    )

    # --------------------------------------------------------
    # REMOVE KNOWN INCORRECT TV SHOW
    # --------------------------------------------------------

    bad_title = (
        "My Next Guest with David Letterman "
        "and Shah Rukh Khan"
    )

    if "title" in netflix.columns:

        before_count = len(netflix)

        netflix = netflix[
            netflix["title"].astype(str).str.strip()
            != bad_title
        ].copy()

        removed_count = (
            before_count - len(netflix)
        )

        print(
            f"\nManual TV-show cleanup removed: "
            f"{removed_count} record(s)"
        )

    # --------------------------------------------------------
    # KEEP IMPORTANT NETFLIX COLUMNS
    # --------------------------------------------------------

    netflix_columns = [
        "title",
        "director",
        "cast",
        "country",
        "listed_in",
        "description",
        "release_year"
    ]

    available_columns = [
        column
        for column in netflix_columns
        if column in netflix.columns
    ]

    netflix = netflix[
        available_columns
    ].copy()

    # --------------------------------------------------------
    # RENAME NETFLIX COLUMNS
    # --------------------------------------------------------

    netflix = netflix.rename(
        columns={
            "listed_in": "genre",
            "description": "overview",
            "release_year": "year"
        }
    )

    netflix["source"] = "Netflix"

    # --------------------------------------------------------
    # LOAD IMDb DATA
    # --------------------------------------------------------

    imdb_path = (
        RAW_DIR
        / "IMDB-Movie-Dataset(2023-1951).csv"
    )

    imdb = pd.read_csv(
        imdb_path
    )

    print(
        f"\nIMDb raw shape: {imdb.shape}"
    )

    # --------------------------------------------------------
    # RENAME IMDb COLUMNS
    # --------------------------------------------------------

    imdb = imdb.rename(
        columns={
            "movie_name": "title"
        }
    )

    # IMDb dataset does not contain country.
    # We create the column so both datasets have
    # the same structure.

    imdb["country"] = ""

    imdb["source"] = "IMDb"

    # --------------------------------------------------------
    # KEEP IMPORTANT IMDb COLUMNS
    # --------------------------------------------------------

    imdb_columns = [
        "title",
        "director",
        "cast",
        "country",
        "genre",
        "overview",
        "year",
        "source"
    ]

    available_imdb_columns = [
        column
        for column in imdb_columns
        if column in imdb.columns
    ]

    imdb = imdb[
        available_imdb_columns
    ].copy()

    # --------------------------------------------------------
    # MAKE COLUMN STRUCTURE IDENTICAL
    # --------------------------------------------------------

    required_columns = [
        "title",
        "director",
        "cast",
        "country",
        "genre",
        "overview",
        "year",
        "source"
    ]

    for column in required_columns:

        if column not in netflix.columns:

            netflix[column] = ""

        if column not in imdb.columns:

            imdb[column] = ""

    netflix = netflix[
        required_columns
    ]

    imdb = imdb[
        required_columns
    ]

    # --------------------------------------------------------
    # COMBINE MOVIE DATASETS
    # --------------------------------------------------------

    movies = pd.concat(
        [
            netflix,
            imdb
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # CLEAN TITLE
    # --------------------------------------------------------

    movies["title"] = (
        movies["title"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Remove empty titles

    movies = movies[
        movies["title"] != ""
    ].copy()

    # --------------------------------------------------------
    # REMOVE DUPLICATE TITLES
    # --------------------------------------------------------

    movies = movies.drop_duplicates(
        subset=["title"],
        keep="first"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # CLEAN TEXT COLUMNS
    # --------------------------------------------------------

    text_columns = [
        "director",
        "cast",
        "country",
        "genre",
        "overview"
    ]

    for column in text_columns:

        movies[column] = (
            movies[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    # --------------------------------------------------------
    # CLEAN YEAR
    # --------------------------------------------------------

    movies["year"] = pd.to_numeric(
        movies["year"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # CREATE COMBINED TEXT
    # --------------------------------------------------------

    movies["combined_text"] = (
        movies["title"] + " "
        + movies["director"] + " "
        + movies["cast"] + " "
        + movies["genre"] + " "
        + movies["overview"]
    )

    # --------------------------------------------------------
    # CREATE INDIAN / BOLLYWOOD FLAG
    # --------------------------------------------------------

    movies["is_indian"] = False

    # Netflix country information

    country_text = (
        movies["country"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    # Indian country keywords

    indian_country_pattern = (
        r"\bindia\b|"
        r"\bindian\b"
    )

    country_match = (
        country_text
        .str.contains(
            indian_country_pattern,
            regex=True,
            na=False
        )
    )

    movies.loc[
        country_match,
        "is_indian"
    ] = True

    # --------------------------------------------------------
    # IMDb INDIAN MOVIE DETECTION
    # --------------------------------------------------------
    #
    # The IMDb dataset does not contain country.
    # Therefore Indian movies are identified using
    # Indian-language / Indian cinema signals in the
    # available metadata.
    #
    # This is intentionally conservative.
    # --------------------------------------------------------

    imdb_mask = (
        movies["source"]
        .astype(str)
        .str.lower()
        .eq("imdb")
    )

    indian_language_pattern = (
        r"hindi|"
        r"tamil|"
        r"telugu|"
        r"malayalam|"
        r"kannada|"
        r"marathi|"
        r"bengali|"
        r"punjabi|"
        r"gujarati"
    )

    imdb_text = (
        movies["genre"].fillna("")
        + " "
        + movies["overview"].fillna("")
        + " "
        + movies["cast"].fillna("")
        + " "
        + movies["director"].fillna("")
    ).str.lower()

    indian_language_match = (
        imdb_text.str.contains(
            indian_language_pattern,
            regex=True,
            na=False
        )
    )

    movies.loc[
        imdb_mask & indian_language_match,
        "is_indian"
    ] = True

    # --------------------------------------------------------
    # MANUAL INDIAN MOVIE SIGNALS
    # --------------------------------------------------------

    indian_title_keywords = [
        "jawan",
        "pathaan",
        "dunki",
        "fighter",
        "animal",
        "tiger",
        "war",
        "brahmastra",
        "kgf",
        "rrr",
        "pushpa",
        "baahubali",
        "drishyam",
        "dangal",
        "3 idiots",
        "pk",
        "kabir singh",
        "om shanti om",
        "chennai express",
        "happy new year",
        "zero",
        "sanak",
        "thugs"
    ]

    title_lower = (
        movies["title"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    title_match = pd.Series(
        False,
        index=movies.index
    )

    for keyword in indian_title_keywords:

        title_match = (
            title_match
            | title_lower.str.contains(
                keyword,
                regex=False,
                na=False
            )
        )

    movies.loc[
        title_match,
        "is_indian"
    ] = True

    # --------------------------------------------------------
    # SAVE MOVIE DATA
    # --------------------------------------------------------

    movie_output = (
        PROCESSED_DIR
        / "movies_cleaned.csv"
    )

    movies.to_csv(
        movie_output,
        index=False
    )

    print(
        f"\nFinal movie dataset shape:"
    )

    print(
        movies.shape
    )

    print(
        "\nMovie sources:"
    )

    print(
        movies["source"].value_counts()
    )

    print(
        "\nIndian movies detected:"
    )

    print(
        movies["is_indian"].value_counts()
    )

    print(
        f"\nMovie data saved to:\n"
        f"{movie_output}"
    )

    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    bad_tv_check = movies[
        movies["title"]
        .str.contains(
            "My Next Guest",
            case=False,
            na=False
        )
    ]

    if len(bad_tv_check) == 0:

        print(
            "\nTV Show check: PASSED"
        )

    else:

        print(
            "\nWARNING: TV Show check failed"
        )

    return movies


# ============================================================
# BOOK PREPROCESSING
# ============================================================

def preprocess_books():

    print("\n" + "=" * 70)
    print("BOOK PREPROCESSING")
    print("=" * 70)

    book_path = (
        RAW_DIR / "data.csv"
    )

    books = pd.read_csv(
        book_path
    )

    print(
        f"\nRaw book shape: {books.shape}"
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATE TITLES
    # --------------------------------------------------------

    books["title"] = (
        books["title"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    books = books[
        books["title"] != ""
    ].copy()

    books = books.drop_duplicates(
        subset=["title"],
        keep="first"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # CLEAN TEXT COLUMNS
    # --------------------------------------------------------

    text_columns = [
        "title",
        "subtitle",
        "authors",
        "categories",
        "description"
    ]

    for column in text_columns:

        if column in books.columns:

            books[column] = (
                books[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

    # --------------------------------------------------------
    # NUMERIC COLUMNS
    # --------------------------------------------------------

    numeric_columns = [
        "published_year",
        "average_rating",
        "num_pages",
        "ratings_count"
    ]

    for column in numeric_columns:

        if column in books.columns:

            books[column] = pd.to_numeric(
                books[column],
                errors="coerce"
            )

    # --------------------------------------------------------
    # CREATE COMBINED TEXT
    # --------------------------------------------------------

    books["combined_text"] = (
        books["title"] + " "
        + books["subtitle"] + " "
        + books["authors"] + " "
        + books["categories"] + " "
        + books["description"]
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    book_output = (
        PROCESSED_DIR
        / "books_cleaned.csv"
    )

    books.to_csv(
        book_output,
        index=False
    )

    print(
        f"\nBook dataset shape:"
    )

    print(
        books.shape
    )

    print(
        f"\nBook data saved to:\n"
        f"{book_output}"
    )

    return books


# ============================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================

def run_preprocessing():

    print("\n")
    print("=" * 70)
    print("MOVIEBOOK AI - DATA PREPROCESSING")
    print("=" * 70)

    preprocess_movies()

    preprocess_books()

    print("\n")
    print("=" * 70)
    print("PREPROCESSING COMPLETED")
    print("=" * 70)


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    run_preprocessing()