import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

BOOK_DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "books_cleaned.csv"
)


# ============================================================
# BOOK RECOMMENDER
# ============================================================

class BookRecommender:

    def __init__(self):

        # ----------------------------------------------------
        # Load processed dataset
        # ----------------------------------------------------

        self.books = pd.read_csv(BOOK_DATA_PATH)

        print("\nBook dataset loaded:")
        print(f"Books: {len(self.books)}")

        # ----------------------------------------------------
        # Make sure combined_text exists
        # ----------------------------------------------------

        if "combined_text" not in self.books.columns:

            self.books["combined_text"] = (
                self.books["title"].fillna("").astype(str)
                + " "
                + self.books["subtitle"].fillna("").astype(str)
                + " "
                + self.books["authors"].fillna("").astype(str)
                + " "
                + self.books["categories"].fillna("").astype(str)
                + " "
                + self.books["description"].fillna("").astype(str)
            ).str.lower()

        # ----------------------------------------------------
        # Fill missing values
        # ----------------------------------------------------

        self.books["combined_text"] = (
            self.books["combined_text"]
            .fillna("")
            .astype(str)
        )

        # ----------------------------------------------------
        # TF-IDF
        # ----------------------------------------------------

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=20000,
            ngram_range=(1, 2)
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.books["combined_text"]
        )

        print(
            f"TF-IDF matrix shape: "
            f"{self.tfidf_matrix.shape}"
        )

        # ----------------------------------------------------
        # Cosine similarity
        # ----------------------------------------------------

        self.similarity_matrix = cosine_similarity(
            self.tfidf_matrix
        )

        # ----------------------------------------------------
        # Title index
        # ----------------------------------------------------

        self.title_index = pd.Series(
            self.books.index,
            index=self.books["title"]
            .str.lower()
            .str.strip()
        ).drop_duplicates()

        print("Book recommendation model ready.")

    # ========================================================
    # RECOMMEND
    # ========================================================

    def recommend(self, book_title, n=10):

        query = book_title.lower().strip()

        # ----------------------------------------------------
        # Exact title
        # ----------------------------------------------------

        if query in self.title_index:

            idx = self.title_index[query]

        else:

            # ------------------------------------------------
            # Partial title search
            # ------------------------------------------------

            matches = self.books[
                self.books["title"]
                .str.lower()
                .str.contains(
                    query,
                    na=False
                )
            ]

            if matches.empty:

                return pd.DataFrame()

            idx = matches.index[0]

        # ----------------------------------------------------
        # Similarity scores
        # ----------------------------------------------------

        similarity_scores = list(
            enumerate(
                self.similarity_matrix[idx]
            )
        )

        similarity_scores = sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True
        )

        # Remove selected book itself
        similarity_scores = [
            item
            for item in similarity_scores
            if item[0] != idx
        ]

        top_results = similarity_scores[:n]

        # ----------------------------------------------------
        # Build result
        # ----------------------------------------------------

        result_indices = [
            item[0]
            for item in top_results
        ]

        result_scores = [
            item[1]
            for item in top_results
        ]

        results = self.books.loc[
            result_indices
        ].copy()

        results["similarity"] = result_scores

        # ----------------------------------------------------
        # Display columns
        # ----------------------------------------------------

        columns = [
            "title",
            "authors",
            "categories",
            "average_rating",
            "ratings_count",
            "similarity"
        ]

        available_columns = [
            col
            for col in columns
            if col in results.columns
        ]

        return results[
            available_columns
        ].reset_index(drop=True)

    # ========================================================
    # SEARCH BOOKS
    # ========================================================

    def search_books(self, query, limit=20):

        query = query.lower().strip()

        results = self.books[
            self.books["title"]
            .str.lower()
            .str.contains(
                query,
                na=False
            )
        ].copy()

        return results.head(limit)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 40)
    print("BOOK RECOMMENDATION MODEL")
    print("=" * 40)

    recommender = BookRecommender()

    print("\nExample recommendations:")

    recommendations = recommender.recommend(
        "Harry Potter and the Sorcerer's Stone",
        n=10
    )

    if recommendations.empty:

        print("No recommendations found.")

    else:

        print(
            recommendations.to_string(
                index=False
            )
        )