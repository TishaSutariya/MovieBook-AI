import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MOVIE_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "movies_cleaned.csv"
)


# ============================================================
# MOVIE RECOMMENDER
# ============================================================

class MovieRecommender:

    def __init__(self):

        print("\n" + "=" * 40)
        print("MOVIE RECOMMENDATION MODEL")
        print("=" * 40)

        # ----------------------------------------------------
        # Load processed movie dataset
        # ----------------------------------------------------

        self.movies = pd.read_csv(MOVIE_FILE)

        print(
            f"Movies loaded: {len(self.movies)}"
        )

        # ----------------------------------------------------
        # Make sure required columns exist
        # ----------------------------------------------------

        required_columns = [
            "title",
            "director",
            "cast",
            "genre",
            "overview",
            "year",
            "source",
            "combined_text"
        ]

        for column in required_columns:

            if column not in self.movies.columns:
                self.movies[column] = ""

        # ----------------------------------------------------
        # Clean combined text
        # ----------------------------------------------------

        self.movies["combined_text"] = (
            self.movies["combined_text"]
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

        self.tfidf_matrix = (
            self.vectorizer.fit_transform(
                self.movies["combined_text"]
            )
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
            self.movies.index,
            index=self.movies["title"]
            .astype(str)
            .str.lower()
        ).drop_duplicates()

        print(
            "TF-IDF matrix shape:",
            self.tfidf_matrix.shape
        )

        print("Movie recommendation model ready.")


    # ========================================================
    # RECOMMEND MOVIES
    # ========================================================

    def recommend(self, movie_title, n=10):

        movie_title = str(movie_title).strip()

        # ----------------------------------------------------
        # Exact title search
        # ----------------------------------------------------

        title_key = movie_title.lower()

        if title_key in self.title_index:

            movie_index = self.title_index[title_key]

        else:

            # ------------------------------------------------
            # Partial title search
            # ------------------------------------------------

            matches = self.movies[
                self.movies["title"]
                .astype(str)
                .str.lower()
                .str.contains(
                    title_key,
                    na=False
                )
            ]

            if matches.empty:

                return pd.DataFrame()

            movie_index = matches.index[0]

        # ----------------------------------------------------
        # Similarity scores
        # ----------------------------------------------------

        similarity_scores = list(
            enumerate(
                self.similarity_matrix[movie_index]
            )
        )

        similarity_scores = sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True
        )

        # Skip the selected movie itself
        similarity_scores = [
            item
            for item in similarity_scores
            if item[0] != movie_index
        ]

        top_movies = similarity_scores[:n]

        movie_indices = [
            item[0]
            for item in top_movies
        ]

        result = self.movies.loc[
            movie_indices,
            [
                "title",
                "genre",
                "director",
                "year",
                "source"
            ]
        ].copy()

        result["similarity"] = [
            round(item[1], 4)
            for item in top_movies
        ]

        return result.reset_index(drop=True)


    # ========================================================
    # SEARCH MOVIES
    # ========================================================

    def search_movies(
        self,
        query,
        limit=20
    ):

        query = str(query).strip().lower()

        if not query:
            return pd.DataFrame()

        matches = self.movies[
            self.movies["title"]
            .astype(str)
            .str.lower()
            .str.contains(
                query,
                na=False
            )
        ]

        return matches[
            [
                "title",
                "genre",
                "director",
                "year",
                "source"
            ]
        ].head(limit).reset_index(drop=True)


# ============================================================
# TEST MODEL
# ============================================================

if __name__ == "__main__":

    recommender = MovieRecommender()

    print("\nExample: Recommendations for Jawan\n")

    recommendations = recommender.recommend(
        "Jawan",
        n=10
    )

    print(recommendations.to_string(index=False))