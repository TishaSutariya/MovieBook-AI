import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.movie_recommender import MovieRecommender
from src.book_recommender import BookRecommender


# ============================================================
# DATA PATHS
# ============================================================

MOVIE_DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "movies_cleaned.csv"
)

BOOK_DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "books_cleaned.csv"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MovieBook AI",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_movie_data():
    if not MOVIE_DATA_PATH.exists():
        st.error(
            f"Movie dataset not found:\n{MOVIE_DATA_PATH}"
        )
        st.stop()

    return pd.read_csv(MOVIE_DATA_PATH)


@st.cache_data
def load_book_data():
    if not BOOK_DATA_PATH.exists():
        st.error(
            f"Book dataset not found:\n{BOOK_DATA_PATH}"
        )
        st.stop()

    return pd.read_csv(BOOK_DATA_PATH)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_movie_model():
    return MovieRecommender()


@st.cache_resource
def load_book_model():
    return BookRecommender()


movies = load_movie_data()
books = load_book_data()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎬📚 MovieBook AI")

st.sidebar.caption(
    "Movie + Book Recommendation System"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🎬 Bollywood Movie Recommendation",
        "🎬 All Movie Recommendation",
        "📚 Book Recommendation",
        "📊 Movie Analytics",
        "📖 Book Analytics",
        "🤖 AI / NLP Insights"
    ]
)

st.sidebar.markdown("---")

st.sidebar.metric(
    "🎬 Movies",
    f"{len(movies):,}"
)

st.sidebar.metric(
    "📚 Books",
    f"{len(books):,}"
)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.title("🎬📚 MovieBook AI")

    st.subheader(
        "Movie and Book Recommendation System"
    )

    st.write(
        """
        MovieBook AI is a content-based recommendation system
        that uses Natural Language Processing and Machine
        Learning techniques to recommend similar movies and books.
        """
    )

    st.markdown("---")

    st.header("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🎬 Movies",
            f"{len(movies):,}"
        )

    with col2:
        st.metric(
            "📚 Books",
            f"{len(books):,}"
        )

    with col3:
        if "source" in movies.columns:
            netflix_count = (
                movies["source"]
                .astype(str)
                .str.lower()
                .eq("netflix")
                .sum()
            )
        else:
            netflix_count = 0

        st.metric(
            "Netflix",
            f"{netflix_count:,}"
        )

    with col4:
        if "is_indian" in movies.columns:
            indian_count = (
                movies["is_indian"]
                .fillna(False)
                .astype(bool)
                .sum()
            )
        else:
            indian_count = 0

        st.metric(
            "🇮🇳 Indian Movies",
            f"{indian_count:,}"
        )

    st.markdown("---")

    # ========================================================
    # HOW IT WORKS
    # ========================================================

    st.header("🧠 How MovieBook AI Works")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            ### 📂 1. Data Processing

            Movie and book datasets are cleaned
            and prepared for recommendation.
            """
        )

    with col2:
        st.markdown(
            """
            ### 🧠 2. NLP & Features

            Movie and book text is converted
            into TF-IDF numerical features.
            """
        )

    with col3:
        st.markdown(
            """
            ### 🎯 3. Similarity Matching

            Cosine similarity compares content
            and ranks similar items.
            """
        )

    st.markdown("---")

    st.header("🇮🇳 Indian Movie Recommendation")

    st.write(
        """
        Choose an Indian movie and discover other
        Indian movies with similar content.
        """
    )

    st.info(
        """
        Similarity percentage is a converted cosine-similarity
        score. It represents textual/content similarity and
        is not a probability or movie rating.
        """
    )


# ============================================================
# INDIAN MOVIE RECOMMENDATION
# ============================================================

elif page == "🎬 Bollywood Movie Recommendation":

    st.title(
        "🇮🇳 Bollywood Movie Recommendation"
    )

    st.write(
        """
        This section recommends only Indian movies from
        the processed dataset.
        """
    )

    if "is_indian" not in movies.columns:
        st.error(
            "Indian movie information is missing. "
            "Run the preprocessing script again."
        )
        st.stop()

    indian_movies = movies[
        movies["is_indian"]
        .fillna(False)
        .astype(bool)
    ].copy()

    if indian_movies.empty:
        st.warning(
            "No Indian movies were detected."
        )
        st.stop()

    st.metric(
        "🇮🇳 Indian Movies Available",
        f"{len(indian_movies):,}"
    )

    st.markdown("---")

    movie_model = load_movie_model()

    indian_titles = sorted(
        indian_movies["title"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_movie = st.selectbox(
        "🎬 Select Bollywood / Indian Movie",
        indian_titles
    )

    number_of_recommendations = st.slider(
        "Number of Recommendations",
        5,
        15,
        10,
        key="bollywood_slider"
    )

    if st.button(
        "🇮🇳 Find Similar Bollywood Movies",
        type="primary"
    ):

        with st.spinner(
            "Finding similar Indian movies..."
        ):

            # Get a larger pool first.
            recommendations = movie_model.recommend(
                selected_movie,
                n=100
            )

        if recommendations.empty:

            st.warning(
                "No recommendations found."
            )

        else:

            # ------------------------------------------------
            # FILTER RECOMMENDATIONS TO INDIAN MOVIES
            # ------------------------------------------------

            indian_titles_set = set(
                indian_movies["title"]
                .astype(str)
                .str.lower()
            )

            recommendations = recommendations[
                recommendations["title"]
                .astype(str)
                .str.lower()
                .isin(indian_titles_set)
            ]

            recommendations = (
                recommendations
                .head(number_of_recommendations)
            )

            if recommendations.empty:

                st.warning(
                    "No other Indian movies were found "
                    "for this selection."
                )

            else:

                st.success(
                    f"Movies similar to: {selected_movie}"
                )

                st.markdown("---")

                for rank, (_, row) in enumerate(
                    recommendations.iterrows(),
                    start=1
                ):

                    col1, col2 = st.columns([5, 1])

                    with col1:

                        st.subheader(
                            f"#{rank} 🎬 {row['title']}"
                        )

                        genre = row.get(
                            "genre",
                            "Not available"
                        )

                        if pd.isna(genre):
                            genre = "Not available"

                        st.write(
                            f"**Genre:** {genre}"
                        )

                        director = row.get(
                            "director",
                            "Not available"
                        )

                        if (
                            pd.isna(director)
                            or not str(director).strip()
                        ):
                            director = "Not available"

                        st.write(
                            f"**Director:** {director}"
                        )

                        year = row.get(
                            "year",
                            None
                        )

                        if pd.notna(year):

                            try:
                                year = int(
                                    float(year)
                                )
                            except (ValueError, TypeError):
                                pass

                        else:
                            year = "N/A"

                        st.write(
                            f"**Year:** {year}"
                        )

                    with col2:

                        similarity = float(
                            row.get(
                                "similarity",
                                0
                            )
                        )

                        similarity_percentage = (
                            similarity * 100
                        )

                        st.metric(
                            "Similarity",
                            f"{similarity_percentage:.1f}%"
                        )

                    st.divider()


# ============================================================
# ALL MOVIE RECOMMENDATION
# ============================================================

elif page == "🎬 All Movie Recommendation":

    st.title(
        "🎬 All Movie Recommendation"
    )

    st.write(
        """
        Search the complete movie dataset and find
        content-based recommendations.
        """
    )

    movie_model = load_movie_model()

    movie_titles = sorted(
        movies["title"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_movie = st.selectbox(
        "Select Movie",
        movie_titles
    )

    number_of_recommendations = st.slider(
        "Number of Recommendations",
        5,
        15,
        10,
        key="all_movie_slider"
    )

    if st.button(
        "🔍 Recommend Movies",
        type="primary"
    ):

        with st.spinner(
            "Finding similar movies..."
        ):

            recommendations = movie_model.recommend(
                selected_movie,
                n=number_of_recommendations
            )

        if recommendations.empty:

            st.warning(
                "No recommendations found."
            )

        else:

            for rank, (_, row) in enumerate(
                recommendations.iterrows(),
                start=1
            ):

                col1, col2 = st.columns([5, 1])

                with col1:

                    st.subheader(
                        f"#{rank} 🎬 {row['title']}"
                    )

                    st.write(
                        f"**Genre:** "
                        f"{row.get('genre', 'Not available')}"
                    )

                    st.write(
                        f"**Director:** "
                        f"{row.get('director', 'Not available')}"
                    )

                    st.write(
                        f"**Year:** "
                        f"{row.get('year', 'N/A')}"
                    )

                    st.caption(
                        f"Source: "
                        f"{row.get('source', 'Unknown')}"
                    )

                with col2:

                    similarity = float(
                        row.get(
                            "similarity",
                            0
                        )
                    )

                    st.metric(
                        "Similarity",
                        f"{similarity * 100:.1f}%"
                    )

                st.divider()


# ============================================================
# BOOK RECOMMENDATION
# ============================================================

elif page == "📚 Book Recommendation":

    st.title(
        "📚 Book Recommendation"
    )

    st.write(
        """
        Select a book to discover similar books.
        Recommendations are organized into different
        content categories.
        """
    )

    book_model = load_book_model()

    book_titles = sorted(
        books["title"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_book = st.selectbox(
        "📚 Select Book",
        book_titles
    )

    number_of_recommendations = st.slider(
        "Number of Recommendations",
        5,
        15,
        10,
        key="book_slider"
    )

    if st.button(
        "🔍 Find Similar Books",
        type="primary"
    ):

        with st.spinner(
            "Finding similar books..."
        ):

            recommendations = book_model.recommend(
                selected_book,
                n=number_of_recommendations
            )

        if recommendations.empty:

            st.warning(
                "No recommendations found."
            )

        else:

            # =================================================
            # OVERALL SIMILARITY
            # =================================================

            st.header(
                "⭐ Overall Similar Books"
            )

            st.caption(
                "Ranked using the complete TF-IDF content representation."
            )

            for rank, (_, row) in enumerate(
                recommendations.iterrows(),
                start=1
            ):

                col1, col2 = st.columns([5, 1])

                with col1:

                    st.subheader(
                        f"#{rank} 📚 {row['title']}"
                    )

                    st.write(
                        f"**Author:** "
                        f"{row.get('authors', 'Not available')}"
                    )

                    st.write(
                        f"**Category:** "
                        f"{row.get('categories', 'Not available')}"
                    )

                with col2:

                    similarity = float(
                        row.get(
                            "similarity",
                            0
                        )
                    )

                    st.metric(
                        "Similarity",
                        f"{similarity * 100:.1f}%"
                    )

                st.divider()

            # =================================================
            # CATEGORY-BASED
            # =================================================

            st.header(
                "📖 Similar by Category"
            )

            selected_row = books[
                books["title"]
                .astype(str)
                .str.lower()
                == selected_book.lower()
            ]

            if not selected_row.empty:

                selected_categories = str(
                    selected_row.iloc[0].get(
                        "categories",
                        ""
                    )
                ).lower()

                if selected_categories.strip():

                    category_mask = (
                        books["categories"]
                        .fillna("")
                        .astype(str)
                        .str.lower()
                        .str.contains(
                            selected_categories,
                            regex=False,
                            na=False
                        )
                    )

                    category_books = books[
                        category_mask
                    ].copy()

                    category_books = category_books[
                        category_books["title"]
                        .astype(str)
                        .str.lower()
                        != selected_book.lower()
                    ]

                    category_books = (
                        category_books
                        .head(number_of_recommendations)
                    )

                    if not category_books.empty:

                        st.dataframe(
                            category_books[
                                [
                                    "title",
                                    "authors",
                                    "categories",
                                    "average_rating"
                                ]
                            ],
                            use_container_width=True,
                            hide_index=True
                        )

                    else:

                        st.info(
                            "No category-based matches found."
                        )

                else:

                    st.info(
                        "Category information is not available."
                    )

            # =================================================
            # AUTHOR-BASED
            # =================================================

            st.header(
                "✍️ Similar by Author"
            )

            selected_author = ""

            if not selected_row.empty:

                selected_author = str(
                    selected_row.iloc[0].get(
                        "authors",
                        ""
                    )
                ).strip()

            if selected_author:

                author_mask = (
                    books["authors"]
                    .fillna("")
                    .astype(str)
                    .str.lower()
                    .str.contains(
                        selected_author.lower(),
                        regex=False,
                        na=False
                    )
                )

                author_books = books[
                    author_mask
                ].copy()

                author_books = author_books[
                    author_books["title"]
                    .astype(str)
                    .str.lower()
                    != selected_book.lower()
                ]

                author_books = (
                    author_books
                    .head(number_of_recommendations)
                )

                if not author_books.empty:

                    st.dataframe(
                        author_books[
                            [
                                "title",
                                "authors",
                                "categories",
                                "average_rating"
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.info(
                        "No other books by the same author were found."
                    )

            else:

                st.info(
                    "Author information is not available."
                )

            # =================================================
            # RATING / POPULARITY
            # =================================================

            st.header(
                "⭐ Highly Rated Related Books"
            )

            if (
                "ratings_count" in books.columns
                and "average_rating" in books.columns
            ):

                popular_books = books[
                    [
                        "title",
                        "authors",
                        "categories",
                        "average_rating",
                        "ratings_count"
                    ]
                ].copy()

                popular_books["average_rating"] = pd.to_numeric(
                    popular_books["average_rating"],
                    errors="coerce"
                )

                popular_books["ratings_count"] = pd.to_numeric(
                    popular_books["ratings_count"],
                    errors="coerce"
                )

                popular_books = (
                    popular_books[
                        popular_books["title"]
                        .astype(str)
                        .str.lower()
                        != selected_book.lower()
                    ]
                    .sort_values(
                        [
                            "average_rating",
                            "ratings_count"
                        ],
                        ascending=False
                    )
                    .head(
                        number_of_recommendations
                    )
                )

                st.dataframe(
                    popular_books,
                    use_container_width=True,
                    hide_index=True
                )


# ============================================================
# MOVIE ANALYTICS
# ============================================================

elif page == "📊 Movie Analytics":

    st.title(
        "📊 Movie Analytics"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Movies",
            f"{len(movies):,}"
        )

    with col2:

        if "is_indian" in movies.columns:

            indian_count = (
                movies["is_indian"]
                .fillna(False)
                .astype(bool)
                .sum()
            )

        else:
            indian_count = 0

        st.metric(
            "🇮🇳 Indian Movies",
            f"{indian_count:,}"
        )

    with col3:

        if "source" in movies.columns:

            netflix_count = (
                movies["source"]
                .astype(str)
                .str.lower()
                .eq("netflix")
                .sum()
            )

        else:
            netflix_count = 0

        st.metric(
            "Netflix",
            f"{netflix_count:,}"
        )

    with col4:

        if "year" in movies.columns:

            release_years = (
                movies["year"].nunique()
            )

        else:
            release_years = 0

        st.metric(
            "Release Years",
            f"{release_years:,}"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # SOURCE
    # --------------------------------------------------------

    st.header(
        "🎞️ Movies by Source"
    )

    if "source" in movies.columns:

        st.bar_chart(
            movies["source"].value_counts()
        )

    # --------------------------------------------------------
    # INDIAN VS OTHER
    # --------------------------------------------------------

    if "is_indian" in movies.columns:

        st.header(
            "🇮🇳 Indian vs Other Movies"
        )

        indian_counts = pd.Series(
            {
                "Indian Movies": int(
                    movies["is_indian"]
                    .fillna(False)
                    .astype(bool)
                    .sum()
                ),
                "Other Movies": int(
                    (
                        ~movies["is_indian"]
                        .fillna(False)
                        .astype(bool)
                    ).sum()
                )
            }
        )

        st.bar_chart(
            indian_counts
        )

    # --------------------------------------------------------
    # RELEASE YEARS
    # --------------------------------------------------------

    st.header(
        "📅 Movies by Release Year"
    )

    if "year" in movies.columns:

        movie_years = pd.to_numeric(
            movies["year"],
            errors="coerce"
        )

        yearly_movies = (
            movie_years
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
        )

        st.line_chart(
            yearly_movies
        )

    # --------------------------------------------------------
    # GENRES
    # --------------------------------------------------------

    st.header(
        "🎭 Top Genres"
    )

    if "genre" in movies.columns:

        genre_counts = (
            movies["genre"]
            .fillna("")
            .astype(str)
            .str.split(",")
            .explode()
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .head(15)
        )

        st.bar_chart(
            genre_counts
        )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    st.header(
        "🎬 Movie Dataset"
    )

    display_columns = [
        "title",
        "genre",
        "director",
        "year",
        "source",
        "is_indian"
    ]

    display_columns = [
        column
        for column in display_columns
        if column in movies.columns
    ]

    st.dataframe(
        movies[
            display_columns
        ].head(50),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# BOOK ANALYTICS
# ============================================================

elif page == "📖 Book Analytics":

    st.title(
        "📖 Book Analytics"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Books",
            f"{len(books):,}"
        )

    with col2:

        st.metric(
            "Authors",
            f"{books['authors'].nunique():,}"
        )

    with col3:

        st.metric(
            "Categories",
            f"{books['categories'].nunique():,}"
        )

    with col4:

        average_rating = pd.to_numeric(
            books["average_rating"],
            errors="coerce"
        ).mean()

        if pd.isna(average_rating):
            average_rating = 0

        st.metric(
            "Average Rating",
            f"{average_rating:.2f}"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # CATEGORIES
    # --------------------------------------------------------

    st.header(
        "📚 Top Book Categories"
    )

    category_counts = (
        books["categories"]
        .fillna("")
        .astype(str)
        .str.split(",")
        .explode()
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .head(15)
    )

    st.bar_chart(
        category_counts
    )

    # --------------------------------------------------------
    # RATINGS
    # --------------------------------------------------------

    st.header(
        "⭐ Rating Distribution"
    )

    rating_values = pd.to_numeric(
        books["average_rating"],
        errors="coerce"
    ).dropna()

    rating_distribution = (
        rating_values
        .round(1)
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        rating_distribution
    )

    # --------------------------------------------------------
    # PUBLISHED YEAR
    # --------------------------------------------------------

    st.header(
        "📅 Books by Published Year"
    )

    published_year = pd.to_numeric(
        books["published_year"],
        errors="coerce"
    )

    books_by_year = (
        published_year
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
    )

    st.line_chart(
        books_by_year
    )

    # --------------------------------------------------------
    # MOST RATED
    # --------------------------------------------------------

    st.header(
        "🔥 Most Rated Books"
    )

    most_rated = books[
        [
            "title",
            "authors",
            "average_rating",
            "ratings_count"
        ]
    ].copy()

    most_rated["ratings_count"] = pd.to_numeric(
        most_rated["ratings_count"],
        errors="coerce"
    )

    most_rated = (
        most_rated
        .sort_values(
            "ratings_count",
            ascending=False
        )
        .head(20)
    )

    st.dataframe(
        most_rated,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# AI / NLP INSIGHTS
# ============================================================

elif page == "🤖 AI / NLP Insights":

    st.title(
        "🤖 AI / NLP Insights"
    )

    st.header(
        "🧠 Recommendation Approach"
    )

    st.write(
        """
        MovieBook AI uses content-based recommendation.
        Instead of comparing users, the system compares
        the content and textual characteristics of movies
        and books.
        """
    )

    # --------------------------------------------------------
    # MOVIE FEATURES
    # --------------------------------------------------------

    st.header(
        "🎬 Movie Features"
    )

    movie_features = pd.DataFrame(
        {
            "Feature": [
                "Title",
                "Director",
                "Cast",
                "Genre",
                "Overview"
            ],
            "Purpose": [
                "Movie identification",
                "Director similarity",
                "Actor similarity",
                "Genre similarity",
                "Story/content similarity"
            ]
        }
    )

    st.dataframe(
        movie_features,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # BOOK FEATURES
    # --------------------------------------------------------

    st.header(
        "📚 Book Features"
    )

    book_features = pd.DataFrame(
        {
            "Feature": [
                "Title",
                "Subtitle",
                "Authors",
                "Categories",
                "Description"
            ],
            "Purpose": [
                "Book identification",
                "Additional title information",
                "Author similarity",
                "Category similarity",
                "Content similarity"
            ]
        }
    )

    st.dataframe(
        book_features,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    st.header(
        "1️⃣ TF-IDF"
    )

    st.write(
        """
        TF-IDF converts text into numerical vectors.
        Words that are useful for distinguishing documents
        receive greater importance.
        """
    )

    st.latex(
        r"TFIDF(t,d) = TF(t,d) \times IDF(t)"
    )

    # --------------------------------------------------------
    # COSINE SIMILARITY
    # --------------------------------------------------------

    st.header(
        "2️⃣ Cosine Similarity"
    )

    st.write(
        """
        Cosine similarity measures the similarity between
        two TF-IDF vectors.
        """
    )

    st.latex(
        r"Cosine\ Similarity = \frac{A \cdot B}{||A|| ||B||}"
    )

    st.info(
        """
        Example:

        A similarity score of 0.252 is displayed as
        approximately 25.2%.

        This is a similarity score, not a rating or
        probability of liking.
        """
    )

    # --------------------------------------------------------
    # EVALUATION NOTE
    # --------------------------------------------------------

    st.header(
        "📊 Recommendation Evaluation"
    )

    st.write(
        """
        This project uses content-based recommendation rather
        than binary classification. Therefore, classification
        metrics such as accuracy, precision, recall, F1-score,
        and confusion matrix are not used for the recommendation
        engine.

        The recommendation output is based on TF-IDF
        representation and cosine-similarity ranking.
        """
    )

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    st.header(
        "3️⃣ Recommendation Pipeline"
    )

    st.code(
        """
Raw Dataset
     ↓
Data Cleaning
     ↓
Feature Engineering
     ↓
Text Combination
     ↓
TF-IDF Vectorization
     ↓
Cosine Similarity
     ↓
Similarity Ranking
     ↓
Top-N Recommendations
        """,
        language="text"
    )

    # --------------------------------------------------------
    # TECHNOLOGY
    # --------------------------------------------------------

    st.header(
        "🛠️ Technology Stack"
    )

    technology_data = pd.DataFrame(
        {
            "Component": [
                "Programming",
                "Data Processing",
                "Machine Learning",
                "NLP",
                "Recommendation",
                "Dashboard"
            ],
            "Technology": [
                "Python",
                "Pandas",
                "Scikit-learn",
                "TF-IDF",
                "Cosine Similarity",
                "Streamlit"
            ]
        }
    )

    st.dataframe(
        technology_data,
        use_container_width=True,
        hide_index=True
    )

    st.success(
        """
        MovieBook AI demonstrates data preprocessing,
        exploratory data analysis, NLP feature engineering,
        TF-IDF vectorization, cosine similarity,
        content-based recommendation and Streamlit
        dashboard development.
        """
    )