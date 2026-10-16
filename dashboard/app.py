# ============================================================
# MOVIEBOOK AI
# FULL-WIDTH POWER BI-INSPIRED DASHBOARD
# Theme: Dark Teal
# No Sidebar | Top Navigation | Responsive Layout
# ============================================================

from pathlib import Path
import sys
import warnings
import html

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

if not (BASE_DIR / "src").exists() and (BASE_DIR.parent / "src").exists():
    BASE_DIR = BASE_DIR.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DATA_DIR = BASE_DIR / "data" / "processed"

MOVIES_PATH = DATA_DIR / "movies_cleaned.csv"
BOOKS_PATH = DATA_DIR / "books_cleaned.csv"

# ============================================================
# RECOMMENDER IMPORTS
# ============================================================

try:
    from src.movie_recommender import MovieRecommender
    MOVIE_IMPORT_ERROR = None
except Exception as exc:
    MovieRecommender = None
    MOVIE_IMPORT_ERROR = str(exc)

try:
    from src.book_recommender import BookRecommender
    BOOK_IMPORT_ERROR = None
except Exception as exc:
    BookRecommender = None
    BOOK_IMPORT_ERROR = str(exc)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MovieBook AI | Intelligence Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# COLOR PALETTE
# ============================================================

BG = "#071A1C"
PANEL = "#102A2C"
PANEL_HOVER = "#173638"
ACCENT = "#2DD4BF"
ACCENT_DARK = "#14B8A6"
TEXT = "#F1F5F9"
MUTED = "#94A3B8"
BORDER = "#244547"

# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    f"""
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    :root {{
        --bg: {BG};
        --panel: {PANEL};
        --accent: {ACCENT};
        --text: {TEXT};
        --muted: {MUTED};
        --border: {BORDER};
    }}

    html, body, [data-testid="stAppViewContainer"] {{
        background: {BG} !important;
        color: {TEXT} !important;
        font-family: 'Inter', sans-serif !important;
    }}

    [data-testid="stHeader"] {{
        background: {BG} !important;
    }}

    [data-testid="stSidebar"],
    [data-testid="collapsedControl"] {{
        display: none !important;
    }}

    [data-testid="stMain"] {{
        background: {BG} !important;
    }}

    .block-container {{
        max-width: 1800px;
        padding: 1.4rem 2.2rem 2rem 2.2rem;
    }}

    h1, h2, h3, h4 {{
        color: {TEXT} !important;
        font-family: 'Inter', sans-serif !important;
    }}

    p, label, li {{
        color: {MUTED};
    }}

    /* TOP BRAND HEADER */

    .top-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 20px;
        padding: 14px 0 20px 0;
        border-bottom: 1px solid {BORDER};
        margin-bottom: 24px;
        flex-wrap: wrap;
    }}

    .brand {{
        display: flex;
        align-items: center;
        gap: 13px;
    }}

    .brand-icon {{
        display: flex;
        justify-content: center;
        align-items: center;
        width: 48px;
        height: 48px;
        border-radius: 14px;
        background: rgba(45, 212, 191, 0.12);
        border: 1px solid rgba(45, 212, 191, 0.3);
        font-size: 24px;
    }}

    .brand-name {{
        color: {TEXT};
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -0.7px;
    }}

    .brand-name span {{
        color: {ACCENT};
    }}

    .brand-subtitle {{
        color: {MUTED};
        font-size: 11px;
        margin-top: 4px;
    }}

    .header-status {{
        color: {ACCENT};
        background: rgba(45, 212, 191, 0.10);
        border: 1px solid rgba(45, 212, 191, 0.25);
        padding: 9px 14px;
        border-radius: 25px;
        font-size: 11px;
        font-weight: 700;
    }}

    /* PAGE INTRODUCTION */

    .page-heading {{
        color: {TEXT};
        font-size: 30px;
        font-weight: 800;
        letter-spacing: -0.8px;
        margin: 0 0 6px 0;
    }}

    .page-description {{
        color: {MUTED};
        font-size: 13px;
        line-height: 1.7;
        margin-bottom: 22px;
    }}

    .hero {{
        background:
            radial-gradient(
                circle at 90% 10%,
                rgba(45,212,191,0.12),
                transparent 35%
            ),
            linear-gradient(120deg, #102A2C, #0B2426);
        border: 1px solid {BORDER};
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 22px;
    }}

    .hero-title {{
        font-size: clamp(28px, 4vw, 42px);
        line-height: 1.2;
        color: {TEXT};
        font-weight: 800;
        letter-spacing: -1px;
    }}

    .hero-title span {{
        color: {ACCENT};
    }}

    .hero-description {{
        max-width: 800px;
        color: {MUTED};
        font-size: 14px;
        line-height: 1.8;
        margin-top: 12px;
    }}

    /* KPI CARDS */

    .metric-card {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 15px;
        padding: 19px;
        min-height: 120px;
        margin-bottom: 12px;
    }}

    .metric-label {{
        color: {MUTED};
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    .metric-value {{
        color: {ACCENT};
        font-size: 29px;
        font-weight: 800;
        margin-top: 13px;
    }}

    .metric-note {{
        color: {MUTED};
        font-size: 11px;
        margin-top: 5px;
    }}

    /* SECTIONS */

    .section-heading {{
        color: {TEXT};
        font-size: 19px;
        font-weight: 750;
        margin-top: 24px;
        margin-bottom: 5px;
    }}

    .section-description {{
        color: {MUTED};
        font-size: 12px;
        margin-bottom: 17px;
    }}

    .content-card {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 14px;
    }}

    .content-card-title {{
        color: {TEXT};
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 8px;
    }}

    .content-card-description {{
        color: {MUTED};
        font-size: 12px;
        line-height: 1.8;
    }}

    .badge {{
        display: inline-block;
        color: {ACCENT};
        background: rgba(45, 212, 191, 0.1);
        border: 1px solid rgba(45, 212, 191, 0.22);
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 10px;
        font-weight: 700;
        margin: 3px;
    }}

    .insight-box {{
        background: #0C2527;
        border-left: 4px solid {ACCENT};
        border-radius: 8px;
        padding: 16px;
        color: {MUTED};
        font-size: 13px;
        line-height: 1.8;
        margin-bottom: 15px;
    }}

    /* TOP NAVIGATION BUTTONS */

    div.stButton > button {{
        background: {PANEL};
        color: {TEXT} !important;
        border: 1px solid {BORDER};
        border-radius: 10px;
        min-height: 43px;
        font-size: 12px;
        font-weight: 700;
        transition: all 0.2s ease;
    }}

    div.stButton > button:hover {{
        background: {PANEL_HOVER};
        color: {ACCENT} !important;
        border-color: {ACCENT};
    }}

    div.stButton > button:focus:not(:active) {{
        border-color: {ACCENT};
        color: {ACCENT} !important;
    }}

    /* INPUTS */

    div[data-baseweb="select"] > div {{
        background: {PANEL} !important;
        border-color: {BORDER} !important;
        color: {TEXT} !important;
    }}

    div[data-baseweb="select"] * {{
        color: {TEXT} !important;
    }}

    .stTextInput input,
    .stNumberInput input {{
        background: {PANEL} !important;
        color: {TEXT} !important;
        border-color: {BORDER} !important;
    }}

    /* DATAFRAMES */

    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 12px;
        overflow: hidden;
    }}

    /* METRICS */

    div[data-testid="stMetric"] {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 13px;
        padding: 16px;
    }}

    div[data-testid="stMetricLabel"] {{
        color: {MUTED};
    }}

    div[data-testid="stMetricValue"] {{
        color: {ACCENT};
    }}

    /* ALERTS */

    div[data-testid="stAlert"] {{
        border-radius: 12px;
    }}

    hr {{
        border-color: {BORDER};
    }}

    /* FOOTER */

    .dashboard-footer {{
        border-top: 1px solid {BORDER};
        margin-top: 35px;
        padding: 20px 0 5px 0;
        text-align: center;
        color: {MUTED};
        font-size: 11px;
        line-height: 2;
    }}

    @media (max-width: 768px) {{
        .block-container {{
            padding-left: 1rem;
            padding-right: 1rem;
        }}

        .hero {{
            padding: 22px;
        }}

        .metric-value {{
            font-size: 23px;
        }}

        .brand-name {{
            font-size: 20px;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_dataset(path_string):
    path = Path(path_string)

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"Could not load {path}: {exc}")
        return pd.DataFrame()


movies_df = load_dataset(str(MOVIES_PATH))
books_df = load_dataset(str(BOOKS_PATH))


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_movie_model():
    if MovieRecommender is None:
        return None

    try:
        return MovieRecommender()
    except Exception as exc:
        print(f"Movie model initialization failed: {exc}")
        return None


@st.cache_resource(show_spinner=False)
def load_book_model():
    if BookRecommender is None:
        return None

    try:
        return BookRecommender()
    except Exception as exc:
        print(f"Book model initialization failed: {exc}")
        return None


movie_model = load_movie_model()
book_model = load_book_model()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates):
    if df.empty:
        return None

    column_map = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in column_map:
            return column_map[candidate.lower()]

    return None


def get_title_column(df, kind):
    if kind == "movie":
        candidates = [
            "title",
            "movie_title",
            "original_title",
            "name",
        ]
    else:
        candidates = [
            "title",
            "book_title",
            "name",
        ]

    return find_column(df, candidates)


def get_creator_column(df, kind):
    if kind == "movie":
        candidates = [
            "director",
            "directors",
            "director_name",
        ]
    else:
        candidates = [
            "authors",
            "author",
            "book_author",
        ]

    return find_column(df, candidates)


def get_text(value, fallback="Not available"):
    if value is None:
        return fallback

    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass

    value = str(value).strip()

    if not value or value.lower() in {"nan", "none", "null"}:
        return fallback

    return value


def safe_html(value):
    return html.escape(str(value))


def section_heading(title, description=""):
    st.markdown(
        f'<div class="section-heading">{safe_html(title)}</div>',
        unsafe_allow_html=True,
    )

    if description:
        st.markdown(
            f'<div class="section-description">'
            f'{safe_html(description)}</div>',
            unsafe_allow_html=True,
        )


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{safe_html(label)}</div>
            <div class="metric-value">{safe_html(value)}</div>
            <div class="metric-note">{safe_html(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_chart(fig, height=350):
    fig.update_layout(
        height=height,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(
            family="Inter, sans-serif",
            color=TEXT,
            size=12,
        ),
        margin=dict(l=20, r=20, t=55, b=20),
        title_font=dict(size=15, color=TEXT),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=MUTED),
        ),
    )

    fig.update_xaxes(
        gridcolor=BORDER,
        zerolinecolor=BORDER,
        tickfont=dict(color=MUTED),
        title_font=dict(color=MUTED),
    )

    fig.update_yaxes(
        gridcolor=BORDER,
        zerolinecolor=BORDER,
        tickfont=dict(color=MUTED),
        title_font=dict(color=MUTED),
    )

    return fig


def show_missing_dataset(name, path):
    st.error(f"The {name} dataset could not be loaded.")

    st.code(str(path))

    st.info(
        "Check the CSV filename and project folder structure. "
        "The dashboard expects the processed CSV inside "
        "data/processed/."
    )


def run_recommendation(model, title, count):
    if model is None:
        raise RuntimeError(
            "The recommendation model could not be initialized. "
            "Check the model import and constructor."
        )

    if not hasattr(model, "recommend"):
        raise AttributeError(
            "The recommender class does not provide a recommend() method."
        )

    try:
        result = model.recommend(title, n=count)
    except TypeError:
        result = model.recommend(title)

    if result is None:
        return pd.DataFrame()

    if isinstance(result, pd.DataFrame):
        return result.copy()

    if isinstance(result, pd.Series):
        return result.to_frame().T

    return pd.DataFrame(result)


def show_recommendations(results, kind):
    if results.empty:
        st.info(
            "No recommendations were returned. Try another title."
        )
        return

    title_col = get_title_column(results, kind)

    if title_col is None:
        st.warning(
            "The recommendation results have no recognized title "
            "column. Showing the raw results instead."
        )

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True,
        )
        return

    creator_col = get_creator_column(results, kind)

    category_col = find_column(
        results,
        ["genre", "genres", "category", "categories", "subject"],
    )

    description_col = find_column(
        results,
        ["overview", "description", "summary", "plot"],
    )

    score_col = find_column(
        results,
        ["similarity", "similarity_score", "score", "confidence"],
    )

    for start in range(0, len(results), 3):
        columns = st.columns(3)

        chunk = results.iloc[start:start + 3]

        for index, (_, row) in enumerate(chunk.iterrows()):
            with columns[index]:
                title = get_text(row.get(title_col))

                creator = (
                    get_text(row.get(creator_col))
                    if creator_col
                    else "Not available"
                )

                category = (
                    get_text(row.get(category_col))
                    if category_col
                    else "Not specified"
                )

                description = (
                    get_text(row.get(description_col))
                    if description_col
                    else "Explore this recommendation."
                )

                description = (
                    description[:200] + "..."
                    if len(description) > 200
                    else description
                )

                st.markdown(
                    f"""
                    <div class="content-card">
                        <div class="content-card-title">
                            {safe_html(title)}
                        </div>
                        <div class="content-card-description">
                            {"Director" if kind == "movie" else "Author"}:
                            {safe_html(creator)}
                        </div>
                        <div style="margin-top:10px;">
                            <span class="badge">
                                {safe_html(category)}
                            </span>
                        </div>
                        <div class="content-card-description"
                             style="margin-top:12px;">
                            {safe_html(description)}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if score_col:
                    score = pd.to_numeric(
                        pd.Series([row.get(score_col)]),
                        errors="coerce",
                    ).iloc[0]

                    if pd.notna(score):
                        st.caption(
                            f"Similarity score: {score:.3f}"
                        )


# ============================================================
# TOP BRAND HEADER
# ============================================================

st.markdown(
    f"""
    <div class="top-header">
        <div class="brand">
            <div class="brand-icon">🎬</div>
            <div>
                <div class="brand-name">
                    MovieBook <span>AI</span>
                </div>
                <div class="brand-subtitle">
                    INTELLIGENT CONTENT DISCOVERY PLATFORM
                </div>
            </div>
        </div>
        <div class="header-status">
            ● RECOMMENDATION SYSTEM
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TOP NAVIGATION — NO SIDEBAR
# ============================================================

PAGES = [
    "Overview",
    "Movies",
    "Books",
    "Movie Analytics",
    "Book Analytics",
    "AI / NLP",
    "Dataset Explorer",
]

if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Overview"

navigation_columns = st.columns(len(PAGES))

for index, page_name in enumerate(PAGES):
    with navigation_columns[index]:
        is_active = st.session_state["active_page"] == page_name

        button_label = (
            f"● {page_name}"
            if is_active
            else page_name
        )

        if st.button(
            button_label,
            key=f"nav_{index}",
            use_container_width=True,
        ):
            st.session_state["active_page"] = page_name
            st.rerun()

page = st.session_state["active_page"]

st.markdown("---")


# ============================================================
# OVERVIEW PAGE
# ============================================================

if page == "Overview":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                Discover something <span>new.</span>
            </div>
            <div class="hero-description">
                One platform for discovering movies and books.
                Explore content-based recommendations, analyze your
                datasets, and learn how machine learning and NLP
                support personalized content discovery.
            </div>
            <div style="margin-top:18px;">
                <span class="badge">MOVIES</span>
                <span class="badge">BOOKS</span>
                <span class="badge">MACHINE LEARNING</span>
                <span class="badge">NLP</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    movie_title_col = get_title_column(movies_df, "movie")
    book_title_col = get_title_column(books_df, "book")

    unique_movies = (
        movies_df[movie_title_col].nunique()
        if movie_title_col
        else 0
    )

    unique_books = (
        books_df[book_title_col].nunique()
        if book_title_col
        else 0
    )

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        metric_card("Movie Records", f"{len(movies_df):,}")

    with k2:
        metric_card("Book Records", f"{len(books_df):,}")

    with k3:
        metric_card("Unique Movies", f"{unique_movies:,}")

    with k4:
        metric_card("Unique Books", f"{unique_books:,}")

    section_heading(
        "Platform Overview",
        "Your content discovery platform at a glance.",
    )

    left, right = st.columns(2)

    with left:
        st.markdown(
            """
            <div class="content-card">
                <div class="content-card-title">
                    🎬 Movie Recommendation Engine
                </div>
                <div class="content-card-description">
                    Explore movies similar to a selected title using
                    your existing recommendation model.
                </div>
                <br>
                <span class="badge">CONTENT-BASED</span>
                <span class="badge">MOVIE DISCOVERY</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Explore Movies →",
            key="overview_movies",
            use_container_width=True,
        ):
            st.session_state["active_page"] = "Movies"
            st.rerun()

    with right:
        st.markdown(
            """
            <div class="content-card">
                <div class="content-card-title">
                    📚 Book Recommendation Engine
                </div>
                <div class="content-card-description">
                    Discover books related to a selected title and
                    explore the available book collection.
                </div>
                <br>
                <span class="badge">BOOK DISCOVERY</span>
                <span class="badge">NLP CONCEPTS</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Explore Books →",
            key="overview_books",
            use_container_width=True,
        ):
            st.session_state["active_page"] = "Books"
            st.rerun()

    section_heading(
        "Dataset Distribution",
        "Number of records available in each collection.",
    )

    distribution = pd.DataFrame(
        {
            "Content": ["Movies", "Books"],
            "Records": [len(movies_df), len(books_df)],
        }
    )

    fig = px.bar(
        distribution,
        x="Content",
        y="Records",
        color="Content",
        color_discrete_sequence=[ACCENT, "#5EEAD4"],
        title="Movie and Book Dataset Size",
        text="Records",
    )

    fig.update_traces(textposition="outside")

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True,
    )

    if movies_df.empty:
        show_missing_dataset("movie", MOVIES_PATH)

    if books_df.empty:
        show_missing_dataset("book", BOOKS_PATH)


# ============================================================
# MOVIE RECOMMENDATIONS PAGE
# ============================================================

elif page == "Movies":

    st.markdown(
        """
        <div class="page-heading">Movie Discovery</div>
        <div class="page-description">
            Select a movie and discover similar titles.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if movies_df.empty:
        show_missing_dataset("movie", MOVIES_PATH)

    else:
        title_col = get_title_column(movies_df, "movie")

        if title_col is None:
            st.error("A movie title column was not found.")

        else:
            titles = sorted(
                movies_df[title_col]
                .dropna()
                .astype(str)
                .str.strip()
                .loc[lambda x: x.ne("")]
                .unique()
                .tolist()
            )

            c1, c2 = st.columns([3, 1])

            with c1:
                selected_title = st.selectbox(
                    "Select a movie",
                    titles,
                    key="movie_selected_title",
                )

            with c2:
                count = st.selectbox(
                    "Recommendations",
                    [3, 6, 9, 12],
                    index=1,
                    key="movie_count",
                )

            if st.button(
                "✨ Find Similar Movies",
                key="movie_recommend_button",
                use_container_width=True,
            ):
                try:
                    with st.spinner("Finding similar movies..."):
                        results = run_recommendation(
                            movie_model,
                            selected_title,
                            count,
                        )

                    st.session_state["movie_results"] = results
                    st.session_state["movie_results_for"] = selected_title

                except Exception as exc:
                    st.error("Movie recommendation failed.")

                    with st.expander("Technical error"):
                        st.code(str(exc))

            if (
                "movie_results" in st.session_state
                and st.session_state.get("movie_results_for")
                == selected_title
            ):
                section_heading(
                    "Recommended Movies",
                    f"Recommendations related to {selected_title}.",
                )

                show_recommendations(
                    st.session_state["movie_results"],
                    "movie",
                )

            with st.expander("View selected movie details"):
                details = movies_df[
                    movies_df[title_col].astype(str) == selected_title
                ]

                if not details.empty:
                    st.dataframe(
                        details.head(1),
                        use_container_width=True,
                        hide_index=True,
                    )


# ============================================================
# BOOK RECOMMENDATIONS PAGE
# ============================================================

elif page == "Books":

    st.markdown(
        """
        <div class="page-heading">Book Discovery</div>
        <div class="page-description">
            Explore the book collection and find similar titles.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if books_df.empty:
        show_missing_dataset("book", BOOKS_PATH)

    else:
        title_col = get_title_column(books_df, "book")

        if title_col is None:
            st.error("A book title column was not found.")

        else:
            titles = sorted(
                books_df[title_col]
                .dropna()
                .astype(str)
                .str.strip()
                .loc[lambda x: x.ne("")]
                .unique()
                .tolist()
            )

            c1, c2 = st.columns([3, 1])

            with c1:
                selected_title = st.selectbox(
                    "Select a book",
                    titles,
                    key="book_selected_title",
                )

            with c2:
                count = st.selectbox(
                    "Recommendations",
                    [3, 6, 9, 12],
                    index=1,
                    key="book_count",
                )

            if st.button(
                "✨ Find Similar Books",
                key="book_recommend_button",
                use_container_width=True,
            ):
                try:
                    with st.spinner("Finding similar books..."):
                        results = run_recommendation(
                            book_model,
                            selected_title,
                            count,
                        )

                    st.session_state["book_results"] = results
                    st.session_state["book_results_for"] = selected_title

                except Exception as exc:
                    st.error("Book recommendation failed.")

                    with st.expander("Technical error"):
                        st.code(str(exc))

            if (
                "book_results" in st.session_state
                and st.session_state.get("book_results_for")
                == selected_title
            ):
                section_heading(
                    "Recommended Books",
                    f"Recommendations related to {selected_title}.",
                )

                show_recommendations(
                    st.session_state["book_results"],
                    "book",
                )

            with st.expander("View selected book details"):
                details = books_df[
                    books_df[title_col].astype(str) == selected_title
                ]

                if not details.empty:
                    st.dataframe(
                        details.head(1),
                        use_container_width=True,
                        hide_index=True,
                    )


# ============================================================
# MOVIE ANALYTICS PAGE
# ============================================================

elif page == "Movie Analytics":

    st.markdown(
        """
        <div class="page-heading">Movie Analytics</div>
        <div class="page-description">
            Analyze movie records, genres, years, and ratings where available.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if movies_df.empty:
        show_missing_dataset("movie", MOVIES_PATH)

    else:
        genre_col = find_column(
            movies_df,
            ["genre", "genres", "listed_in", "category"],
        )

        year_col = find_column(
            movies_df,
            ["release_year", "year", "release_date"],
        )

        rating_col = find_column(
            movies_df,
            ["rating", "vote_average", "imdb_rating", "score"],
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card("Total Records", f"{len(movies_df):,}")

        with c2:
            metric_card("Dataset Columns", str(len(movies_df.columns)))

        with c3:
            metric_card(
                "Missing Cells",
                f"{int(movies_df.isna().sum().sum()):,}",
            )

        with c4:
            metric_card(
                "Duplicate Rows",
                f"{int(movies_df.duplicated().sum()):,}",
            )

        left, right = st.columns(2)

        with left:
            if genre_col:
                genres = (
                    movies_df[genre_col]
                    .dropna()
                    .astype(str)
                    .str.split(r",|;|\|")
                    .explode()
                    .str.strip()
                )

                genres = genres[genres.ne("")]

                genre_counts = genres.value_counts().head(12)

                if not genre_counts.empty:
                    chart_df = genre_counts.rename_axis(
                        "Genre"
                    ).reset_index(name="Count")

                    fig = px.bar(
                        chart_df,
                        x="Count",
                        y="Genre",
                        orientation="h",
                        title="Most Common Genres",
                        color="Count",
                        color_continuous_scale=[
                            "#164E4A",
                            ACCENT,
                        ],
                    )

                    fig.update_layout(
                        yaxis={"categoryorder": "total ascending"}
                    )

                    st.plotly_chart(
                        style_chart(fig),
                        use_container_width=True,
                    )
            else:
                st.info("No genre column was found.")

        with right:
            if year_col:
                if year_col.lower() == "release_date":
                    years = pd.to_datetime(
                        movies_df[year_col],
                        errors="coerce",
                    ).dt.year
                else:
                    years = pd.to_numeric(
                        movies_df[year_col],
                        errors="coerce",
                    )

                year_counts = (
                    years.dropna()
                    .astype(int)
                    .value_counts()
                    .sort_index()
                    .rename_axis("Year")
                    .reset_index(name="Records")
                )

                if not year_counts.empty:
                    fig = px.area(
                        year_counts,
                        x="Year",
                        y="Records",
                        title="Movies by Year",
                    )

                    fig.update_traces(
                        line_color=ACCENT,
                        fillcolor="rgba(45,212,191,0.18)",
                    )

                    st.plotly_chart(
                        style_chart(fig),
                        use_container_width=True,
                    )
            else:
                st.info("No release-year column was found.")

        if rating_col:
            ratings = pd.to_numeric(
                movies_df[rating_col],
                errors="coerce",
            ).dropna()

            if not ratings.empty:
                fig = px.histogram(
                    x=ratings,
                    nbins=20,
                    title="Movie Rating Distribution",
                )

                fig.update_traces(marker_color=ACCENT)

                st.plotly_chart(
                    style_chart(fig),
                    use_container_width=True,
                )

        section_heading("Movie Dataset Preview")

        st.dataframe(
            movies_df.head(100),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# BOOK ANALYTICS PAGE
# ============================================================

elif page == "Book Analytics":

    st.markdown(
        """
        <div class="page-heading">Book Analytics</div>
        <div class="page-description">
            Explore authors, publication years, ratings, and book records.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if books_df.empty:
        show_missing_dataset("book", BOOKS_PATH)

    else:
        author_col = get_creator_column(books_df, "book")

        year_col = find_column(
            books_df,
            [
                "publication_year",
                "year",
                "published_year",
                "year_of_publication",
            ],
        )

        rating_col = find_column(
            books_df,
            ["average_rating", "rating", "book_rating", "score"],
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card("Total Records", f"{len(books_df):,}")

        with c2:
            metric_card("Dataset Columns", str(len(books_df.columns)))

        with c3:
            metric_card(
                "Missing Cells",
                f"{int(books_df.isna().sum().sum()):,}",
            )

        with c4:
            metric_card(
                "Duplicate Rows",
                f"{int(books_df.duplicated().sum()):,}",
            )

        left, right = st.columns(2)

        with left:
            if author_col:
                authors = (
                    books_df[author_col]
                    .dropna()
                    .astype(str)
                    .str.split(r",|;|\|")
                    .explode()
                    .str.strip()
                )

                authors = authors[authors.ne("")]

                author_counts = authors.value_counts().head(12)

                if not author_counts.empty:
                    chart_df = author_counts.rename_axis(
                        "Author"
                    ).reset_index(name="Records")

                    fig = px.bar(
                        chart_df,
                        x="Records",
                        y="Author",
                        orientation="h",
                        title="Authors with Most Records",
                        color="Records",
                        color_continuous_scale=[
                            "#164E4A",
                            ACCENT,
                        ],
                    )

                    fig.update_layout(
                        yaxis={"categoryorder": "total ascending"}
                    )

                    st.plotly_chart(
                        style_chart(fig),
                        use_container_width=True,
                    )
            else:
                st.info("No author column was found.")

        with right:
            if year_col:
                years = pd.to_numeric(
                    books_df[year_col],
                    errors="coerce",
                )

                year_counts = (
                    years.dropna()
                    .astype(int)
                    .value_counts()
                    .sort_index()
                    .rename_axis("Year")
                    .reset_index(name="Records")
                )

                if not year_counts.empty:
                    fig = px.line(
                        year_counts,
                        x="Year",
                        y="Records",
                        markers=True,
                        title="Books by Publication Year",
                    )

                    fig.update_traces(
                        line_color=ACCENT,
                        marker_color=ACCENT,
                    )

                    st.plotly_chart(
                        style_chart(fig),
                        use_container_width=True,
                    )
            else:
                st.info("No publication-year column was found.")

        if rating_col:
            ratings = pd.to_numeric(
                books_df[rating_col],
                errors="coerce",
            ).dropna()

            if not ratings.empty:
                fig = px.histogram(
                    x=ratings,
                    nbins=20,
                    title="Book Rating Distribution",
                )

                fig.update_traces(marker_color=ACCENT)

                st.plotly_chart(
                    style_chart(fig),
                    use_container_width=True,
                )

        section_heading("Book Dataset Preview")

        st.dataframe(
            books_df.head(100),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# AI / NLP PAGE
# ============================================================

elif page == "AI / NLP":

    st.markdown(
        """
        <div class="page-heading">AI & NLP Insights</div>
        <div class="page-description">
            Understand the techniques commonly used in content-based
            recommendation systems.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)

    with left:
        st.markdown(
            """
            <div class="content-card">
                <div class="content-card-title">
                    01. Text Processing
                </div>
                <div class="content-card-description">
                    Text information can be cleaned and standardized
                    before being converted into numerical features.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="content-card">
                <div class="content-card-title">
                    02. TF-IDF
                </div>
                <div class="content-card-description">
                    TF-IDF assigns weights to terms according to their
                    frequency in a document and their distribution
                    across the collection.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="content-card">
                <div class="content-card-title">
                    03. Cosine Similarity
                </div>
                <div class="content-card-description">
                    Cosine similarity compares numerical vectors to
                    estimate how similar their directions are.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="content-card">
                <div class="content-card-title">
                    04. Recommendation Ranking
                </div>
                <div class="content-card-description">
                    Candidate items can be ranked using their similarity
                    scores, and the highest-ranked eligible items can
                    be returned to the user.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_heading("TF-IDF Formula")

    st.latex(
        r"\operatorname{TFIDF}(t,d)"
        r"=\operatorname{TF}(t,d)"
        r"\times \operatorname{IDF}(t)"
    )

    section_heading("Cosine Similarity Formula")

    st.latex(
        r"\operatorname{similarity}(A,B)"
        r"=\frac{A\cdot B}{\|A\|\|B\|}"
    )

    st.markdown(
        """
        <div class="insight-box">
            <strong>Important:</strong>
            These are common recommendation-system techniques.
            The exact method used by MovieBook AI depends on the
            implementation in your existing recommender files.
        </div>
        """,
        unsafe_allow_html=True,
    )

    comparison = pd.DataFrame(
        {
            "Method": [
                "Content-Based Filtering",
                "Collaborative Filtering",
                "Hybrid Recommendation",
            ],
            "Input": [
                "Item content and metadata",
                "User-item interactions",
                "Content and interactions",
            ],
            "Advantage": [
                "Can work without extensive user history",
                "Can learn preference patterns",
                "Combines multiple information sources",
            ],
        }
    )

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# DATASET EXPLORER
# ============================================================

elif page == "Dataset Explorer":

    st.markdown(
        """
        <div class="page-heading">Dataset Explorer</div>
        <div class="page-description">
            Search, inspect, and understand your processed datasets.
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_type = st.radio(
        "Select dataset",
        ["Movies", "Books"],
        horizontal=True,
    )

    selected_df = (
        movies_df
        if selected_type == "Movies"
        else books_df
    )

    if selected_df.empty:
        show_missing_dataset(
            selected_type.lower(),
            MOVIES_PATH if selected_type == "Movies" else BOOKS_PATH,
        )

    else:
        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card("Rows", f"{len(selected_df):,}")

        with c2:
            metric_card("Columns", f"{len(selected_df.columns):,}")

        with c3:
            metric_card(
                "Missing Values",
                f"{int(selected_df.isna().sum().sum()):,}",
            )

        search_column = st.selectbox(
            "Search in",
            ["All columns"] + list(selected_df.columns),
        )

        search_text = st.text_input(
            "Search dataset",
            placeholder="Type a title, author, genre, or keyword...",
        )

        filtered_df = selected_df.copy()

        if search_text.strip():
            if search_column == "All columns":
                mask = filtered_df.astype(str).apply(
                    lambda col: col.str.contains(
                        search_text.strip(),
                        case=False,
                        na=False,
                        regex=False,
                    )
                ).any(axis=1)

            else:
                mask = filtered_df[search_column].astype(str).str.contains(
                    search_text.strip(),
                    case=False,
                    na=False,
                    regex=False,
                )

            filtered_df = filtered_df[mask]

        st.caption(
            f"Found {len(filtered_df):,} matching records."
        )

        st.dataframe(
            filtered_df.head(1000),
            use_container_width=True,
            hide_index=True,
        )

        section_heading("Missing Values by Column")

        missing_df = pd.DataFrame(
            {
                "Column": selected_df.columns,
                "Missing Values": [
                    int(selected_df[col].isna().sum())
                    for col in selected_df.columns
                ],
            }
        )

        missing_df["Missing Percentage"] = (
            missing_df["Missing Values"]
            / max(len(selected_df), 1)
            * 100
        ).round(2)

        missing_df = missing_df.sort_values(
            "Missing Values",
            ascending=False,
        )

        missing_nonzero = missing_df[
            missing_df["Missing Values"] > 0
        ].head(20)

        if missing_nonzero.empty:
            st.success("No missing values were found.")
        else:
            fig = px.bar(
                missing_nonzero,
                x="Missing Values",
                y="Column",
                orientation="h",
                title="Columns with Missing Values",
                color="Missing Values",
                color_continuous_scale=[
                    "#164E4A",
                    ACCENT,
                ],
            )

            fig.update_layout(
                yaxis={"categoryorder": "total ascending"}
            )

            st.plotly_chart(
                style_chart(fig),
                use_container_width=True,
            )

        with st.expander("View column data types"):
            dtype_df = pd.DataFrame(
                {
                    "Column": selected_df.columns,
                    "Data Type": [
                        str(dtype)
                        for dtype in selected_df.dtypes
                    ],
                }
            )

            st.dataframe(
                dtype_df,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f"""
    <div class="dashboard-footer">
        <div style="
            color:{ACCENT};
            font-size:15px;
            font-weight:800;">
            MovieBook AI
        </div>
        Movies • Books • Content-Based Recommendations • Data Analytics
        <br>
        Built with Python, Streamlit, Pandas and Plotly
    </div>
    """,
    unsafe_allow_html=True,
)