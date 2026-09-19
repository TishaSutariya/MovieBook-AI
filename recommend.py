import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CLEAN_FILE = BASE_DIR / "data" / "processed" / "netflix_cleaned.csv"
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"


def tmdb_search_movie(title):
    if not TMDB_API_KEY:
        return None
    try:
        url = f"{TMDB_BASE}/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": title, "include_adult": "false"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        results = r.json().get("results", [])
        return results[0] if results else None
    except Exception:
        return None


def get_movie_poster(title):
    result = tmdb_search_movie(title)
    if not result:
        return None
    poster_path = result.get("poster_path")
    return f"{TMDB_IMG}{poster_path}" if poster_path else None


def get_recommendations(title, top_n=5):
    df = pd.read_csv(CLEAN_FILE)

    if "combined" not in df.columns or "title" not in df.columns:
        return []

    df = df.dropna(subset=["title", "combined"]).reset_index(drop=True)

    if title not in df["title"].values:
        return []

    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(df["combined"])
    similarity = cosine_similarity(matrix, matrix)

    idx = df[df["title"] == title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

    recommendations = []
    for movie_index, _ in scores:
        movie_title = df.iloc[movie_index]["title"]
        recommendations.append(
            {
                "title": movie_title,
                "poster": get_movie_poster(movie_title),
            }
        )

    return recommendations