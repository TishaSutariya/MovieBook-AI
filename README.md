# Netflix Movie Recommendation Dashboard

A beginner-friendly Streamlit project that recommends movies using text similarity and displays posters using the TMDB API.

## Features
- Searchable movie dropdown.
- Top 5 similar movie recommendations.
- TMDB poster fetching.
- Separate tabs for recommendations and trending titles.
- Dark, app-like dashboard layout.
- Private API key handling with `.env`.

## Project Structure
- `dashboard/app.py` — Streamlit dashboard.
- `recommend.py` — recommendation logic and TMDB poster lookup.
- `preprocess.py` — data cleaning script.
- `eda.py` — exploratory data analysis.
- `data/raw/` — raw dataset.
- `data/processed/` — cleaned dataset.

## Requirements
```bash
pip install pandas scikit-learn streamlit requests python-dotenv
```

## TMDB API Key Setup
Create a `.env` file in the project root:

```env
TMDB_API_KEY=your_tmdb_key_here
```

Add this to `.gitignore`:

```gitignore
.env
__pycache__/
venv/
```

Never hardcode your TMDB key inside Python files if you plan to push the project to GitHub.

## How to Run
1. Clean the data:
```bash
python preprocess.py
```

2. Start the dashboard:
```bash
streamlit run dashboard/app.py
```

## Notes
- Posters are fetched from TMDB using the movie title.
- If a poster is not found, the app still works and shows a fallback message.
- The trending section is curated for a clean demo-style dashboard.