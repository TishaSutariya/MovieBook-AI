# 🎬📚 MovieBook AI – Movie & Book Recommendation System

MovieBook AI is a **content-based recommendation system** for movies and books. It uses **Natural Language Processing (NLP), TF-IDF vectorization, and cosine similarity** to recommend items with similar content and metadata.

The project also includes an interactive **Streamlit dashboard** for recommendations, dataset analytics, and NLP-based insights.

---

## 🚀 Features

### 🎬 Movie Recommendation

* Content-based movie recommendation
* Uses movie metadata such as title, genre, cast, director, country, and overview
* TF-IDF vectorization for text representation
* Cosine similarity for finding similar movies
* Search and recommendation functionality
* Indian/Bollywood movie recommendation section
* Movie similarity scores

### 📚 Book Recommendation

* Content-based book recommendation
* Uses book metadata and descriptions
* TF-IDF-based text representation
* Cosine similarity for recommendations
* Book search and recommendation functionality
* Category and author-based exploration

### 📊 Analytics

* Movie dataset statistics
* Book dataset statistics
* Indian movie analysis
* Genre and year-based analysis
* Interactive charts and visualizations

### 🤖 AI / NLP Insights

* Explanation of TF-IDF
* Explanation of cosine similarity
* Text preprocessing concepts
* Recommendation workflow
* Similarity-based recommendation insights
* Technology stack overview

---

## 🧠 How the Recommendation System Works

The project follows a content-based recommendation approach.

```text
Raw Datasets
     ↓
Data Cleaning & Preprocessing
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
```

### 1. Data Preprocessing

The raw movie and book datasets are cleaned and transformed into structured datasets.

For movies, relevant fields include:

* Title
* Genre
* Director
* Cast
* Country
* Overview
* Year

For books, relevant information includes:

* Title
* Authors
* Categories
* Description
* Published year
* Ratings information

---

### 2. Feature Engineering

Relevant text fields are combined to create a representation that captures the item's available content and metadata.

For example:

```text
Title + Genre + Director + Cast + Overview
```

This combined text is then used for NLP processing.

---

### 3. TF-IDF Vectorization

TF-IDF converts textual information into numerical vectors.

It gives higher importance to words that are useful for distinguishing one item from another.

The project uses:

```text
TF-IDF Vectorizer
```

with unigram and bigram features.

---

### 4. Cosine Similarity

Cosine similarity measures how similar two TF-IDF vectors are.

```text
Cosine Similarity =
(A · B) / (||A|| × ||B||)
```

A higher similarity value indicates that the two items have more similar textual representations.

---

### 5. Recommendation

When a user selects a movie or book:

```text
Selected Item
      ↓
TF-IDF Representation
      ↓
Similarity Calculation
      ↓
Similarity Ranking
      ↓
Top Similar Items
```

The system returns the most similar movies or books.

---

## 📊 Recommendation Evaluation

This project uses **content-based recommendation**, rather than a classification model.

Therefore, classification metrics such as:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

are not directly applicable to the current recommendation pipeline.

The system instead uses **cosine similarity and Top-N ranking** to generate recommendations.

For a future version, recommendation-specific evaluation such as **Precision@K, Recall@K, Hit Rate@K, or MAP** could be implemented if suitable ground-truth user preference data is available.

---

## 🗂️ Project Structure

```text
MovieBook-AI/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   ├── book1-100k.csv
│   │   ├── data.csv
│   │   ├── IMDB-Movie-Dataset(2023-1951).csv
│   │   └── movies_on_netflix.csv
│   │
│   └── processed/
│       ├── books_cleaned.csv
│       ├── movies_cleaned.csv
│       └── netflix_cleaned.csv
│
├── src/
│   ├── book_recommender.py
│   ├── movie_recommender.py
│   ├── eda.py
│   ├── main.py
│   └── preprocess.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies Used

| Technology   | Purpose                      |
| ------------ | ---------------------------- |
| Python       | Main programming language    |
| Pandas       | Data processing              |
| NumPy        | Numerical operations         |
| Scikit-learn | TF-IDF and cosine similarity |
| NLP          | Text-based recommendation    |
| Streamlit    | Interactive web dashboard    |
| Matplotlib   | Data visualization           |
| Seaborn      | Data visualization           |
| Git & GitHub | Version control              |

---

## 📁 Datasets

The project uses multiple movie and book datasets.

### Movie Data

* Netflix movie dataset
* IMDb movie dataset

The movie datasets are cleaned and combined into a processed movie dataset for recommendation.

### Book Data

Two book datasets are used for different purposes:

* Book recommendation dataset
* Larger book dataset for analytics and exploratory analysis

The datasets are preprocessed before being used by the application.

---

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/TishaSutariya/MovieBook-AI.git
```

Navigate to the project:

```bash
cd MovieBook-AI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Run the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

The application will open in your browser.

---

## 📌 Main Dashboard Sections

The Streamlit application contains:

1. 🏠 Home
2. 🎬 Bollywood Movie Recommendation
3. 🎬 All Movie Recommendation
4. 📚 Book Recommendation
5. 📊 Movie Analytics
6. 📖 Book Analytics
7. 🤖 AI / NLP Insights

---

## 🔮 Future Improvements

Possible improvements include:

* User preference-based recommendations
* Collaborative filtering
* Hybrid recommendation system
* Recommendation evaluation using real user interaction data
* More advanced NLP models
* Improved recommendation personalization
* Larger and regularly updated datasets
* User login and personalized recommendation history

---

## 👩‍💻 Author

**Tisha Sutariya**

B.Tech Artificial Intelligence & Data Science
Sarvajanik College of Engineering & Technology, Surat

GitHub: **TishaSutariya**

---

## ⭐ Project

If you find this project useful, consider giving the repository a star on GitHub.
