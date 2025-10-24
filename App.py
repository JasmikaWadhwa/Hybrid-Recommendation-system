import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os
import matplotlib.pyplot as plt

st.title("Hybrid Movie Recommender")

# --- Data loading ---
@st.cache_data
def load_cleaned_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        # Ensure expected columns exist
        expected = {"movieId", "tmdbId", "title", "genres_text", "avg_rating", "norm_rating", "rating_count"}
        missing = expected.difference(df.columns)
        if missing:
            st.error(f"Missing columns in cleaned data: {missing}. Please re-run preprocessing notebook.")
        return df
    except FileNotFoundError:
        st.error("cleaned_movies.csv not found. Please open and run eda.ipynb first.")
        return pd.DataFrame()


data = load_cleaned_data("cleaned_movies.csv")
if data.empty:
    st.stop()

# --- Content-based features (TF-IDF on genres text) ---
@st.cache_resource
def build_tfidf_matrix(genres_text: pd.Series):
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    matrix = tfidf_vectorizer.fit_transform(genres_text.fillna(""))
    return matrix


tfidf_matrix = build_tfidf_matrix(data["genres_text"])

# --- UI: movie selection ---
movie_titles = sorted(data["title"].dropna().unique().tolist())
selected_movie = st.selectbox("Pick a movie you like:", movie_titles, index=0 if movie_titles else None)


def get_tmdb_api_key() -> str | None:
    # Prefer Streamlit secrets, fallback to environment variable
    if "tmdb_api_key" in st.secrets:
        return st.secrets["tmdb_api_key"]
    return os.getenv("TMDB_API_KEY")


def fetch_poster_url(tmdb_id: float | int | None, api_key: str | None) -> str | None:
    if not api_key:
        return None
    try:
        # Try by tmdbId first when available
        if pd.notna(tmdb_id):
            tmdb_id_int = int(tmdb_id)
            r = requests.get(
                f"https://api.themoviedb.org/3/movie/{tmdb_id_int}", params={"api_key": api_key}, timeout=10
            )
            if r.ok:
                poster_path = r.json().get("poster_path")
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w200{poster_path}"
        return None
    except Exception:
        return None


if st.button("Recommend") and selected_movie:
    # Locate selected movie index (first match if duplicates)
    selected_idx_list = data.index[data["title"] == selected_movie].tolist()
    if not selected_idx_list:
        st.warning("Selected movie not found in data.")
        st.stop()
    idx = selected_idx_list[0]

    # Cosine similarities against all titles
    similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).ravel()

    # Hybrid score: 0.7 * content + 0.3 * normalized popularity (avg rating)
    norm_rating = data["norm_rating"].fillna(0.0).values
    final_score = 0.7 * similarities + 0.3 * norm_rating

    # Build results excluding the selected movie itself
    results = data.copy()
    results["final_score"] = final_score
    results = results[results.index != idx]
    recommendations = results.sort_values("final_score", ascending=False).head(10)

    st.subheader("Top 10 Recommendations")
    api_key = get_tmdb_api_key()
    for _, row in recommendations.iterrows():
        poster_url = fetch_poster_url(row.get("tmdbId"), api_key)
        st.markdown(f"**Title:** {str(row['title']).title()}")
        st.markdown(f"**Genres:** {row['genres_text']}")
        st.markdown(f"**Average Rating:** {round(float(row['avg_rating']), 2)}")
        if poster_url:
            st.image(poster_url, width=150)
        st.markdown("---")

    # Charts
    st.subheader("Top Genres Chart")
    genres_flat: list[str] = []
    for g in data["genres_text"].fillna(""):
        genres_flat += str(g).split()
    if genres_flat:
        genre_counts = pd.Series(genres_flat).value_counts().head(10)
        fig, ax = plt.subplots()
        genre_counts.plot(kind="bar", ax=ax)
        ax.set_ylabel("Count")
        ax.set_xlabel("Genre")
        st.pyplot(fig)

    st.subheader("Rating Distribution (Average per Movie)")
    fig2, ax2 = plt.subplots()
    data["avg_rating"].hist(bins=20, ax=ax2)
    ax2.set_xlabel("Average Rating")
    ax2.set_ylabel("Number of Movies")
    st.pyplot(fig2)
