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

# --- Ratings loading and user–item pivot (for item–item CF) ---
@st.cache_data
def load_ratings(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, usecols=["userId", "movieId", "rating"])  # small and sufficient
    except FileNotFoundError:
        st.warning("ratings.csv not found. Item–item CF will be skipped.")
        return pd.DataFrame(columns=["userId", "movieId", "rating"]).astype({
            "userId": "int64", "movieId": "int64", "rating": "float64"
        })


@st.cache_resource
def build_user_item_pivot(ratings_df: pd.DataFrame) -> pd.DataFrame:
    if ratings_df.empty:
        return pd.DataFrame()
    pivot = ratings_df.pivot_table(index="userId", columns="movieId", values="rating", aggfunc="mean")
    pivot = pivot.fillna(0.0)
    return pivot


# Lazily load ratings and build pivot only when needed
RATINGS_CSV_PATH = os.environ.get("RATINGS_CSV_PATH", "ml-latest-small 2/ratings.csv")
ratings_df_cached = None
user_item_pivot_cached = None

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

    # Content-based similarity (genres TF-IDF)
    content_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).ravel()

    # Item–item collaborative similarity using user–item pivot
    global ratings_df_cached, user_item_pivot_cached
    if ratings_df_cached is None:
        ratings_df_cached = load_ratings(RATINGS_CSV_PATH)
    if user_item_pivot_cached is None:
        user_item_pivot_cached = build_user_item_pivot(ratings_df_cached)

    item_sim = np.zeros_like(content_sim)
    if not user_item_pivot_cached.empty:
        # Align pivot columns (movieIds) with our data order
        movie_ids_series = data.loc[:, "movieId"] if "movieId" in data.columns else None
        if movie_ids_series is not None:
            # Build matrix for item vectors: users x items
            pivot = user_item_pivot_cached
            # Ensure all movieIds appear in pivot (add missing columns filled with 0)
            missing_cols = [m for m in movie_ids_series if m not in pivot.columns]
            if missing_cols:
                for m in missing_cols:
                    pivot[m] = 0.0
            pivot = pivot.loc[:, movie_ids_series]  # reorder columns to match data

            # Compute cosine sim between selected movie column and all columns
            from numpy.linalg import norm
            target_vec = pivot.iloc[:, idx].values.astype(float)
            matrix = pivot.values.astype(float)
            # Cosine similarity manually to avoid large sklearn compute for 1 vector
            numerator = matrix.T @ target_vec  # shape: (n_items,)
            denom = (norm(matrix, axis=0) * norm(target_vec) + 1e-12)
            item_sim = numerator / denom
            item_sim = np.nan_to_num(item_sim, nan=0.0, posinf=0.0, neginf=0.0)

    # Hybrid score: blend content and item–item CF
    # Default weights; you can tune in UI if desired
    w_content = 0.7
    w_cf = 0.3
    final_score = w_content * content_sim + w_cf * item_sim

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
