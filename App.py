# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import matplotlib.pyplot as plt

# Load preprocessed data
data = pd.read_csv('cleaned_movies.csv')

st.title("Hybrid Movie Recommender")

# Prepare average ratings and normalization
avg_ratings = data.groupby('movieId')['rating'].mean().reset_index()
rating_min, rating_max = avg_ratings['rating'].min(), avg_ratings['rating'].max()
avg_ratings['norm_rating'] = (avg_ratings['rating'] - rating_min) / (rating_max - rating_min)

# Merge normalized average ratings
data = pd.merge(data, avg_ratings[['movieId', 'norm_rating']], on='movieId', how='left')

# TF-IDF on 'overview' + 'genres_text'
data['combined'] = (data['overview'].fillna('') + ' ' + data['genres_text'].fillna(''))
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['combined'])

# Movie selection interface
movie_titles = data['title'].unique()
selected_movie = st.selectbox("Pick a movie you like:", sorted(movie_titles))

if st.button("Recommend"):
    idx = data[data['title'] == selected_movie].index[0]
    similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    # Combine with normalized average rating
    final_score = 0.7 * similarities + 0.3 * data['norm_rating']
    data['final_score'] = final_score

    # Top-10 recommendations (exclude selected movie)
    recommendations = data[data['title'] != selected_movie].sort_values('final_score', ascending=False).head(10)
    
    st.subheader("Top 10 Recommendations")
    for _, row in recommendations.iterrows():
        # TMDb API Poster
        poster_url = None
        tmdb_key = "YOUR_TMDB_API_KEY"
        response = requests.get(
            f"https://api.themoviedb.org/3/search/movie?api_key={tmdb_key}&query={row['title']}"
        )
        if response.ok and response.json()['results']:
            poster_path = response.json()['results'][0].get('poster_path', None)
            if poster_path:
                poster_url = f"https://image.tmdb.org/t/p/w200{poster_path}"
        st.markdown(f"**Title:** {row['title'].title()}")
        st.markdown(f"**Genres:** {row['genres_text']}")
        st.markdown(f"**Average Rating:** {round(row['rating'],2)}")
        if poster_url:
            st.image(poster_url, width=150)
        st.markdown("---")
    
    # Charts (Top genres)
    st.subheader("Top Genres Chart")
    genres_flat = []
    for g in data['genres_text']:
        genres_flat += g.split()
    genre_counts = pd.Series(genres_flat).value_counts().head(10)
    fig, ax = plt.subplots()
    genre_counts.plot(kind='bar', ax=ax)
    st.pyplot(fig)

    # Rating distribution
    st.subheader("Rating Distribution")
    fig2, ax2 = plt.subplots()
    data['rating'].hist(bins=20, ax=ax2)
    st.pyplot(fig2)
