import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader, SVD
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Set style for plots
sns.set_theme(style='whitegrid')

@st.cache_data
def load_movies():
    """Load movies data"""
    movies = pd.read_csv("movies.csv")
    return movies

@st.cache_data
def load_ratings():
    """Load ratings data"""
    ratings = pd.read_csv("ratings.csv")
    return ratings

@st.cache_data
def load_cleaned_movies():
    """Load cleaned movies data"""
    try:
        cleaned = pd.read_csv("cleaned_movies.csv")
        return cleaned
    except FileNotFoundError:
        # If cleaned_movies.csv doesn't exist, create it
        return create_cleaned_movies()

def create_cleaned_movies():
    """Create cleaned movies data from raw data"""
    movies = load_movies()
    ratings = load_ratings()
    
    # Clean movie titles and genres
    movies['title'] = movies['title'].fillna('').str.strip().str.lower()
    movies['genres'] = movies['genres'].fillna('(no genres listed)')
    movies['genres_list'] = movies['genres'].str.split('|')
    movies['genres_list'] = movies['genres_list'].apply(
        lambda lst: [g.strip().lower().replace('-', ' ') for g in lst] if isinstance(lst, list) else []
    )
    movies['genres_text'] = movies['genres_list'].apply(lambda lst: ' '.join(sorted(set(lst))))
    
    # Calculate ratings statistics
    ratings = ratings.dropna(subset=['movieId', 'rating'])
    agg = ratings.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        rating_count=('rating', 'size')
    ).reset_index()
    
    if not agg.empty:
        min_r, max_r = agg['avg_rating'].min(), agg['avg_rating'].max()
        agg['norm_rating'] = (agg['avg_rating'] - min_r) / (max_r - min_r) if max_r > min_r else 0.5
    else:
        agg['norm_rating'] = pd.Series(dtype=float)
    
    # Merge data
    df = movies.merge(agg, on='movieId', how='left')
    df['avg_rating'] = df['avg_rating'].fillna(0.0)
    df['norm_rating'] = df['norm_rating'].fillna(0.0)
    df['rating_count'] = df['rating_count'].fillna(0).astype(int)
    
    out_cols = ['movieId', 'title', 'genres_text', 'avg_rating', 'norm_rating', 'rating_count']
    cleaned = df[out_cols].copy()
    
    # Save for future use
    cleaned.to_csv('cleaned_movies.csv', index=False)
    
    return cleaned

@st.cache_data
def train_model(ratings):
    """Train SVD model"""
    reader = Reader(rating_scale=(ratings['rating'].min(), ratings['rating'].max()))
    data = Dataset.load_from_df(ratings[['userId','movieId','rating']], reader)
    trainset = data.build_full_trainset()
    algo = SVD(n_factors=50, n_epochs=20, random_state=42)
    algo.fit(trainset)
    return algo, trainset

def rating_to_stars(rating, max_rating=5):
    """Convert rating to star display"""
    if pd.isna(rating) or rating == 0:
        return "No rating"
    
    # Normalize rating to 5-star scale if needed
    if rating > 5:
        rating = rating / 2  # Assuming max rating is 10
    
    full_stars = int(rating)
    half_star = 1 if rating - full_stars >= 0.5 else 0
    empty_stars = max_rating - full_stars - half_star
    
    stars = "★" * full_stars + "☆" * half_star + "☆" * empty_stars
    return f"{stars} ({rating:.1f})"

def get_movie_rating(movie_id, ratings_df):
    """Get average rating for a movie"""
    movie_ratings = ratings_df[ratings_df['movieId'] == movie_id]['rating']
    if len(movie_ratings) == 0:
        return 0
    return movie_ratings.mean()
