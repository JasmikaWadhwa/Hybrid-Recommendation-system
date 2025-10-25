import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_movies, load_ratings, load_cleaned_movies
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

st.set_page_config(
    page_title="Statistics 📊",
    layout="wide",
    page_icon="📊"
)

# Set style for plots
sns.set_theme(style='whitegrid')

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main-header {
        text-align: center;
        color: #2c3e50;
        font-size: 2.5rem;
        font-weight: 300;
        margin-bottom: 2rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .section-header {
        color: #2c3e50;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: rgba(255,255,255,0.95);
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📊 Movie Dataset Statistics</h1>', unsafe_allow_html=True)

# Load data
@st.cache_data
def load_all_data():
    movies = load_movies()
    ratings = load_ratings()
    cleaned = load_cleaned_movies()
    return movies, ratings, cleaned

movies, ratings, cleaned = load_all_data()

# Basic statistics
st.markdown('<h2 class="section-header">📈 Dataset Overview</h2>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Movies", f"{len(movies):,}")
with col2:
    st.metric("Total Ratings", f"{len(ratings):,}")
with col3:
    st.metric("Unique Users", f"{ratings['userId'].nunique():,}")
with col4:
    st.metric("Avg Rating", f"{ratings['rating'].mean():.2f}")

# Genre Distribution Bar Plot
st.markdown('<h2 class="section-header">🎭 Genre Distribution</h2>', unsafe_allow_html=True)

# Extract genres from cleaned data
genres_flat = []
for g in cleaned['genres_text'].fillna(''):
    genres_flat += str(g).split()

if genres_flat:
    genre_counts = pd.Series(genres_flat).value_counts().head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=genre_counts.values, y=genre_counts.index, ax=ax, hue=genre_counts.index, palette='Blues_r', legend=False)
    ax.set_title('Top 10 Genres in Dataset', fontsize=16, fontweight='bold')
    ax.set_xlabel('Number of Movies', fontsize=12)
    ax.set_ylabel('Genre', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

# Rating Distribution Histogram
st.markdown('<h2 class="section-header">⭐ Rating Distribution</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Individual ratings distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(ratings['rating'], bins=10, kde=True, ax=ax, color='#4c72b0')
    ax.set_title('Individual Rating Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Rating', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    # Average rating per movie distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(cleaned['avg_rating'], bins=20, kde=True, ax=ax, color='#e74c3c')
    ax.set_title('Average Rating Distribution (per Movie)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Average Rating', fontsize=12)
    ax.set_ylabel('Number of Movies', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

# Pairplot
st.markdown('<h2 class="section-header">🔗 Rating Metrics Correlation</h2>', unsafe_allow_html=True)

pair_cols = ['avg_rating', 'norm_rating', 'rating_count']
subset = cleaned[pair_cols].copy()
subset['rating_count_clipped'] = subset['rating_count'].clip(upper=subset['rating_count'].quantile(0.99))

fig = sns.pairplot(
    subset.rename(columns={'rating_count_clipped': 'rating_count (clipped)'}),
    vars=['avg_rating', 'norm_rating', 'rating_count (clipped)'],
    diag_kind='hist',
    corner=True,
    plot_kws={'alpha': 0.6}
)
fig.fig.suptitle('Pairplot: Rating Metrics Correlation', y=1.02, fontsize=16, fontweight='bold')
st.pyplot(fig.fig)

# Pie Chart for Rating Distribution
st.markdown('<h2 class="section-header">🥧 Rating Distribution (Pie Chart)</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Individual ratings pie chart
    rating_counts = ratings['rating'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(rating_counts)))
    wedges, texts, autotexts = ax.pie(rating_counts.values, labels=rating_counts.index, 
                                     autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('Individual Rating Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    # Top genres pie chart
    top_genres = genre_counts.head(8)
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(top_genres)))
    wedges, texts, autotexts = ax.pie(top_genres.values, labels=top_genres.index, 
                                     autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('Top 8 Genres Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

# Model Performance Analysis
st.markdown('<h2 class="section-header">🎯 Model Performance Analysis</h2>', unsafe_allow_html=True)

# Split data for evaluation
ratings_train, ratings_test = train_test_split(ratings, test_size=0.2, random_state=42)

def rmse(y_true, y_pred):
    """Compute Root Mean Squared Error."""
    return sqrt(mean_squared_error(y_true, y_pred))

def evaluate_predictions(test_df, pred_func):
    """Evaluate recommender using a given prediction function."""
    preds, trues = [], []
    for _, row in test_df.iterrows():
        user, movie, true_r = int(row['userId']), int(row['movieId']), row['rating']
        try:
            pred_r = pred_func(user, movie)
            preds.append(pred_r)
            trues.append(true_r)
        except Exception:
            continue
    if not preds:
        return np.nan
    return rmse(trues, preds)

# Baseline predictor
global_mean = ratings_train['rating'].mean()
def baseline_predictor(user, movie):
    return global_mean

baseline_rmse = evaluate_predictions(ratings_test, baseline_predictor)

# Collaborative filtering predictor
user_means = ratings_train.groupby('userId')['rating'].mean()
movie_means = ratings_train.groupby('movieId')['rating'].mean()

def cf_predictor(user, movie):
    """Predict rating using weighted combination of user and movie mean."""
    u_mean = user_means.get(user, global_mean)
    m_mean = movie_means.get(movie, global_mean)
    return (u_mean + m_mean) / 2.0

cf_rmse = evaluate_predictions(ratings_test, cf_predictor)

# Display results
results = pd.DataFrame({
    'Model': ['Baseline (Global Mean)', 'Collaborative Filtering'],
    'RMSE': [baseline_rmse, cf_rmse]
})

col1, col2 = st.columns([1, 1])

with col1:
    st.dataframe(results, use_container_width=True)

with col2:
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(results['Model'], results['RMSE'], color=['#6baed6', '#9ecae1'])
    ax.set_title('Model RMSE Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('RMSE (lower is better)', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    plt.xticks(rotation=20)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)

# Additional Statistics
st.markdown('<h2 class="section-header">📋 Additional Statistics</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h4>Rating Statistics</h4>
        <p><strong>Min Rating:</strong> {:.1f}</p>
        <p><strong>Max Rating:</strong> {:.1f}</p>
        <p><strong>Median Rating:</strong> {:.1f}</p>
        <p><strong>Std Deviation:</strong> {:.2f}</p>
    </div>
    """.format(
        ratings['rating'].min(),
        ratings['rating'].max(),
        ratings['rating'].median(),
        ratings['rating'].std()
    ), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h4>Movie Statistics</h4>
        <p><strong>Movies with Ratings:</strong> {:,}</p>
        <p><strong>Avg Ratings per Movie:</strong> {:.1f}</p>
        <p><strong>Most Rated Movie:</strong> {:,} ratings</p>
        <p><strong>Least Rated Movie:</strong> {:,} ratings</p>
    </div>
    """.format(
        len(cleaned[cleaned['rating_count'] > 0]),
        ratings.groupby('movieId').size().mean(),
        ratings.groupby('movieId').size().max(),
        ratings.groupby('movieId').size().min()
    ), unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h4>User Statistics</h4>
        <p><strong>Avg Ratings per User:</strong> {:.1f}</p>
        <p><strong>Most Active User:</strong> {:,} ratings</p>
        <p><strong>Least Active User:</strong> {:,} ratings</p>
        <p><strong>Sparsity:</strong> {:.2%}</p>
    </div>
    """.format(
        ratings.groupby('userId').size().mean(),
        ratings.groupby('userId').size().max(),
        ratings.groupby('userId').size().min(),
        1 - (len(ratings) / (ratings['userId'].nunique() * ratings['movieId'].nunique()))
    ), unsafe_allow_html=True)

