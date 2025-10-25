import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from scipy.sparse import vstack
import math

st.set_page_config(page_title="Movie Recommender", layout="wide", page_icon="🎬")

st.markdown(
    """
    <style>
    .stApp { 
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .card { 
        background: rgba(255,255,255,0.95); 
        padding: 24px; 
        border-radius: 16px; 
        box-shadow: 0 8px 32px rgba(0,0,0,0.1); 
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    .muted { color:#6b7280; font-size:14px; }
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
    .rating-stars {
        color: #ffd700;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h1 class="main-header">🎬 Movie Recommender</h1>', unsafe_allow_html=True)

def load_defaults(base_dir="."):
    base = Path(base_dir)
    movies = pd.read_csv(base / "movies.csv")
    ratings = pd.read_csv(base / "ratings.csv")
    return movies, ratings

def preprocess_movies(movies):
    m = movies.copy()
    for candidate in ['movieId','movie_id','movieID','id']:
        if candidate in m.columns:
            m.rename(columns={candidate:'movieId'}, inplace=True)
            break
    if 'title' not in m.columns:
        raise ValueError("movies.csv must have a 'title' column.")
    m['title'] = m['title'].astype(str)
    m['genres'] = m.get('genres', '').fillna('(no genres listed)').astype(str)
    m['content'] = m['title'].str.lower() + " " + m['genres'].str.replace('|', ' ')
    m = m.reset_index(drop=True)
    return m

def build_tfidf(movies):
    tfidf = TfidfVectorizer(stop_words='english', max_features=8000)
    tfidf_matrix = tfidf.fit_transform(movies['content'])
    return tfidf, tfidf_matrix

@st.cache_data
def train_svd(ratings, n_factors=50, n_epochs=20, random_state=42):
    # surprise expects a dataframe with userId, itemId, rating
    reader = Reader(rating_scale=(ratings['rating'].min(), ratings['rating'].max()))
    data = Dataset.load_from_df(ratings[['userId','movieId','rating']], reader)
    trainset = data.build_full_trainset()
    algo = SVD(n_factors=n_factors, n_epochs=n_epochs, random_state=random_state)
    algo.fit(trainset)
    return algo

def rating_to_stars(rating, max_rating=5):
    """Convert rating to star display"""
    if pd.isna(rating) or rating == 0:
        return "No rating"

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

def top_popular_movies(movies, ratings, top_n=10):
    agg = ratings.groupby('movieId').agg(rating_count=('rating','count'), avg_rating=('rating','mean')).reset_index()
    pop = movies.merge(agg, on='movieId', how='left').fillna({'rating_count':0, 'avg_rating':0.0})
    top = pop.sort_values(['rating_count','avg_rating'], ascending=[False, False]).head(top_n)
    return top

def content_scores_for_user(movies, tfidf_matrix, tfidf, user_seen_ids):
    # create a mean vector of the user's seen movies (based on their indices)
    idx_map = {mid: idx for idx, mid in enumerate(movies['movieId'].values)}
    seen_idx = [idx_map[m] for m in user_seen_ids if m in idx_map]
    if not seen_idx:
        return None  # user cold-start for content
    # mean vector
    vectors = tfidf_matrix[seen_idx]
    mean_vec = vectors.mean(axis=0)  
    sims = linear_kernel(mean_vec, tfidf_matrix).flatten()
    return sims  

def content_similar_to_title(movies, tfidf_matrix, query_title, top_n=10):
 
    q = query_title.strip().lower()
    if q == "":
        return pd.DataFrame([])
    
    exact_match = movies[movies['title'].str.lower() == q]
    if not exact_match.empty:
        idx = exact_match.index[0]
    else:
      
        candidates = movies[movies['title'].str.lower().str.contains(q, na=False)]
        if not candidates.empty:
            idx = candidates.index[0]
        else:
            tokens = set(q.split())
            if not tokens:
                return pd.DataFrame([])
            
            movies_copy = movies.copy()
            movies_copy['__token_overlap'] = movies_copy['title'].str.lower().apply(
                lambda s: len(tokens.intersection(set(s.split()))) if pd.notna(s) else 0
            )
         
            candidates = movies_copy[movies_copy['__token_overlap'] > 0].sort_values('__token_overlap', ascending=False)
            if candidates.empty:
                return pd.DataFrame([])
            
            idx = candidates.index[0]
    
    try:
        sims = linear_kernel(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
        similar_idx = sims.argsort()[::-1]
        
        recs = []
        count = 0
        for i in similar_idx:
            if i == idx:
                continue  # Skip the query movie itself
            if count >= top_n:
                break
                
            recs.append({
                'movieId': movies.iloc[i]['movieId'], 
                'title': movies.iloc[i]['title'], 
                'score': float(sims[i])
            })
            count += 1
            
        return pd.DataFrame(recs)
        
    except Exception as e:
       
        return pd.DataFrame([])

def recommend_for_user_hybrid(user_id, movies, ratings, svd_algo, tfidf_matrix, alpha=0.5, top_n=10):

    movie_ids = movies['movieId'].values

    user_ratings = ratings[ratings['userId'] == user_id]
    seen = set(user_ratings['movieId'].unique())

    if user_ratings.empty:
  
        pop = top_popular_movies(movies, ratings, top_n=top_n*2)
  
        pop = pop.head(top_n)
        pop = pop[['movieId','title','rating_count','avg_rating']]
        pop['score'] = pop['avg_rating'] + np.log1p(pop['rating_count'])/10.0
        return pop[['movieId','title','score']].reset_index(drop=True)

    candidate_mask = ~movies['movieId'].isin(seen)
    candidate_movies = movies[candidate_mask].reset_index(drop=True)
    candidate_ids = candidate_movies['movieId'].values


    collab_scores = []
    for mid in candidate_ids:
        # surprise's predict requires raw ids
        pred = svd_algo.predict(str(user_id), str(mid))  # using str to match training if we used strings
        collab_scores.append(pred.est)
    collab_scores = np.array(collab_scores, dtype=float)


    content_sims = content_scores_for_user(movies.reset_index(drop=True), tfidf_matrix, None, list(seen))
    if content_sims is None:
        # no content for user (shouldn't happen since movies have content)
        content_scores = np.zeros_like(collab_scores)
    else:

        full_movies = movies.reset_index(drop=True)
        idx_map = {m: idx for idx, m in enumerate(full_movies['movieId'].values)}
        content_scores = np.array([content_sims[idx_map[mid]] for mid in candidate_ids], dtype=float)


    def normalize_vec(v):
        if np.all(np.isnan(v)) or v.max() - v.min() < 1e-9:
            return np.zeros_like(v)
        return (v - v.min()) / (v.max() - v.min())

    collab_n = normalize_vec(collab_scores)
    content_n = normalize_vec(content_scores)

    final = alpha * content_n + (1 - alpha) * collab_n
    top_idx = np.argsort(final)[::-1][:top_n]
    recs = []
    for i in top_idx:
        recs.append({'movieId': int(candidate_ids[i]), 'title': candidate_movies.iloc[i]['title'], 'score': float(final[i])})
    return pd.DataFrame(recs)

movies, ratings = load_defaults()

movies = preprocess_movies(movies)

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown('<h2 class="section-header">⚙️ Recommendation Settings</h2>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    alpha = st.slider("Content vs Collaborative Balance", min_value=0.0, max_value=1.0, value=0.5, step=0.05, help="Higher values favor content similarity, lower values favor collaborative filtering")
with col2:
    top_n = st.slider("Number of Recommendations", min_value=5, max_value=25, value=10)
st.markdown("</div>", unsafe_allow_html=True)

with st.spinner("Training SVD (collaborative) — this may take a moment..."):

    ratings_s = ratings.copy()
    ratings_s['userId'] = ratings_s['userId'].astype(str)
    ratings_s['movieId'] = ratings_s['movieId'].astype(str)
    svd_algo = train_svd(ratings_s, n_factors=50, n_epochs=20)

with st.spinner("Building TF-IDF (content)"):
    tfidf_vec, tfidf_matrix = build_tfidf(movies)

st.markdown('<h2 class="section-header">🎬 Get Movie Recommendations</h2>', unsafe_allow_html=True)

col1, col2 = st.columns([1,1])
with col1:
    st.markdown("### 👤 User Recommendations")
    st.markdown("Get personalized recommendations based on your movie history")
    
    # Provide list of users; convert to ints for display but model used strings
    unique_users = sorted(ratings['userId'].unique())
    user_choice = st.selectbox("Choose user ID:", options=[-1] + unique_users, help="Select a user ID to get personalized recommendations")
    
    custom_fav = st.text_input("New user? Enter your favorite movies (comma-separated):", placeholder="e.g., The Dark Knight, Inception, Pulp Fiction", help="For new users, enter 1-3 favorite movie titles to get personalized recommendations")
    
    if st.button("🎯 Get My Recommendations", type="primary"):
            if user_choice == -1:
               
                if custom_fav.strip():
                    favs = [t.strip() for t in custom_fav.split(",")][:3]
                    # try to find the first matching movie index
                    matched = movies[movies['title'].str.lower().str.contains(favs[0].lower())] if favs else pd.DataFrame()
                    # content-based similar using first favourite
                    if not matched.empty:
                        recs = content_similar_to_title(movies, tfidf_matrix, favs[0], top_n=top_n)
                        if recs.empty:
                            recs = top_popular_movies(movies, ratings, top_n=top_n)[['movieId','title']].assign(score=0.0)
                    else:
                        recs = top_popular_movies(movies, ratings, top_n=top_n)[['movieId','title']].assign(score=0.0)
                else:
                    recs = top_popular_movies(movies, ratings, top_n=top_n)[['movieId','title']].assign(score=0.0)
                
                st.success(f"🎉 Found {len(recs)} recommendations for you!")
                
                cols = st.columns(2)
                for i, (_, row) in enumerate(recs.iterrows()):
                    if i % 2 == 0:
                        container = cols[0].container()
                    else:
                        container = cols[1].container()
                    with container:
                        st.markdown(f"📷 **{row['title']}**")
                        # Get and display rating
                        movie_rating = get_movie_rating(row['movieId'], ratings)
                        stars_display = rating_to_stars(movie_rating)
                        st.markdown(f'<span class="rating-stars">{stars_display}</span>', unsafe_allow_html=True)
            else:
                # convert user id to string for model
                user_str = str(user_choice)
                recs = recommend_for_user_hybrid(user_str, movies, ratings, svd_algo, tfidf_matrix, alpha=alpha, top_n=top_n)
                if recs.empty:
                    st.info("No recommendations found for this user.")
                else:
                    st.success(f"🎉 Found {len(recs)} personalized recommendations!")
                    # display with posters if available
                    rows = []
                    for _, r in recs.iterrows():
                        rows.append(r)
                    df = pd.DataFrame(rows)
                    # show cards with posters
                    cols = st.columns(2)
                    for i, (_, row) in enumerate(df.iterrows()):
                        if i % 2 == 0:
                            container = cols[0].container()
                        else:
                            container = cols[1].container()
                        with container:
                            st.markdown(f"📷 **{row['title']}**")
                            # Get and display rating
                            movie_rating = get_movie_rating(row['movieId'], ratings)
                            stars_display = rating_to_stars(movie_rating)
                            st.markdown(f'<span class="rating-stars">{stars_display}</span>', unsafe_allow_html=True)

with col2:
    st.markdown("### 🔍 Find Similar Movies")
    st.markdown("Discover movies similar to your favorite titles")
    
    q = st.text_input("Enter a movie title:", placeholder="e.g., The Matrix", help="Type any movie title to find similar movies")
    
    if st.button("🔍 Find Similar Movies", type="primary"):
        if q.strip() == "":
            st.warning("Please enter a movie title to find similar movies.")
        else:
            with st.spinner(f"Searching for movies similar to '{q}'..."):
                recs2 = content_similar_to_title(movies, tfidf_matrix, q, top_n=top_n)
            
            if recs2.empty:
                st.info("No similar movies found. Try a different title or check the spelling.")
                # Show some suggestions
                st.markdown("**💡 Try searching for:**")
                sample_titles = movies['title'].head(5).tolist()
                for title in sample_titles:
                    st.markdown(f"- {title}")
            else:
                st.success(f"🎉 Found {len(recs2)} movies similar to '{q}'!")
                
                # Display results with camera emoji
                for i, (_, r) in enumerate(recs2.iterrows()):
                    with st.container():
                        st.markdown(f"📷 **{r['title']}**")
                        # Get and display rating
                        movie_rating = get_movie_rating(r['movieId'], ratings)
                        stars_display = rating_to_stars(movie_rating)
                        st.markdown(f'<span class="rating-stars">{stars_display}</span>', unsafe_allow_html=True)
                        st.markdown("---")


st.markdown('<h2 class="section-header">📊 Most Popular Movies</h2>', unsafe_allow_html=True)
top_pop = top_popular_movies(movies, ratings, top_n=10)

display_data = []
for _, row in top_pop.iterrows():
    stars_display = rating_to_stars(row['avg_rating'])
    display_data.append({
        'Title': row['title'],
        'Rating': stars_display,
        'Votes': int(row['rating_count'])
    })

display_df = pd.DataFrame(display_data)
st.table(display_df)

with st.expander("ℹ️ About This Recommender", expanded=False):
    st.markdown("""
    <div style="padding: 16px; background: rgba(255,255,255,0.1); border-radius: 8px;">
        <h4 style="color: #2c3e50; margin-top: 0;">How It Works</h4>
        <p>This hybrid recommender combines two powerful techniques:</p>
        <ul>
            <li><strong>🤝 Collaborative Filtering</strong>: Uses SVD to find users with similar tastes and recommend movies they liked</li>
            <li><strong>📝 Content-Based Filtering</strong>: Analyzes movie genres and titles to find similar content</li>
        </ul>
       
    </div>
    """, unsafe_allow_html=True)
