
import streamlit as st
import pandas as pd


st.set_page_config(layout="wide", page_title="🎭 Genres Dashboard", page_icon="🎬")


st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important; font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;}
    .title-text {text-align:center;font-size:44px;font-weight:900;color:#2c3e50;text-shadow:0 2px 4px rgba(0,0,0,0.1);}
    .subtitle-text {text-align:center;font-size:18px;color:#34495e;margin-bottom:1.2em;}
    .genre-box {background:rgba(255,255,255,0.95);border-radius:16px;padding:12px;box-shadow:0 8px 32px rgba(0,0,0,0.1);border:1px solid rgba(255,255,255,0.2);backdrop-filter:blur(10px);}
    .movie-pill {display:inline-block;background:white;border-radius:12px;padding:4px 10px;margin:4px 6px;color:#2c3e50;border:1px solid #c3cfe2;font-weight:600;box-shadow:0 2px 8px rgba(0,0,0,0.1);}
    .stExpander > div {background: transparent;}
    .count-badge {background:#667eea;color:white;border-radius:12px;padding:2px 8px;font-size:12px;margin-left:6px;}
    .stTextInput label {font-weight: bold; color: #2c3e50 !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

GENRE_COLUMNS = [
    'unknown','Action','Adventure','Animation','Children','Comedy','Crime',
    'Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery',
    'Romance','Sci-Fi','Thriller','War','Western'
]

@st.cache_data(show_spinner=False)
def load_movies_with_genres() -> pd.DataFrame:
    df = pd.read_csv(
        "http://files.grouplens.org/datasets/movielens/ml-100k/u.item",
        sep="|",
        encoding="latin-1",
        header=None,
        names=['movie_id','title','release_date','video_release_date','IMDb_URL'] + GENRE_COLUMNS
    )
    return df[['movie_id', 'title'] + GENRE_COLUMNS]

def main():
    # Try to reuse session state
    movies_ss = st.session_state.get("movies", None)
    if movies_ss is not None and set(GENRE_COLUMNS).issubset(set(movies_ss.columns)):
        movies_df = movies_ss.copy()
    else:
        movies_df = load_movies_with_genres()
        st.session_state["movies"] = movies_df

    # UI
    st.markdown("<div class='title-text'>🎭 Browse by Genre</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle-text'>Click a genre to see all matching movies ✨</div>", unsafe_allow_html=True)

    title_filter = st.text_input("🔎 Filter titles (applies inside each genre):", "")

    DISPLAY_GENRES = [g for g in GENRE_COLUMNS if g != 'unknown']
    cols = st.columns(3)

    for idx, genre in enumerate(DISPLAY_GENRES):
        with cols[idx % 3]:
            genre_mask = movies_df[genre] == 1
            genre_movies = movies_df.loc[genre_mask, ['title']].copy()

            if title_filter:
                genre_movies = genre_movies[genre_movies['title'].str.contains(title_filter, case=False, na=False)]

            count = int(genre_movies.shape[0])
            with st.expander(genre, expanded=False):
                st.caption(f"{count} movies")
                if count == 0:
                    st.info("No movies match the current filter.")
                else:
                    movie_titles = genre_movies['title'].sort_values().tolist()
                    pills_html = "".join([f"<span class='movie-pill'>{t}</span>" for t in movie_titles])
                    st.markdown(f"<div class='genre-box'>{pills_html}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
