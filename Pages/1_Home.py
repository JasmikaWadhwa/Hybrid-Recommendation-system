import streamlit as st
from utils import load_movies, load_ratings, train_model

st.set_page_config(
    page_title="Movie Recommender 💖",
    layout="wide",
    page_icon="🍿"
)

st.markdown("""
    <style>
    /* Full app background */
    .stApp {
        background: linear-gradient(to right, #ffe6f0, #fff5fa);
        font-family: "Trebuchet MS", sans-serif;
    }

    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #ffe6f0, #ffccf9);
    }

    /* Main title style */
    .title {
        font-size: 3rem;
        font-weight: bold;
        color: #ff4b9f;
        text-align: center;
        margin-bottom: 0.5em;
        text-shadow: 2px 2px 4px #ffcce0;
    }

    /* Subtitle */
    .subtitle {
        font-size: 1.3rem;
        color: #cc2e72;
        text-align: center;
        margin-bottom: 2em;
    }

    /* Box style */
    .box {
        background: white;
        padding: 20px;
        border-radius: 20px;
        box-shadow: 0px 4px 10px rgba(255, 128, 171, 0.3);
        margin-bottom: 20px;
    }

    /* Button style */
    .stButton button {
        background: linear-gradient(135deg, #ff99cc, #ff66b2);
        color: white;
        border-radius: 12px;
        font-weight: bold;
        font-size: 16px;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #ff66b2, #ff3385);
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("<div class='title'>🍓 One Stop for MOVIES 🎬</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Your cute little guide to finding the perfect movie 💕✨</div>", unsafe_allow_html=True)


with st.spinner("✨ Loading your magical movie universe... please wait 💫"):
    if "movies" not in st.session_state:
        st.session_state.movies = load_movies()
    if "ratings" not in st.session_state:
        st.session_state.ratings = load_ratings()
    if "model" not in st.session_state:
        trained_model, _ = train_model(st.session_state.ratings)
        st.session_state.model = trained_model  

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
        <div class="box">
            <h3 style="color:#ff4b9f; text-align:center;">💡 How it works</h3>
            <p style="text-align:center; color:#555;">
                We'll find you the <b>perfect movie</b> you'll love based on other users with similar taste. 🎀.
            </p>
            <p style="text-align:center; color:#555;">
                We hope you have a <b>mesmerizing</b> experience in the world of recommendations handpicked by us just for you💖.
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <hr style="border: 1px solid #ff99cc;">
    <p style="text-align:center; color:#cc2e72;">
        Made with 💖, popcorn 🍿, and a touch of sparkle ✨
    </p>
""", unsafe_allow_html=True)