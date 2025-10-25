import streamlit as st

def faq_page():
  
    st.markdown("""
    <style>
        /* Full app background gradient */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        }

       
        section[data-testid="stSidebar"] {
            background: linear-gradient(to bottom, #f5f7fa, #c3cfe2) !important;
            color: #2c3e50;
        }

     
        .faq-title {
            color: #2c3e50;
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .faq-subtitle {
            color: #34495e;
            font-size: 18px;
            text-align: center;
            margin-bottom: 30px;
        }

        .stExpanderHeader {
            font-weight: bold;
            color: #2c3e50;
        }
        .stExpanderContent {
            background-color: rgba(255,255,255,0.95) !important;
            border-radius: 10px;
            padding: 10px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }


        .faq-text {
            font-size: 16px;
            color: #34495e;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='faq-title'>❓ Frequently Asked Questions</div>", unsafe_allow_html=True)
    st.markdown("<div class='faq-subtitle'>Find answers to common questions here ✨</div>", unsafe_allow_html=True)

    faqs = [
        

        ("Is my data private?",
         "Personally identifiable information is stored for reference. Your ratings and preferences remain anonymous."),

        ("How accurate are the recommendations?",
         "The recommendations are predictions based on past user ratings. They are not guaranteed, "
         "but usually reflect your taste fairly well."),

        ("Can I use my own ratings?",
         "Currently, the system uses the ml-small-dataset data for recommendations. "
         "Future updates may allow personal rating uploads."),

        ("How many recommendations will I get?",
         "You can get the maximum possible personalized movie recommendations based on your User ID, Genres, available OTTs."),

        ("Why do I need to enter a User ID?",
         "The User ID identifies which user's preferences to use for generating recommendations. "
         "It allows the system to tailor suggestions specifically for you."),

        ("Can I explore the full movie dataset?",
         "Yes! You can browse by Genres and Search for the movies you like"),

        ("Can I see visual statistics of movies and ratings?",
         "Absolutely! Check the 'Staitics' page for ratings distribution and charts.")
    ]

    for question, answer in faqs:
        with st.expander(f"💖 {question}", expanded=False):
            st.markdown(f"<p class='faq-text'>{answer}</p>", unsafe_allow_html=True)

faq_page()
