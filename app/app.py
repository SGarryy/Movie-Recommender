import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import streamlit as st
import pandas as pd
from src.recommender import recommend, search_movie, get_top_rated

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.markdown("""
<style>
    .main { background-color: #141414; }
    h1, h2, h3, p, div { color: #ffffff; }
    .stTextInput input { background-color: #1f1f1f; color: white; }
    .movie-card {
        background-color: #1f1f1f;
        padding: 15px;
        border-radius: 8px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎬 Movie Recommender")

tabs = st.tabs(["🔍 Search & Recommend", "⭐ Top Rated", "👤 For You"])

with tabs[0]:
    st.subheader("Find Similar Movies")
    query = st.text_input("Search for a movie")

    if query:
        results = search_movie(query)
        if results.empty:
            st.warning("No movies found.")
        else:
            selected = st.selectbox(
                "Select a movie",
                options=results['movie_id'].tolist(),
                format_func=lambda x: results[results['movie_id'] == x]['title'].values[0]
            )
            if st.button("Get Recommendations"):
                with st.spinner("Finding similar movies..."):
                    recs = recommend(movie_id=selected, top_n=10)
                if recs.empty:
                    st.warning("No recommendations found.")
                else:
                    st.subheader("Recommended Movies")
                    for _, row in recs.iterrows():
                        st.markdown(f"""
                        <div class="movie-card">
                            <h3>{row['title']} ({int(row['year']) if row['year'] else 'N/A'})</h3>
                            <p>Genres: {row['genres']}</p>
                            <p>Score: {round(row['score'], 3)}</p>
                        </div>
                        """, unsafe_allow_html=True)

with tabs[1]:
    st.subheader("Top Rated Movies")
    with st.spinner("Loading..."):
        top = get_top_rated(20)
    if not top.empty:
        for _, row in top.iterrows():
            st.markdown(f"""
            <div class="movie-card">
                <h3>{row['title']} ({int(row['year']) if row['year'] else 'N/A'})</h3>
                <p>Genres: {row['genres']}</p>
                <p>Average Rating: {round(row['avg_rating'], 2)} ⭐ ({int(row['rating_count'])} ratings)</p>
            </div>
            """, unsafe_allow_html=True)

with tabs[2]:
    st.subheader("Personalized Recommendations")
    user_id = st.number_input("Enter your User ID", min_value=1, step=1)
    if st.button("Get My Recommendations"):
        with st.spinner("Generating recommendations..."):
            recs = recommend(user_id=int(user_id), top_n=10)
        if recs.empty:
            st.warning("No recommendations found for this user.")
        else:
            for _, row in recs.iterrows():
                st.markdown(f"""
                <div class="movie-card">
                    <h3>{row['title']} ({int(row['year']) if row['year'] else 'N/A'})</h3>
                    <p>Genres: {row['genres']}</p>
                </div>
                """, unsafe_allow_html=True)