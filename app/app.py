import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommender import get_top_rated, recommend, search_movie

st.set_page_config(page_title="Movie Recommender", layout="wide")


@st.cache_data(show_spinner=False)
def cached_search_movie(query):
    return search_movie(query)


@st.cache_data(show_spinner=False)
def cached_top_rated(limit):
    return get_top_rated(limit)


@st.cache_data(show_spinner=False)
def cached_recommend(movie_id, user_id, top_n):
    return recommend(movie_id=movie_id, user_id=user_id, top_n=top_n)


def format_year(year):
    if pd.notna(year):
        try:
            year_value = int(year)
            if year_value > 0:
                return str(year_value)
        except (TypeError, ValueError):
            pass
    return "N/A"


def render_movie_card(row, score_label=None):
    title = row.get("title", "Unknown")
    genres = row.get("genres", "") or "N/A"

    with st.container():
        st.markdown(f"**{title} ({format_year(row.get('year'))})**")
        st.write(f"Genres: {genres}")
        if score_label:
            st.caption(score_label)
        st.divider()


st.title("Movie Recommender")

tabs = st.tabs(["Search and Recommend", "Top Rated", "For You"])

with tabs[0]:
    st.subheader("Find Similar Movies")
    query = st.text_input("Search for a movie")
    if query:
        results = cached_search_movie(query)
        if results.empty:
            st.warning("No movies found.")
        else:
            movie_lookup = dict(zip(results["movie_id"], results["title"]))
            selected = st.selectbox(
                "Select a movie",
                options=results["movie_id"].tolist(),
                format_func=lambda movie_id: movie_lookup.get(movie_id, str(movie_id)),
            )
            if st.button("Get Recommendations", key="search_recommendations"):
                with st.spinner("Finding similar movies..."):
                    recs = cached_recommend(int(selected), None, 10)
                if recs.empty:
                    st.warning("No recommendations found.")
                else:
                    st.subheader("Recommended Movies")
                    for _, row in recs.iterrows():
                        score_display = (
                            f"Score: {round(row['score'], 3)}" if "score" in recs.columns else None
                        )
                        render_movie_card(row, score_display)

with tabs[1]:
    st.subheader("Top Rated Movies")
    with st.spinner("Loading..."):
        top = cached_top_rated(20)
    if top.empty:
        st.info("No top-rated movies are available yet.")
    else:
        for _, row in top.iterrows():
            score_display = (
                f"Average Rating: {round(row['avg_rating'], 2)} "
                f"({int(row['rating_count'])} ratings)"
            )
            render_movie_card(row, score_display)

with tabs[2]:
    st.subheader("Personalized Recommendations")
    user_id = st.number_input("Enter your User ID", min_value=1, step=1)
    if st.button("Get My Recommendations", key="user_recommendations"):
        with st.spinner("Generating recommendations..."):
            recs = cached_recommend(None, int(user_id), 10)
        if recs.empty:
            st.warning("No recommendations found for this user.")
        else:
            for _, row in recs.iterrows():
                score_display = (
                    f"Score: {round(row['score'], 3)}" if "score" in recs.columns else None
                )
                render_movie_card(row, score_display)
