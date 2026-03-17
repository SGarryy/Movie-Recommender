import pandas as pd

from src.models.hybrid import hybrid_recommend
from src.preprocessing.cleaner import (
    clean_movies,
    clean_ratings,
    load_movies_prefer_processed,
    load_ratings_prefer_processed,
)

RECOMMENDATION_COLUMNS = ["movie_id", "title", "genres", "year", "score", "source"]
SEARCH_COLUMNS = ["movie_id", "title", "year", "genres"]
TOP_RATED_COLUMNS = [
    "movie_id",
    "avg_rating",
    "rating_count",
    "title",
    "genres",
    "year",
]


def recommend(movie_id=None, user_id=None, top_n=10):
    if movie_id is None and user_id is None:
        return pd.DataFrame(columns=RECOMMENDATION_COLUMNS)

    results = hybrid_recommend(movie_id=movie_id, user_id=user_id, top_n=top_n)
    if results.empty:
        return pd.DataFrame(columns=RECOMMENDATION_COLUMNS)

    return results.reset_index(drop=True)


def search_movie(title_query):
    query = (title_query or "").strip()
    if not query:
        return pd.DataFrame(columns=SEARCH_COLUMNS)

    movies_df = clean_movies(load_movies_prefer_processed())
    mask = movies_df["title"].str.contains(query, case=False, na=False, regex=False)
    results = movies_df.loc[mask, SEARCH_COLUMNS]
    return results.reset_index(drop=True)


def get_top_rated(top_n=10, min_ratings=100):
    ratings_df = clean_ratings(load_ratings_prefer_processed())
    movies_df = clean_movies(load_movies_prefer_processed())

    if ratings_df.empty or movies_df.empty:
        return pd.DataFrame(columns=TOP_RATED_COLUMNS)

    top = (
        ratings_df.groupby("movie_id")
        .agg(avg_rating=("rating", "mean"), rating_count=("rating", "count"))
        .reset_index()
    )
    top = top[top["rating_count"] >= min_ratings]
    top = top.sort_values(["avg_rating", "rating_count"], ascending=[False, False]).head(top_n)
    top = top.merge(movies_df[["movie_id", "title", "genres", "year"]], on="movie_id")
    return top[TOP_RATED_COLUMNS].reset_index(drop=True)
