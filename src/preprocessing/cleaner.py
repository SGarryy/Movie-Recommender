import pandas as pd
from sqlalchemy import text

from src.db.connection import get_connection


def _read_dataframe(query, params=None):
    with get_connection().connect() as connection:
        return pd.read_sql_query(text(query), connection, params=params)


def load_movies():
    return _read_dataframe("SELECT movie_id, title, year, genres FROM movies")


def load_ratings():
    return _read_dataframe("SELECT user_id, movie_id, rating FROM ratings")


def load_tags():
    return _read_dataframe("SELECT movie_id, tag FROM tags")


def load_movies_prefer_processed():
    from src.preprocessing.artifacts import has_processed_movies, load_processed_movies

    if has_processed_movies():
        return load_processed_movies()
    return load_movies()


def load_ratings_prefer_processed():
    from src.preprocessing.artifacts import has_processed_ratings, load_processed_ratings

    if has_processed_ratings():
        return load_processed_ratings()
    return load_ratings()


def clean_movies(df):
    if df.empty:
        return pd.DataFrame(columns=["movie_id", "title", "year", "genres"])

    cleaned = df.copy()
    cleaned["movie_id"] = pd.to_numeric(cleaned["movie_id"], errors="coerce")
    cleaned["year"] = pd.to_numeric(cleaned["year"], errors="coerce")
    cleaned["title"] = cleaned["title"].fillna("Unknown").astype(str).str.strip()
    cleaned["genres"] = cleaned["genres"].fillna("").astype(str)
    cleaned = cleaned.dropna(subset=["movie_id"])
    cleaned["movie_id"] = cleaned["movie_id"].astype(int)
    cleaned["year"] = cleaned["year"].fillna(0).astype(int)
    cleaned = cleaned.drop_duplicates(subset="movie_id").reset_index(drop=True)
    return cleaned


def clean_ratings(df):
    if df.empty:
        return pd.DataFrame(columns=["user_id", "movie_id", "rating"])

    cleaned = df.copy()
    cleaned["user_id"] = pd.to_numeric(cleaned["user_id"], errors="coerce")
    cleaned["movie_id"] = pd.to_numeric(cleaned["movie_id"], errors="coerce")
    cleaned["rating"] = pd.to_numeric(cleaned["rating"], errors="coerce")
    cleaned = cleaned.dropna(subset=["user_id", "movie_id", "rating"])
    cleaned["user_id"] = cleaned["user_id"].astype(int)
    cleaned["movie_id"] = cleaned["movie_id"].astype(int)
    cleaned["rating"] = cleaned["rating"].clip(0.5, 5.0).astype(float)
    cleaned = cleaned.drop_duplicates(
        subset=["user_id", "movie_id"], keep="last"
    ).reset_index(drop=True)
    return cleaned
