import pandas as pd
from sqlalchemy import text

from src.db.connection import get_connection


def get_movie_by_id(movie_id):
    query = text(
        "SELECT movie_id, title, year, genres "
        "FROM movies WHERE movie_id = :movie_id"
    )
    with get_connection().connect() as connection:
        row = connection.execute(query, {"movie_id": int(movie_id)}).mappings().first()
    return dict(row) if row else None


def get_all_movies():
    query = text("SELECT movie_id, title, year, genres FROM movies")
    with get_connection().connect() as connection:
        return pd.read_sql_query(query, connection)


def get_ratings_by_user(user_id):
    query = text("SELECT movie_id, rating FROM ratings WHERE user_id = :user_id")
    with get_connection().connect() as connection:
        return pd.read_sql_query(query, connection, params={"user_id": int(user_id)})


def insert_ratings_batch(df, batch_size=1000):
    required_columns = ["user_id", "movie_id", "rating", "timestamp"]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Ratings batch is missing required columns: {', '.join(missing_columns)}"
        )

    if df.empty:
        return 0

    rows = df[required_columns].copy()
    rows["user_id"] = rows["user_id"].astype(int)
    rows["movie_id"] = rows["movie_id"].astype(int)
    rows["rating"] = rows["rating"].astype(float)
    rows["timestamp"] = rows["timestamp"].astype(int)
    payload = rows.to_dict("records")
    statement = text(
        "INSERT INTO ratings (user_id, movie_id, rating, [timestamp]) "
        "VALUES (:user_id, :movie_id, :rating, :timestamp)"
    )

    with get_connection().begin() as connection:
        for start in range(0, len(payload), batch_size):
            connection.execute(statement, payload[start:start + batch_size])

    return len(payload)
