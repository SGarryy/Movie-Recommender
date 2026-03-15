import pandas as pd
from src.db.connection import get_connection


def get_movie_by_id(movie_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM movies WHERE movie_id = ?", (movie_id,))
    row = cursor.fetchone()
    conn.close()
    return row


def get_all_movies():
    conn = get_connection()
    query = "SELECT movie_id, title, year, genres FROM movies"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def get_ratings_by_user(user_id):
    conn = get_connection()
    query = "SELECT movie_id, rating FROM ratings WHERE user_id = ?"
    df = pd.read_sql(query, conn, params=(user_id,))
    conn.close()
    return df


def insert_ratings_batch(df):
    conn = get_connection()
    cursor = conn.cursor()
    batch_size = 1000
    rows = list(df.itertuples(index=False, name=None))
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        cursor.executemany(
            "INSERT INTO ratings (user_id, movie_id, rating, timestamp) VALUES (?, ?, ?, ?)",
            batch
        )
        conn.commit()
    conn.close()