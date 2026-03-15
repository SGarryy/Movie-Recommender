import os
import pandas as pd
from src.db.connection import get_connection


DATA_DIR = os.path.join("data", "raw", "ml-10M100K")
TMDB_DIR = os.path.join("data", "raw", "tmdb")


def load_movies():
    path = os.path.join(DATA_DIR, "movies.dat")
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1"
    )
    df["year"] = df["title"].str.extract(r"\((\d{4})\)").astype("Int64")
    df["title"] = df["title"].str.replace(r"\s*\(\d{4}\)", "", regex=True).str.strip()
    return df


def load_users():
    path = os.path.join(DATA_DIR, "users.dat")
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding="latin-1"
    )
    return df


def load_ratings():
    path = os.path.join(DATA_DIR, "ratings.dat")
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1"
    )
    return df


def load_tags():
    path = os.path.join(DATA_DIR, "tags.dat")
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "tag", "timestamp"],
        encoding="latin-1"
    )
    return df


def insert_movies(df):
    conn = get_connection()
    cursor = conn.cursor()
    df = df.copy()
    df["movie_id"] = df["movie_id"].astype(int)
    df["year"] = df["year"].apply(lambda x: int(x) if pd.notna(x) else None)
    df["title"] = df["title"].astype(str)
    df["genres"] = df["genres"].astype(str)
    rows = list(df[["movie_id", "title", "year", "genres"]].itertuples(index=False, name=None))
    batch_size = 1000
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        cursor.executemany(
            "INSERT INTO movies (movie_id, title, year, genres) VALUES (?, ?, ?, ?)",
            batch
        )
        conn.commit()
    conn.close()
    print(f"Inserted {len(df)} movies")


def insert_users(df):
    conn = get_connection()
    cursor = conn.cursor()
    rows = list(df.itertuples(index=False, name=None))
    batch_size = 1000
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        cursor.executemany(
            "INSERT INTO users (user_id, gender, age, occupation, zip_code) VALUES (?, ?, ?, ?, ?)",
            batch
        )
        conn.commit()
    conn.close()
    print(f"Inserted {len(df)} users")


def insert_ratings(df):
    conn = get_connection()
    cursor = conn.cursor()
    df = df.copy()
    df["user_id"] = df["user_id"].astype(int)
    df["movie_id"] = df["movie_id"].astype(int)
    df["rating"] = df["rating"].astype(float)
    df["timestamp"] = df["timestamp"].astype(int)
    rows = list(df.itertuples(index=False, name=None))
    batch_size = 1000
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        cursor.executemany(
            "INSERT INTO ratings (user_id, movie_id, rating, timestamp) VALUES (?, ?, ?, ?)",
            batch
        )
        conn.commit()
    conn.close()
    print(f"Inserted {len(df)} ratings")


def insert_tags(df):
    conn = get_connection()
    cursor = conn.cursor()
    df = df.copy()
    df["user_id"] = df["user_id"].astype(int)
    df["movie_id"] = df["movie_id"].astype(int)
    df["tag"] = df["tag"].astype(str)
    df["timestamp"] = df["timestamp"].astype(int)
    rows = list(df.itertuples(index=False, name=None))
    batch_size = 1000
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        cursor.executemany(
            "INSERT INTO tags (user_id, movie_id, tag, timestamp) VALUES (?, ?, ?, ?)",
            batch
        )
        conn.commit()
    conn.close()
    print(f"Inserted {len(df)} tags")


if __name__ == "__main__":
    print("Loading movies...")
    movies = load_movies()
    insert_movies(movies)

    print("Loading ratings...")
    ratings = load_ratings()
    insert_ratings(ratings)

    print("Loading tags...")
    tags = load_tags()
    insert_tags(tags)

    print("Done.")