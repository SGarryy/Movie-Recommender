from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.db.connection import get_connection


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ml-10M100K"
TMDB_DIR = PROJECT_ROOT / "data" / "raw" / "tmdb"


def _load_dat_file(filename, columns):
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Expected dataset file at {path}")

    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=columns,
        encoding="latin-1",
    )


def _insert_dataframe(df, statement, columns, batch_size=5000):
    if df.empty:
        return 0

    rows = df[columns].to_dict("records")
    with get_connection().begin() as connection:
        for start in range(0, len(rows), batch_size):
            connection.execute(text(statement), rows[start:start + batch_size])
    return len(rows)


def load_movies():
    df = _load_dat_file("movies.dat", ["movie_id", "title", "genres"])
    df["year"] = df["title"].str.extract(r"\((\d{4})\)").astype("Int64")
    df["title"] = df["title"].str.replace(r"\s*\(\d{4}\)", "", regex=True).str.strip()
    return df


def load_users():
    columns = ["user_id", "gender", "age", "occupation", "zip_code"]
    path = DATA_DIR / "users.dat"

    if not path.exists():
        return pd.DataFrame(columns=columns)

    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=columns,
        encoding="latin-1",
    )


def load_ratings():
    return _load_dat_file("ratings.dat", ["user_id", "movie_id", "rating", "timestamp"])


def load_tags():
    return _load_dat_file("tags.dat", ["user_id", "movie_id", "tag", "timestamp"])


def insert_movies(df):
    records = df.copy()
    records["movie_id"] = records["movie_id"].astype(int)
    records["year"] = records["year"].apply(lambda value: int(value) if pd.notna(value) else None)
    records["title"] = records["title"].fillna("").astype(str)
    records["genres"] = records["genres"].fillna("").astype(str)

    inserted = _insert_dataframe(
        records,
        "INSERT INTO movies (movie_id, title, year, genres) "
        "VALUES (:movie_id, :title, :year, :genres)",
        ["movie_id", "title", "year", "genres"],
    )
    print(f"Inserted {inserted} movies")
    return inserted


def insert_users(df):
    if df.empty:
        print("No users file found; skipping users load.")
        return 0

    records = df.copy()
    records["user_id"] = records["user_id"].astype(int)
    records["age"] = records["age"].astype(int)
    records["occupation"] = records["occupation"].astype(int)
    records["gender"] = records["gender"].fillna("").astype(str)
    records["zip_code"] = records["zip_code"].fillna("").astype(str)

    inserted = _insert_dataframe(
        records,
        "INSERT INTO users (user_id, gender, age, occupation, zip_code) "
        "VALUES (:user_id, :gender, :age, :occupation, :zip_code)",
        ["user_id", "gender", "age", "occupation", "zip_code"],
    )
    print(f"Inserted {inserted} users")
    return inserted


def insert_ratings(df):
    records = df.copy()
    records["user_id"] = records["user_id"].astype(int)
    records["movie_id"] = records["movie_id"].astype(int)
    records["rating"] = records["rating"].astype(float)
    records["timestamp"] = records["timestamp"].astype(int)

    inserted = _insert_dataframe(
        records,
        "INSERT INTO ratings (user_id, movie_id, rating, [timestamp]) "
        "VALUES (:user_id, :movie_id, :rating, :timestamp)",
        ["user_id", "movie_id", "rating", "timestamp"],
    )
    print(f"Inserted {inserted} ratings")
    return inserted


def insert_tags(df):
    records = df.copy()
    records["user_id"] = records["user_id"].astype(int)
    records["movie_id"] = records["movie_id"].astype(int)
    records["tag"] = records["tag"].fillna("").astype(str)
    records["timestamp"] = records["timestamp"].astype(int)

    inserted = _insert_dataframe(
        records,
        "INSERT INTO tags (user_id, movie_id, tag, [timestamp]) "
        "VALUES (:user_id, :movie_id, :tag, :timestamp)",
        ["user_id", "movie_id", "tag", "timestamp"],
    )
    print(f"Inserted {inserted} tags")
    return inserted


if __name__ == "__main__":
    print("Loading movies...")
    insert_movies(load_movies())

    users = load_users()
    if not users.empty:
        print("Loading users...")
        insert_users(users)
    else:
        print("users.dat not found in MovieLens 10M; skipping user profile import.")

    print("Loading ratings...")
    insert_ratings(load_ratings())

    print("Loading tags...")
    insert_tags(load_tags())

    print("Done.")
