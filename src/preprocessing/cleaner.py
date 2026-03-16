import pandas as pd
from src.db.connection import get_connection


def load_movies():
    engine = get_connection()
    df = pd.read_sql("SELECT movie_id, title, year, genres FROM movies", engine)
    return df


def load_ratings():
    engine = get_connection()
    df = pd.read_sql("SELECT user_id, movie_id, rating FROM ratings", engine)
    return df


def load_tags():
    engine = get_connection()
    df = pd.read_sql("SELECT movie_id, tag FROM tags", engine)
    return df


def clean_movies(df):
    df = df.copy()
    df['genres'] = df['genres'].fillna('')
    df['title'] = df['title'].fillna('Unknown')
    df['year'] = df['year'].fillna(0).astype(int)
    df = df.drop_duplicates(subset='movie_id')
    return df


def clean_ratings(df):
    df = df.copy()
    df = df.dropna(subset=['user_id', 'movie_id', 'rating'])
    df['rating'] = df['rating'].clip(0.5, 5.0)
    df = df.drop_duplicates(subset=['user_id', 'movie_id'])
    return df