import pandas as pd
from src.models.hybrid import hybrid_recommend
from src.models.content_based import get_similar_movies
from src.preprocessing.cleaner import load_movies, clean_movies


def recommend(movie_id=None, user_id=None, top_n=10):
    if movie_id is None and user_id is None:
        return pd.DataFrame()

    results = hybrid_recommend(movie_id=movie_id, user_id=user_id, top_n=top_n)

    if isinstance(results, list) or results.empty:
        return pd.DataFrame()

    return results


def search_movie(title_query):
    movies_df = clean_movies(load_movies())
    mask = movies_df['title'].str.contains(title_query, case=False, na=False)
    results = movies_df[mask][['movie_id', 'title', 'year', 'genres']]
    return results.reset_index(drop=True)


def get_top_rated(top_n=10):
    from src.preprocessing.cleaner import load_ratings, clean_ratings
    ratings_df = clean_ratings(load_ratings())
    movies_df = clean_movies(load_movies())

    top = (
        ratings_df.groupby('movie_id')
        .agg(avg_rating=('rating', 'mean'), rating_count=('rating', 'count'))
        .reset_index()
    )
    top = top[top['rating_count'] >= 100]
    top = top.sort_values('avg_rating', ascending=False).head(top_n)
    top = top.merge(movies_df[['movie_id', 'title', 'genres', 'year']], on='movie_id')

    return top.reset_index(drop=True)