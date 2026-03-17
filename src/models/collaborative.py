from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from src.preprocessing.cleaner import (
    clean_movies,
    clean_ratings,
    load_movies_prefer_processed,
    load_ratings_prefer_processed,
)
from src.preprocessing.feature_engineering import build_user_item_matrix

COLLABORATIVE_COLUMNS = ["movie_id", "title", "genres", "year", "predicted_rating"]


def _empty_recommendations():
    return pd.DataFrame(columns=COLLABORATIVE_COLUMNS)


def train_svd(matrix, n_components=50):
    min_dimension = min(matrix.shape)
    if min_dimension < 2:
        raise ValueError("Need at least two users and two movies to train SVD.")

    bounded_components = min(n_components, min_dimension - 1)
    svd = TruncatedSVD(n_components=bounded_components, random_state=42)
    svd.fit(matrix)
    return svd


@lru_cache(maxsize=1)
def _get_trained_artifacts():
    from src.preprocessing.artifacts import (
        has_collaborative_artifacts,
        load_collaborative_artifacts,
    )

    if has_collaborative_artifacts():
        return load_collaborative_artifacts()

    ratings_df = clean_ratings(load_ratings_prefer_processed())
    matrix, user_index, movie_index = build_user_item_matrix(ratings_df)
    model = train_svd(matrix)
    return model, matrix, user_index, movie_index


def clear_collaborative_cache():
    _get_trained_artifacts.cache_clear()


def get_trained_artifacts():
    return _get_trained_artifacts()


def get_user_recommendations(user_id, model, matrix, user_index, movie_index, top_n=10):
    if len(user_index) == 0 or user_id not in user_index.values:
        return _empty_recommendations()

    user_idx = list(user_index).index(user_id)
    user_vector = matrix[user_idx]
    user_transformed = model.transform(user_vector)
    predicted_ratings = np.clip(model.inverse_transform(user_transformed).flatten(), 0.0, 5.0)

    rated_movies = matrix[user_idx].nonzero()[1]
    predicted_ratings[rated_movies] = 0

    ranked_indices = np.argsort(predicted_ratings)[::-1]
    ranked_indices = [idx for idx in ranked_indices if predicted_ratings[idx] > 0][:top_n]
    if not ranked_indices:
        return _empty_recommendations()

    movies_df = clean_movies(load_movies_prefer_processed())
    recommendation_scores = pd.DataFrame(
        {
            "movie_id": movie_index[ranked_indices].astype(int),
            "predicted_rating": predicted_ratings[ranked_indices],
        }
    )
    results = recommendation_scores.merge(
        movies_df[["movie_id", "title", "genres", "year"]],
        on="movie_id",
        how="left",
    )
    results = results.dropna(subset=["title"]).sort_values(
        "predicted_rating", ascending=False
    )
    return results[COLLABORATIVE_COLUMNS].reset_index(drop=True)
