from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing.cleaner import clean_movies, load_movies_prefer_processed
from src.preprocessing.feature_engineering import build_tfidf_matrix

CONTENT_COLUMNS = ["movie_id", "title", "genres", "year", "similarity_score"]


def _empty_results():
    return pd.DataFrame(columns=CONTENT_COLUMNS)


@lru_cache(maxsize=1)
def _get_movies_and_matrix():
    from src.preprocessing.artifacts import has_content_artifacts, load_content_artifacts

    if has_content_artifacts():
        movies_df, matrix = load_content_artifacts()
        return movies_df.reset_index(drop=True), matrix

    movies_df = clean_movies(load_movies_prefer_processed())
    matrix, _ = build_tfidf_matrix(movies_df)
    return movies_df.reset_index(drop=True), matrix


def clear_content_cache():
    _get_movies_and_matrix.cache_clear()


def get_similar_movies(movie_id, top_n=10):
    movies_df, matrix = _get_movies_and_matrix()

    if movies_df.empty or movie_id not in movies_df["movie_id"].values:
        return _empty_results()

    movie_idx = movies_df[movies_df["movie_id"] == movie_id].index[0]
    movie_vector = matrix[movie_idx]

    if movie_vector.nnz == 0:
        return _empty_results()

    similarity_scores = cosine_similarity(movie_vector, matrix).flatten()
    similarity_scores[movie_idx] = 0

    top_indices = np.argsort(similarity_scores)[::-1][:top_n]
    top_indices = [index for index in top_indices if similarity_scores[index] > 0]
    if not top_indices:
        return _empty_results()

    results = movies_df.iloc[top_indices][["movie_id", "title", "genres", "year"]].copy()
    results["similarity_score"] = similarity_scores[top_indices]
    return results.reset_index(drop=True)
