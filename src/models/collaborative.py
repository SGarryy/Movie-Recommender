import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from src.preprocessing.cleaner import load_ratings, clean_ratings
from src.preprocessing.feature_engineering import build_user_item_matrix


def train_svd(matrix, n_components=50):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(matrix)
    return svd


def get_user_recommendations(user_id, model, matrix, user_index, movie_index, top_n=10):
    if user_id not in user_index.values:
        return []

    user_idx = list(user_index).index(user_id)
    user_vector = matrix[user_idx]
    user_transformed = model.transform(user_vector)
    predicted_ratings = model.inverse_transform(user_transformed).flatten()

    rated_movies = matrix[user_idx].nonzero()[1]
    predicted_ratings[rated_movies] = 0

    top_indices = np.argsort(predicted_ratings)[::-1][:top_n]

    ratings_df = load_ratings()
    ratings_df = clean_ratings(ratings_df)
    movies_df = pd.read_sql if False else None

    from src.preprocessing.cleaner import load_movies, clean_movies
    movies_df = clean_movies(load_movies())

    recommended_movie_ids = movie_index[top_indices]
    results = movies_df[movies_df['movie_id'].isin(recommended_movie_ids)][['movie_id', 'title', 'genres', 'year']]
    results = results.copy()
    results['predicted_rating'] = predicted_ratings[top_indices[:len(results)]]

    return results.reset_index(drop=True)