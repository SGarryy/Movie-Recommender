import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing.cleaner import load_movies, clean_movies
from src.preprocessing.feature_engineering import build_tfidf_matrix


def _get_movies_and_matrix():
    df = clean_movies(load_movies())
    matrix, vectorizer = build_tfidf_matrix(df)
    return df.reset_index(drop=True), matrix


def get_similar_movies(movie_id, top_n=10):
    df, matrix = _get_movies_and_matrix()

    if movie_id not in df['movie_id'].values:
        return []

    idx = df[df['movie_id'] == movie_id].index[0]
    movie_vector = matrix[idx]

    if movie_vector.nnz == 0:
        return []

    similarity_scores = cosine_similarity(movie_vector, matrix).flatten()
    similarity_scores[idx] = 0

    top_indices = np.argsort(similarity_scores)[::-1][:top_n]
    top_indices = [i for i in top_indices if similarity_scores[i] > 0]

    results = df.iloc[top_indices][['movie_id', 'title', 'genres', 'year']].copy()
    results['similarity_score'] = similarity_scores[top_indices]

    return results.reset_index(drop=True)