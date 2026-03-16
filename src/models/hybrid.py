import pandas as pd
from src.models.content_based import get_similar_movies
from src.models.collaborative import train_svd, get_user_recommendations
from src.preprocessing.cleaner import load_ratings, clean_ratings
from src.preprocessing.feature_engineering import build_user_item_matrix


def hybrid_recommend(movie_id=None, user_id=None, top_n=10):
    content_results = pd.DataFrame()
    collab_results = pd.DataFrame()

    if movie_id is not None:
        content_results = get_similar_movies(movie_id, top_n=top_n)

    if user_id is not None:
        ratings_df = clean_ratings(load_ratings())
        matrix, user_index, movie_index = build_user_item_matrix(ratings_df)
        model = train_svd(matrix)
        collab_results = get_user_recommendations(
            user_id, model, matrix, user_index, movie_index, top_n=top_n
        )

    if content_results.empty and collab_results.empty:
        return pd.DataFrame()

    if content_results.empty:
        return collab_results

    if collab_results.empty:
        return content_results

    content_results = content_results.rename(columns={'similarity_score': 'score'})
    collab_results = collab_results.rename(columns={'predicted_rating': 'score'})

    content_results['score'] = content_results['score'] / content_results['score'].max()
    collab_results['score'] = collab_results['score'] / collab_results['score'].max()

    content_results['source'] = 'content'
    collab_results['source'] = 'collaborative'

    combined = pd.concat([content_results, collab_results], ignore_index=True)
    combined = combined.drop_duplicates(subset='movie_id')
    combined = combined.sort_values('score', ascending=False).head(top_n)

    return combined.reset_index(drop=True)