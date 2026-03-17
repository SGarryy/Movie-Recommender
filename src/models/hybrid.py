import pandas as pd

from src.models.collaborative import get_trained_artifacts, get_user_recommendations
from src.models.content_based import get_similar_movies

HYBRID_COLUMNS = ["movie_id", "title", "genres", "year", "score", "source"]


def _normalize_scores(df):
    if df.empty:
        return df

    normalized = df.copy()
    max_score = normalized["score"].max()
    if pd.isna(max_score) or max_score <= 0:
        normalized["score"] = 0.0
    else:
        normalized["score"] = normalized["score"] / max_score
    return normalized


def hybrid_recommend(movie_id=None, user_id=None, top_n=10):
    content_results = pd.DataFrame(columns=HYBRID_COLUMNS)
    collab_results = pd.DataFrame(columns=HYBRID_COLUMNS)

    if movie_id is not None:
        content_results = get_similar_movies(movie_id, top_n=top_n)

    if user_id is not None:
        try:
            model, matrix, user_index, movie_index = get_trained_artifacts()
            collab_results = get_user_recommendations(
                user_id, model, matrix, user_index, movie_index, top_n=top_n
            )
        except ValueError:
            collab_results = pd.DataFrame(
                columns=["movie_id", "title", "genres", "year", "predicted_rating"]
            )

    if content_results.empty and collab_results.empty:
        return pd.DataFrame(columns=HYBRID_COLUMNS)

    if content_results.empty:
        return collab_results.rename(columns={"predicted_rating": "score"}).assign(
            source="collaborative"
        )

    if collab_results.empty:
        return content_results.rename(columns={"similarity_score": "score"}).assign(
            source="content"
        )

    content_results = content_results.rename(columns={"similarity_score": "score"})
    collab_results = collab_results.rename(columns={"predicted_rating": "score"})

    content_results = _normalize_scores(content_results)
    collab_results = _normalize_scores(collab_results)

    content_results["source"] = "content"
    collab_results["source"] = "collaborative"

    combined = pd.concat([content_results, collab_results], ignore_index=True)
    combined = (
        combined.sort_values("score", ascending=False)
        .groupby("movie_id", as_index=False)
        .agg(
            title=("title", "first"),
            genres=("genres", "first"),
            year=("year", "first"),
            score=("score", "max"),
            source=("source", lambda values: "+".join(sorted(set(values)))),
        )
        .sort_values("score", ascending=False)
        .head(top_n)
    )
    return combined[HYBRID_COLUMNS].reset_index(drop=True)
