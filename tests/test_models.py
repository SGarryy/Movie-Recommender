import os
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.models.collaborative as collaborative
import src.models.content_based as content_based
import src.models.hybrid as hybrid
import src.preprocessing.artifacts as artifacts
import src.recommender as recommender
from src.db.connection import clear_connection_cache
from src.models.collaborative import get_user_recommendations, train_svd
from src.models.content_based import clear_content_cache, get_similar_movies
from src.preprocessing.cleaner import clean_movies, clean_ratings
from src.preprocessing.feature_engineering import build_tfidf_matrix, build_user_item_matrix
from src.recommender import get_top_rated, search_movie


@pytest.fixture(autouse=True)
def clear_caches_between_tests():
    clear_connection_cache()
    clear_content_cache()
    collaborative.clear_collaborative_cache()
    yield
    clear_content_cache()
    collaborative.clear_collaborative_cache()


@pytest.fixture
def workspace_tmp_dir():
    base_dir = Path(".tmp") / f"test-{uuid.uuid4().hex}"
    base_dir.mkdir(parents=True, exist_ok=True)
    yield base_dir


@pytest.fixture
def sample_movies():
    return pd.DataFrame(
        [
            {
                "movie_id": 1,
                "title": "Toy Story",
                "year": 1995,
                "genres": "Animation|Children|Comedy",
            },
            {
                "movie_id": 2,
                "title": "Toy Story 2",
                "year": 1999,
                "genres": "Animation|Children|Comedy",
            },
            {
                "movie_id": 3,
                "title": "Die Hard",
                "year": 1988,
                "genres": "Action|Thriller",
            },
            {
                "movie_id": 4,
                "title": "A Bug's Life",
                "year": 1998,
                "genres": "Animation|Children|Comedy",
            },
        ]
    )


@pytest.fixture
def sample_ratings():
    return pd.DataFrame(
        [
            {"user_id": 1, "movie_id": 1, "rating": 5.0},
            {"user_id": 1, "movie_id": 3, "rating": 2.0},
            {"user_id": 2, "movie_id": 1, "rating": 4.5},
            {"user_id": 2, "movie_id": 2, "rating": 4.0},
            {"user_id": 3, "movie_id": 2, "rating": 4.5},
            {"user_id": 3, "movie_id": 4, "rating": 5.0},
        ]
    )


def test_clean_movies():
    df = pd.DataFrame(
        [
            {"movie_id": 1, "title": None, "year": None, "genres": None},
            {"movie_id": 1, "title": "Duplicate", "year": 2000, "genres": "Drama"},
            {"movie_id": 2, "title": "Heat", "year": "1995", "genres": "Crime|Drama"},
        ]
    )

    cleaned = clean_movies(df)

    assert cleaned["title"].isnull().sum() == 0
    assert cleaned["genres"].isnull().sum() == 0
    assert cleaned["movie_id"].duplicated().sum() == 0
    assert cleaned.loc[0, "title"] == "Unknown"
    assert cleaned.loc[0, "year"] == 0


def test_clean_ratings():
    df = pd.DataFrame(
        [
            {"user_id": 1, "movie_id": 1, "rating": 6.0},
            {"user_id": 1, "movie_id": 1, "rating": 4.0},
            {"user_id": 2, "movie_id": 2, "rating": 0.0},
            {"user_id": None, "movie_id": 3, "rating": 4.0},
        ]
    )

    cleaned = clean_ratings(df)

    assert cleaned["rating"].min() >= 0.5
    assert cleaned["rating"].max() <= 5.0
    assert cleaned[["user_id", "movie_id"]].duplicated().sum() == 0
    assert len(cleaned) == 2


def test_build_tfidf_matrix(sample_movies):
    matrix, vectorizer = build_tfidf_matrix(clean_movies(sample_movies))
    assert matrix.shape[0] == len(sample_movies)
    assert matrix.shape[1] > 0
    assert "toy" in vectorizer.vocabulary_


def test_build_user_item_matrix(sample_ratings):
    matrix, user_index, movie_index = build_user_item_matrix(clean_ratings(sample_ratings))
    assert matrix.shape[0] == len(user_index)
    assert matrix.shape[1] == len(movie_index)


def test_get_similar_movies(monkeypatch, sample_movies):
    monkeypatch.setattr(content_based, "load_movies_prefer_processed", lambda: sample_movies)
    monkeypatch.setattr(artifacts, "has_content_artifacts", lambda: False)

    results = get_similar_movies(1, top_n=5)

    assert not results.empty
    assert "movie_id" in results.columns
    assert "title" in results.columns
    assert 1 not in results["movie_id"].values
    assert results.iloc[0]["movie_id"] in {2, 4}


def test_search_movie(monkeypatch, sample_movies):
    monkeypatch.setattr(recommender, "load_movies_prefer_processed", lambda: sample_movies)

    results = search_movie("Toy Story")

    assert not results.empty
    assert "title" in results.columns


def test_search_movie_treats_query_as_literal_text(monkeypatch, sample_movies):
    monkeypatch.setattr(recommender, "load_movies_prefer_processed", lambda: sample_movies)

    results = search_movie("Toy Story (")

    assert results.empty


def test_get_top_rated(monkeypatch, sample_movies):
    ratings = pd.DataFrame(
        [
            {"user_id": 1, "movie_id": 1, "rating": 4.5},
            {"user_id": 2, "movie_id": 1, "rating": 4.0},
            {"user_id": 3, "movie_id": 2, "rating": 5.0},
            {"user_id": 4, "movie_id": 2, "rating": 4.5},
            {"user_id": 5, "movie_id": 3, "rating": 3.0},
        ]
    )

    monkeypatch.setattr(recommender, "load_movies_prefer_processed", lambda: sample_movies)
    monkeypatch.setattr(recommender, "load_ratings_prefer_processed", lambda: ratings)

    results = get_top_rated(10, min_ratings=2)

    assert not results.empty
    assert "avg_rating" in results.columns
    assert results["avg_rating"].max() <= 5.0
    assert set(results["movie_id"]) == {1, 2}
    assert results.iloc[0]["movie_id"] == 2


def test_train_svd_caps_components_for_small_matrices():
    matrix = csr_matrix([[5.0, 0.0], [0.0, 4.0], [3.0, 1.0]])
    model = train_svd(matrix, n_components=50)
    assert model.components_.shape[0] == 1


def test_get_user_recommendations_preserves_rank_order(monkeypatch):
    class DummyModel:
        def transform(self, user_vector):
            return np.array([[1.0]])

        def inverse_transform(self, user_vector):
            return np.array([[4.9, 2.0, 4.6]])

    matrix = csr_matrix([[5.0, 0.0, 0.0]])
    user_index = pd.Index([42])
    movie_index = pd.Index([10, 20, 30])
    movies = pd.DataFrame(
        [
            {"movie_id": 30, "title": "Third", "year": 2003, "genres": "Drama"},
            {"movie_id": 20, "title": "Second", "year": 2002, "genres": "Comedy"},
            {"movie_id": 10, "title": "First", "year": 2001, "genres": "Action"},
        ]
    )

    monkeypatch.setattr(collaborative, "load_movies_prefer_processed", lambda: movies)

    results = get_user_recommendations(42, DummyModel(), matrix, user_index, movie_index, top_n=2)

    assert results["movie_id"].tolist() == [30, 20]
    assert results["predicted_rating"].tolist() == [4.6, 2.0]


def test_hybrid_recommend_merges_duplicate_movies(monkeypatch):
    content_results = pd.DataFrame(
        [
            {
                "movie_id": 1,
                "title": "Toy Story",
                "genres": "Animation",
                "year": 1995,
                "similarity_score": 0.9,
            },
            {
                "movie_id": 2,
                "title": "Toy Story 2",
                "genres": "Animation",
                "year": 1999,
                "similarity_score": 0.3,
            },
        ]
    )
    collaborative_results = pd.DataFrame(
        [
            {
                "movie_id": 2,
                "title": "Toy Story 2",
                "genres": "Animation",
                "year": 1999,
                "predicted_rating": 4.8,
            },
            {
                "movie_id": 3,
                "title": "Die Hard",
                "genres": "Action",
                "year": 1988,
                "predicted_rating": 4.2,
            },
        ]
    )

    monkeypatch.setattr(hybrid, "get_similar_movies", lambda movie_id, top_n=10: content_results.copy())
    monkeypatch.setattr(
        hybrid,
        "get_trained_artifacts",
        lambda: ("model", "matrix", pd.Index([7]), pd.Index([1, 2, 3])),
    )
    monkeypatch.setattr(
        hybrid,
        "get_user_recommendations",
        lambda user_id, model, matrix, user_index, movie_index, top_n=10: collaborative_results.copy(),
    )

    results = hybrid.hybrid_recommend(movie_id=1, user_id=7, top_n=5)

    assert results["movie_id"].tolist() == [1, 2, 3]
    merged_row = results[results["movie_id"] == 2].iloc[0]
    assert merged_row["source"] == "collaborative+content"


def test_similar_movies_invalid_id(monkeypatch, sample_movies):
    monkeypatch.setattr(content_based, "load_movies_prefer_processed", lambda: sample_movies)
    monkeypatch.setattr(artifacts, "has_content_artifacts", lambda: False)

    results = get_similar_movies(999999, top_n=5)

    assert results.empty


def test_build_processed_artifacts_and_prefer_processed(
    monkeypatch, workspace_tmp_dir, sample_movies, sample_ratings
):
    tags = pd.DataFrame(
        [
            {"movie_id": 1, "tag": "pixar"},
            {"movie_id": 1, "tag": "toys"},
            {"movie_id": 2, "tag": "sequel"},
        ]
    )

    monkeypatch.setattr(artifacts, "PROCESSED_DIR", workspace_tmp_dir)
    monkeypatch.setattr(artifacts, "MOVIES_CLEAN_PATH", workspace_tmp_dir / "movies_clean.pkl")
    monkeypatch.setattr(artifacts, "RATINGS_CLEAN_PATH", workspace_tmp_dir / "ratings_clean.pkl")
    monkeypatch.setattr(artifacts, "CONTENT_MOVIES_PATH", workspace_tmp_dir / "content_movies.pkl")
    monkeypatch.setattr(
        artifacts, "CONTENT_MATRIX_PATH", workspace_tmp_dir / "content_tfidf_matrix.npz"
    )
    monkeypatch.setattr(
        artifacts, "COLLAB_MATRIX_PATH", workspace_tmp_dir / "collaborative_user_item_matrix.npz"
    )
    monkeypatch.setattr(
        artifacts, "COLLAB_INDEX_PATH", workspace_tmp_dir / "collaborative_indexes.pkl"
    )
    monkeypatch.setattr(
        artifacts, "COLLAB_MODEL_PATH", workspace_tmp_dir / "collaborative_svd.pkl"
    )

    monkeypatch.setattr(artifacts, "load_raw_movies", lambda: sample_movies.copy())
    monkeypatch.setattr(artifacts, "load_raw_ratings", lambda: sample_ratings.copy())
    monkeypatch.setattr(artifacts, "load_raw_tags", lambda: tags.copy())

    summary = artifacts.build_processed_artifacts()

    assert summary["movies_rows"] == len(sample_movies)
    assert artifacts.has_processed_movies()
    assert artifacts.has_processed_ratings()
    assert artifacts.has_content_artifacts()
    assert artifacts.has_collaborative_artifacts()

    loaded_movies = artifacts.load_processed_movies()
    loaded_ratings = artifacts.load_processed_ratings()
    content_movies, content_matrix = artifacts.load_content_artifacts()
    model, matrix, user_index, movie_index = artifacts.load_collaborative_artifacts()

    assert not loaded_movies.empty
    assert not loaded_ratings.empty
    assert "tags" in content_movies.columns
    assert content_matrix.shape[0] == len(sample_movies)
    assert matrix.shape[0] == len(user_index)
    assert matrix.shape[1] == len(movie_index)
    assert hasattr(model, "inverse_transform")
