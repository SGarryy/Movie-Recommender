from pathlib import Path
import pickle

import pandas as pd
from scipy.sparse import load_npz, save_npz

from src.db.load_data import load_movies as load_raw_movies
from src.db.load_data import load_ratings as load_raw_ratings
from src.db.load_data import load_tags as load_raw_tags
from src.models.collaborative import train_svd
from src.preprocessing.cleaner import clean_movies, clean_ratings
from src.preprocessing.feature_engineering import build_tfidf_matrix, build_user_item_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MOVIES_CLEAN_PATH = PROCESSED_DIR / "movies_clean.pkl"
RATINGS_CLEAN_PATH = PROCESSED_DIR / "ratings_clean.pkl"
CONTENT_MOVIES_PATH = PROCESSED_DIR / "content_movies.pkl"
CONTENT_MATRIX_PATH = PROCESSED_DIR / "content_tfidf_matrix.npz"
COLLAB_MATRIX_PATH = PROCESSED_DIR / "collaborative_user_item_matrix.npz"
COLLAB_INDEX_PATH = PROCESSED_DIR / "collaborative_indexes.pkl"
COLLAB_MODEL_PATH = PROCESSED_DIR / "collaborative_svd.pkl"


def ensure_processed_dir():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _save_pickle(path: Path, value) -> None:
    with path.open("wb") as file_handle:
        pickle.dump(value, file_handle)


def _load_pickle(path: Path):
    with path.open("rb") as file_handle:
        return pickle.load(file_handle)


def _build_content_movies(movies_df: pd.DataFrame, tags_df: pd.DataFrame) -> pd.DataFrame:
    if tags_df.empty:
        tags_grouped = pd.DataFrame(columns=["movie_id", "tags"])
    else:
        tags = tags_df.copy()
        tags["movie_id"] = pd.to_numeric(tags["movie_id"], errors="coerce")
        tags["tag"] = tags["tag"].fillna("").astype(str).str.strip()
        tags = tags.dropna(subset=["movie_id"])
        tags["movie_id"] = tags["movie_id"].astype(int)
        tags_grouped = (
            tags.groupby("movie_id")["tag"]
            .apply(lambda values: " ".join(sorted({value for value in values if value})))
            .reset_index(name="tags")
        )

    return movies_df.merge(tags_grouped, on="movie_id", how="left").fillna({"tags": ""})


def build_processed_artifacts():
    ensure_processed_dir()

    movies_df = clean_movies(load_raw_movies())
    ratings_df = clean_ratings(load_raw_ratings())
    tags_df = load_raw_tags()
    content_movies_df = _build_content_movies(movies_df, tags_df)

    movies_df.to_pickle(MOVIES_CLEAN_PATH)
    ratings_df.to_pickle(RATINGS_CLEAN_PATH)
    content_movies_df.to_pickle(CONTENT_MOVIES_PATH)

    content_matrix, _ = build_tfidf_matrix(content_movies_df)
    save_npz(CONTENT_MATRIX_PATH, content_matrix)

    collab_matrix, user_index, movie_index = build_user_item_matrix(ratings_df)
    save_npz(COLLAB_MATRIX_PATH, collab_matrix)
    _save_pickle(COLLAB_INDEX_PATH, {"user_index": user_index, "movie_index": movie_index})

    collab_model = train_svd(collab_matrix)
    _save_pickle(COLLAB_MODEL_PATH, collab_model)

    return {
        "movies_rows": len(movies_df),
        "ratings_rows": len(ratings_df),
        "content_movies_rows": len(content_movies_df),
        "content_matrix_shape": content_matrix.shape,
        "collaborative_matrix_shape": collab_matrix.shape,
    }


def has_processed_movies() -> bool:
    return MOVIES_CLEAN_PATH.exists()


def has_processed_ratings() -> bool:
    return RATINGS_CLEAN_PATH.exists()


def has_content_artifacts() -> bool:
    return CONTENT_MOVIES_PATH.exists() and CONTENT_MATRIX_PATH.exists()


def has_collaborative_artifacts() -> bool:
    return (
        COLLAB_MATRIX_PATH.exists()
        and COLLAB_INDEX_PATH.exists()
        and COLLAB_MODEL_PATH.exists()
    )


def load_processed_movies():
    return pd.read_pickle(MOVIES_CLEAN_PATH)


def load_processed_ratings():
    return pd.read_pickle(RATINGS_CLEAN_PATH)


def load_content_artifacts():
    movies_df = pd.read_pickle(CONTENT_MOVIES_PATH)
    matrix = load_npz(CONTENT_MATRIX_PATH)
    return movies_df, matrix


def load_collaborative_artifacts():
    matrix = load_npz(COLLAB_MATRIX_PATH)
    indexes = _load_pickle(COLLAB_INDEX_PATH)
    model = _load_pickle(COLLAB_MODEL_PATH)
    return model, matrix, indexes["user_index"], indexes["movie_index"]
