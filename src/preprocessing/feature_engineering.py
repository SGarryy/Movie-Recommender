import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_matrix(df):
    if df.empty:
        raise ValueError("Movies dataframe is empty.")

    feature_df = df.copy()
    tags_series = feature_df.get("tags", pd.Series("", index=feature_df.index)).fillna("")
    genres = feature_df.get("genres", pd.Series("", index=feature_df.index)).fillna("")
    titles = feature_df.get("title", pd.Series("", index=feature_df.index)).fillna("")

    feature_df["content"] = (
        genres.astype(str).str.replace("|", " ", regex=False)
        + " "
        + titles.astype(str)
        + " "
        + tags_series.astype(str)
    ).str.strip().str.lower()

    min_df = 2 if len(feature_df) > 1 else 1
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=min_df)
    matrix = vectorizer.fit_transform(feature_df["content"])
    return matrix, vectorizer


def build_user_item_matrix(df):
    if df.empty:
        raise ValueError("Ratings dataframe is empty.")

    ratings = df.copy()
    ratings["user_id"] = pd.to_numeric(ratings["user_id"], errors="coerce")
    ratings["movie_id"] = pd.to_numeric(ratings["movie_id"], errors="coerce")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["user_id", "movie_id", "rating"])

    if ratings.empty:
        raise ValueError("Ratings dataframe does not contain valid rows.")

    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["movie_id"] = ratings["movie_id"].astype(int)
    user_ids = ratings["user_id"].astype("category")
    movie_ids = ratings["movie_id"].astype("category")

    matrix = csr_matrix(
        (ratings["rating"].values, (user_ids.cat.codes, movie_ids.cat.codes)),
        shape=(user_ids.cat.categories.size, movie_ids.cat.categories.size),
    )
    return matrix, user_ids.cat.categories, movie_ids.cat.categories
