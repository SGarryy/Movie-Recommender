import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


def build_tfidf_matrix(df):
    df = df.copy()
    tags_grouped = df.get('tags', pd.Series([''] * len(df)))
    df['content'] = (
        df['genres'].str.replace('|', ' ', regex=False) + ' ' +
        df['title']
    )
    df['content'] = df['content'].fillna('').str.lower()
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2
    )
    matrix = vectorizer.fit_transform(df['content'])
    return matrix, vectorizer


def build_user_item_matrix(df):
    user_ids = df['user_id'].astype('category')
    movie_ids = df['movie_id'].astype('category')
    matrix = csr_matrix(
        (df['rating'].values, (user_ids.cat.codes, movie_ids.cat.codes)),
        shape=(user_ids.cat.categories.size, movie_ids.cat.categories.size)
    )
    return matrix, user_ids.cat.categories, movie_ids.cat.categories