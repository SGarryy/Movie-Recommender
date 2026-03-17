import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from src.preprocessing.cleaner import load_movies, clean_movies, load_ratings, clean_ratings
from src.preprocessing.feature_engineering import build_tfidf_matrix, build_user_item_matrix
from src.models.content_based import get_similar_movies
from src.recommender import search_movie, get_top_rated


def test_load_movies():
    df = load_movies()
    assert not df.empty
    assert 'movie_id' in df.columns
    assert 'title' in df.columns
    assert 'genres' in df.columns


def test_clean_movies():
    df = clean_movies(load_movies())
    assert df['title'].isnull().sum() == 0
    assert df['genres'].isnull().sum() == 0
    assert df['movie_id'].duplicated().sum() == 0


def test_clean_ratings():
    df = clean_ratings(load_ratings())
    assert df['rating'].min() >= 0.5
    assert df['rating'].max() <= 5.0
    assert df[['user_id', 'movie_id']].duplicated().sum() == 0


def test_build_tfidf_matrix():
    df = clean_movies(load_movies())
    matrix, vectorizer = build_tfidf_matrix(df)
    assert matrix.shape[0] == len(df)
    assert matrix.shape[1] > 0


def test_build_user_item_matrix():
    df = clean_ratings(load_ratings())
    matrix, user_index, movie_index = build_user_item_matrix(df)
    assert matrix.shape[0] == len(user_index)
    assert matrix.shape[1] == len(movie_index)


def test_get_similar_movies():
    results = get_similar_movies(1, top_n=5)
    assert len(results) > 0
    assert 'movie_id' in results.columns
    assert 'title' in results.columns
    assert 1 not in results['movie_id'].values


def test_search_movie():
    results = search_movie('Toy Story')
    assert not results.empty
    assert 'title' in results.columns


def test_get_top_rated():
    results = get_top_rated(10)
    assert not results.empty
    assert 'avg_rating' in results.columns
    assert results['avg_rating'].max() <= 5.0


def test_search_movie_no_results():
    results = search_movie('xyzabcnonexistentmovie123')
    assert results.empty


def test_similar_movies_invalid_id():
    results = get_similar_movies(999999, top_n=5)
    assert results == [] or (hasattr(results, 'empty') and results.empty)