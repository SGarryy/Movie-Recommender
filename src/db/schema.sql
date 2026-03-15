USE MovieRecommender;
GO

CREATE TABLE users (
    user_id INT PRIMARY KEY,
    gender CHAR(1),
    age INT,
    occupation INT,
    zip_code VARCHAR(10)
);
GO

CREATE TABLE movies (
    movie_id INT PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    year INT,
    genres VARCHAR(500)
);
GO

CREATE TABLE ratings (
    rating_id INT IDENTITY(1,1) PRIMARY KEY,
    user_id INT NOT NULL,
    movie_id INT NOT NULL,
    rating FLOAT NOT NULL,
    timestamp BIGINT,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
);
GO

CREATE TABLE tags (
    tag_id INT IDENTITY(1,1) PRIMARY KEY,
    user_id INT NOT NULL,
    movie_id INT NOT NULL,
    tag VARCHAR(500),
    timestamp BIGINT,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
);
GO

CREATE INDEX idx_ratings_user ON ratings(user_id);
CREATE INDEX idx_ratings_movie ON ratings(movie_id);
CREATE INDEX idx_tags_movie ON tags(movie_id);
CREATE INDEX idx_movies_year ON movies(year);
GO