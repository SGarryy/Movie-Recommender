IF DB_ID(N'MovieRecommender') IS NULL
BEGIN
    CREATE DATABASE MovieRecommender;
END;
GO

USE MovieRecommender;
GO

IF OBJECT_ID(N'dbo.movies', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.movies (
        movie_id INT NOT NULL PRIMARY KEY,
        title VARCHAR(255) NOT NULL,
        year INT NULL,
        genres VARCHAR(500) NOT NULL DEFAULT ''
    );
END;
GO

IF OBJECT_ID(N'dbo.users', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.users (
        user_id INT NOT NULL PRIMARY KEY,
        gender CHAR(1) NULL,
        age INT NULL,
        occupation INT NULL,
        zip_code VARCHAR(10) NULL
    );
END;
GO

IF OBJECT_ID(N'dbo.ratings', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.ratings (
        rating_id INT IDENTITY(1,1) PRIMARY KEY,
        user_id INT NOT NULL,
        movie_id INT NOT NULL,
        rating DECIMAL(2,1) NOT NULL,
        [timestamp] BIGINT NULL,
        CONSTRAINT CK_ratings_rating_range CHECK (rating >= 0.5 AND rating <= 5.0),
        CONSTRAINT FK_ratings_movies FOREIGN KEY (movie_id) REFERENCES dbo.movies(movie_id)
    );
END;
GO

IF OBJECT_ID(N'dbo.tags', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.tags (
        tag_id INT IDENTITY(1,1) PRIMARY KEY,
        user_id INT NOT NULL,
        movie_id INT NOT NULL,
        tag VARCHAR(500) NOT NULL,
        [timestamp] BIGINT NULL,
        CONSTRAINT FK_tags_movies FOREIGN KEY (movie_id) REFERENCES dbo.movies(movie_id)
    );
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = N'idx_ratings_user_movie'
      AND object_id = OBJECT_ID(N'dbo.ratings')
)
BEGIN
    CREATE UNIQUE INDEX idx_ratings_user_movie ON dbo.ratings(user_id, movie_id);
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = N'idx_ratings_movie'
      AND object_id = OBJECT_ID(N'dbo.ratings')
)
BEGIN
    CREATE INDEX idx_ratings_movie ON dbo.ratings(movie_id);
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = N'idx_tags_movie'
      AND object_id = OBJECT_ID(N'dbo.tags')
)
BEGIN
    CREATE INDEX idx_tags_movie ON dbo.tags(movie_id);
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = N'idx_movies_year'
      AND object_id = OBJECT_ID(N'dbo.movies')
)
BEGIN
    CREATE INDEX idx_movies_year ON dbo.movies(year);
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = N'idx_movies_title'
      AND object_id = OBJECT_ID(N'dbo.movies')
)
BEGIN
    CREATE INDEX idx_movies_title ON dbo.movies(title);
END;
GO
