# Movie Recommendation System

A movie recommendation project built with Python, scikit-learn, Streamlit, and Microsoft SQL Server using the MovieLens 10M dataset.

## Recommendation Techniques

| Technique | Description |
| --- | --- |
| Content-Based Filtering | TF-IDF vectorization and cosine similarity on movie metadata |
| Collaborative Filtering | Truncated SVD on the user-item rating matrix |
| Hybrid Model | Combines content and collaborative scores with safe fallbacks |

## Project Structure

```text
movie-recommender/
|-- app/
|   `-- app.py
|-- data/
|   |-- processed/
|   `-- raw/
|-- notebooks/
|   `-- 01_EDA.ipynb
|-- src/
|   |-- db/
|   |   |-- connection.py
|   |   |-- load_data.py
|   |   |-- queries.py
|   |   `-- schema.sql
|   |-- models/
|   |   |-- collaborative.py
|   |   |-- content_based.py
|   |   `-- hybrid.py
|   |-- preprocessing/
|   |   |-- cleaner.py
|   |   `-- feature_engineering.py
|   `-- recommender.py
|-- tests/
|   `-- test_models.py
|-- .env.example
|-- pytest.ini
`-- requirements.txt
```

## Tech Stack

- Python 3.13
- Microsoft SQL Server
- Pandas, NumPy, SciPy, scikit-learn
- Streamlit
- MovieLens 10M

## Setup

### 1. Clone the repo

```powershell
git clone https://github.com/SGarryy/Movie-Recommender.git
cd Movie-Recommender
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt --only-binary=:all:
```

### 4. Configure environment variables

```powershell
Copy-Item .env.example .env
```

Available database settings:

- `DB_SERVER`
- `DB_NAME`
- `DB_DRIVER`
- `DB_USER`
- `DB_PASSWORD`
- `DB_ENCRYPT`
- `DB_TRUST_SERVER_CERTIFICATE`

### 5. Set up SQL Server

- Run `src/db/schema.sql`
- Load data with `python -m src.db.load_data`
- `users.dat` is not included in MovieLens 10M, so user demographics are optional in this project

### 6. Build processed artifacts (optional but recommended)

```powershell
.\.venv\Scripts\python.exe -m src.preprocessing.build_processed
```

This populates `data/processed` with cleaned datasets, a cached TF-IDF matrix, and trained collaborative-filtering artifacts. The app will use them automatically when present and fall back to the database otherwise.

### 7. Run the app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

The automated suite uses deterministic unit tests and does not require a live SQL Server instance.

## Security Notes

- `.env` is ignored by git
- SQLAlchemy hides bound parameters in raised SQL errors
- Database queries are parameterized
- Streamlit renders dataset text without injecting raw HTML into the page
- TLS settings are explicit and configurable through `.env`

## Author

Gaurav Singh

- LinkedIn: https://linkedin.com/in/gauravsingh-ai
- GitHub: https://github.com/SGarryy
