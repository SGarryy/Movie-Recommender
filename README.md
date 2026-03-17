# 🎬 Movie Recommendation System

A machine learning system built with Python, Scikit-learn, and Microsoft SQL Server using the MovieLens 10M dataset.

## Recommendation Techniques

| Technique | Description |
|---|---|
| Content-Based Filtering | TF-IDF Vectorization + Cosine Similarity on movie metadata |
| Collaborative Filtering | SVD (Truncated) on user-item rating matrix |
| Hybrid Model | Weighted ensemble with cold-start fallback |

## Project Structure
```
movie-recommender/
├── data/
│   ├── raw/               
│   └── processed/         
├── notebooks/             
│   └── 01_EDA.ipynb       
├── src/
│   ├── db/                
│   │   ├── connection.py  
│   │   ├── queries.py     
│   │   ├── load_data.py   
│   │   └── schema.sql     
│   ├── models/            
│   │   ├── content_based.py
│   │   ├── collaborative.py
│   │   └── hybrid.py      
│   ├── preprocessing/     
│   │   ├── cleaner.py     
│   │   └── feature_engineering.py
│   └── recommender.py     
├── app/
│   └── app.py             
├── tests/
│   └── test_models.py     
├── .env.example           
└── requirements.txt       
```

## Tech Stack

- **Language:** Python 3.13
- **Database:** Microsoft SQL Server
- **Libraries:** Pandas, NumPy, Scikit-learn, SciPy, Matplotlib, Seaborn, Streamlit
- **Dataset:** MovieLens 10M — 10M ratings, 10,681 movies, 71,567 users

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/SGarryy/Movie-Recommender.git
cd Movie-Recommender
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt --only-binary=:all:
```

### 4. Configure environment variables
```bash
cp .env.example .env
```

### 5. Set up SQL Server
- Create database and user using `src/db/schema.sql`
- Load data: `python -m src.db.load_data`

### 6. Run the app
```bash
$env:PYTHONPATH = "path/to/movie-recommender"
streamlit run app/app.py
```

## Dataset

MovieLens 10M from [GroupLens](https://grouplens.org/datasets/movielens/10m/) — 10 million ratings across 10,681 movies.

## Tests
```bash
pytest tests/test_models.py -v
```

10 tests covering data loading, cleaning, feature engineering, and recommendation models.

## Security

- Credentials stored in `.env` — never committed
- SQL Server bound to localhost only
- Parameterized queries throughout
- Limited DB user with SELECT/INSERT only

## Author

**Gaurav Singh**
[LinkedIn](https://linkedin.com/in/gauravsingh-ai) 
[GitHub](https://github.com/SGarryy)