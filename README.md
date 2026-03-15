# 🎬 Movie Recommendation System

A machine learning project built with Python, Scikit-learn, and Microsoft SQL Server — 
using the MovieLens 10M dataset to deliver personalized movie recommendations.

---

## 🧠 Recommendation Techniques

| Technique | Description |
|---|---|
| Content-Based Filtering | TF-IDF Vectorization + Cosine Similarity on movie metadata |
| Collaborative Filtering | SVD (Truncated) + KNN on user-item rating matrix |
| Hybrid Model | Weighted ensemble of both approaches with cold-start fallback |

---

## 🗂️ Project Structure
```
movie-recommender/
├── data/
│   ├── raw/               ← MovieLens 10M + TMDB (gitignored)
│   └── processed/         ← Cleaned data (gitignored)
├── notebooks/             ← EDA + experimentation
├── src/
│   ├── db/                ← SQL Server connection + queries
│   ├── models/            ← ML models
│   ├── preprocessing/     ← Data cleaning + feature engineering
│   └── recommender.py     ← Main recommendation interface
├── app/
│   └── app.py             ← Streamlit frontend
├── tests/                 ← Unit tests
├── .env.example           ← Environment variable template
└── requirements.txt       ← Pinned dependencies
```

---

## ⚙️ Tech Stack

- **Language:** Python 3.10+
- **Database:** Microsoft SQL Server
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Streamlit
- **Dataset:** [MovieLens 10M](https://grouplens.org/datasets/movielens/10m/) + 
[TMDB Metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

---

## 🚀 Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/movie-recommender.git
cd movie-recommender
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Fill in your SQL Server credentials in .env
```

### 5. Set up database
```bash
# Run schema in SQL Server Management Studio
# Then load data:
python src/db/load_data.py
```

### 6. Run the app
```bash
streamlit run app/app.py
```

---

## 📊 Dataset

- **MovieLens 10M** — 10 million ratings, 100K tags, 10,681 movies, 71,567 users
- **TMDB Metadata** — Movie overviews, genres, cast for enriched content features

---

## 🔐 Security

- All credentials stored in `.env` (never committed)
- SQL Server bound to localhost only
- Parameterized queries throughout — no SQL injection risk
- Limited DB user with SELECT/INSERT only

---

## 📈 Model Evaluation

| Model | Metric | Score |
|---|---|---|
| Content-Based | Precision@10 | TBD |
| Collaborative (SVD) | RMSE | TBD |
| Hybrid | Precision@10 | TBD |

*Scores updated after training*

---

## 👤 Author

**Gaurav Singh**  
[LinkedIn](https://linkedin.com/in/gauravsingh-ai) • [GitHub](https://github.com/SGarryy)