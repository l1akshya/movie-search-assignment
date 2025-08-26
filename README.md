# 🎬 Movie Semantic Search Assignment

This repository contains my solution for **Assignment-1: Semantic Search on Movie Plots**.  
The task was to build a simple semantic search engine using **SentenceTransformers** to find the most relevant movies based on their plots.

---

## 📌 Project Overview
- Loads a dataset of movies (`movies.csv`).
- Generates embeddings for each movie plot using the **all-MiniLM-L6-v2** model.
- Implements a search function `search_movies(query, top_n)` that:
  - Encodes the query into an embedding.
  - Computes cosine similarity between the query and movie plots.
  - Returns the **top N most relevant movies** with similarity scores.

---

## 🛠️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/movie-search-assignment.git
cd movie-search-assignment/Assignment-1


2 create a virtual enviorment 

python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

3 Install Dependencies 

pip install -r requirements.txt

4 Repository Structure 

movie-search-assignment/
├── .github/workflows/python-tests.yml   # GitHub Actions workflow
├── Assignment-1/
│   ├── movie_search.py                  # Main implementation
│   ├── movies.csv                       # Dataset
│   ├── requirements.txt                 # Dependencies
│   └── tests/
│       └── test_movie_search.py         # Unit tests
├── README.md



