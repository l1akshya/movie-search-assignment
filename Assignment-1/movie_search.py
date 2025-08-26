import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset and create embeddings (global for testing)

# Load the Sentence Transformer model

# Convert the 'plot of the movies into an embedding


def search_movies(query, top_n=5):
    #TO_DO logic for implementing